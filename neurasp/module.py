import functools
import itertools
from operator import mul
from typing import Mapping, Sequence

import torch
from torch import nn, Tensor


class NeurASPModule(nn.Module):
    """The compiled program as a differentiable PyTorch module.

    Holds every stable model as the input atoms it selects, grouped by output. Given a
    batch of atom probabilities, an output's probability is the sum over its models of
    the product of their atoms' probabilities. The module never calls Clingo.

    The models are stored once, unpadded, grouped by output (models), with where each
    output's models start (model_starts) and how many there are (model_counts).
    forward() builds its projection from the counts on each call (see output).
    """

    models: Tensor
    model_starts: Tensor
    model_counts: Tensor

    def __init__(
        self, models: Mapping[int, Tensor | Sequence[Tensor]], sparse: bool = True
    ) -> None:
        """Compile stable models into the module's buffers.

        Args:
            models (Mapping[int, Tensor | Sequence[Tensor]]): output column -> the
                stable models with that output, each a row of input-column positions,
                one per categorical input: either one (models, inputs) int matrix, as
                ModuleGenerator returns, or a sequence of one int tensor per model. The
                keys must be 0, 1, ..., n - 1 in that order. There must be at least one
                model overall; an output with no models gets a zero-probability column.
            sparse (bool): build forward()'s projection matrix as a sparse COO tensor
                (the default) rather than a dense one. Both give the same result. The
                projection holds one nonzero per model, so the sparse layout stores the
                models rather than models * outputs, and measured faster in forward() at
                mnistAdd's size. Pass False to fall back to the dense matrix.
        """
        super(NeurASPModule, self).__init__()
        assert any(
            len(output_models) > 0 for output_models in models.values()
        ), "models must hold at least one stable model (an output may have none)"
        index_tensor, model_starts, model_counts = NeurASPModule._build_matrices(models)
        # Buffers rather than plain attributes so that .to(device) moves them with the module.
        self.register_buffer("models", index_tensor)
        self.register_buffer("model_starts", model_starts)
        self.register_buffer("model_counts", model_counts)
        self.sparse = sparse
        # Sizes kept on the host, so that the forward passes never read them back from
        # the device: the number of stable models, and the most models of any output.
        self.model_count = len(index_tensor)
        self.max_model_count = int(model_counts.max())
        # Each output's model count on the host, for model_total.
        self.host_model_counts = model_counts.numpy()

    @property
    def output(self) -> Tensor:
        """The projection from models to outputs, (models, outputs), with a 1 where a
        model has an output: sparse COO, or dense if the module was built with
        sparse=False. Built on each access, as forward() builds it, since training never
        uses it.
        """
        return self._projection()

    def forward(self, input: Tensor) -> Tensor:
        """Probability of every output: for each, the sum over its models of the product
        of their atoms' probabilities.

        Args:
            input (Tensor): a batch of atom probabilities, of shape (batch, atoms).
        Returns:
            Tensor: the probability of each output, of shape (batch, outputs).
        """
        assert (
            input.dim() == 2
        ), f"expected input of shape (batch, atoms), got {tuple(input.shape)}"
        # (batch, models, atoms_per_model) -> (batch, models)
        model_probabilities = NeurASPModule._product(input[:, self.models])
        # (batch, models) @ (models, outputs) -> (batch, outputs). The projection is
        # built here rather than kept, since training never uses it.
        return model_probabilities @ self._projection()

    def model_total(self, outputs: Tensor) -> int:
        """The number of models of all of outputs together, read on the host: the
        model_total to pass to forward_single_output for a batch whose expected outputs
        are outputs.

        Args:
            outputs (Tensor): outputs (keys of models), on the CPU, such as a batch's
                labels before they are copied to the device; any shape.
        Returns:
            int: the sum of the outputs' model counts.
        """
        assert outputs.device.type == "cpu", "model_total reads the outputs on the CPU"
        return int(self.host_model_counts[outputs.numpy()].sum())

    def forward_single_output(
        self,
        input: Tensor,
        expected_outputs: Tensor,
        model_total: int | None = None,
    ) -> Tensor:
        """Probability of one given output per batch element, evaluating only that
        output's models; cheaper than forward() when only the observed output matters.

        Args:
            input (Tensor): a batch of atom probabilities, of shape (batch, atoms).
            expected_outputs (Tensor): one output (a key of models) per batch element,
                of shape (batch,), int64. Best passed on the CPU, as a DataLoader gives
                them: model_total is then read from them there, and they are copied to
                input's device (non-blocking) here. On input's device they are used as
                they are.
            model_total (int | None): the number of models of all the expected outputs
                together, exactly (see model_total, which reads it from the outputs on
                the CPU without waiting for the device). Given, or read from CPU
                expected_outputs, the batch's models are evaluated as one flat list,
                with no padding; a wrong value gives wrong results. Without it, for
                expected_outputs already on another device, each element's models are
                padded to max_model_count, the most of any output, which on card sum_4
                made the pass 2.9 times slower.
        Returns:
            Tensor: of shape (batch,), the probability of each element's expected
                output: forward()[b, expected_outputs[b]] for element b.
        """
        assert (
            input.dim() == 2
        ), f"expected input of shape (batch, atoms), got {tuple(input.shape)}"
        assert expected_outputs.shape == (
            input.size(0),
        ), f"expected expected_outputs of shape ({input.size(0)},), got {tuple(expected_outputs.shape)}"
        if expected_outputs.device.type == "cpu":
            # Reading the total on the CPU costs no sync with the device.
            if model_total is None:
                model_total = self.model_total(expected_outputs)
            expected_outputs = expected_outputs.to(input.device, non_blocking=True)
        starts = self.model_starts[expected_outputs]
        counts = self.model_counts[expected_outputs]
        if model_total is None:
            return self._padded_single_output(input, starts, counts)
        # The batch's models one after another: element b's are models starts[b] to
        # starts[b] + counts[b] - 1, and element[i] says whose the i-th one is. The
        # total length comes from the host: computing it here would have to copy it
        # from the device, forcing a GPU->CPU sync on every call. (model_total,)
        batch = torch.arange(input.size(0), device=input.device)
        element = torch.repeat_interleave(batch, counts, output_size=model_total)
        first = counts.cumsum(0) - counts
        position = torch.arange(model_total, device=input.device) - first[element]
        models = self.models[starts[element] + position]
        # Row element[i] of input indexed by model i, then each element's sum:
        # (model_total, atoms_per_model) -> (model_total,) -> (batch,)
        products = NeurASPModule._product(input[element.unsqueeze(1), models])
        return products.new_zeros(input.size(0)).index_add_(0, element, products)

    def _padded_single_output(
        self, input: Tensor, starts: Tensor, counts: Tensor
    ) -> Tensor:
        """forward_single_output without model_total: each element's models, those
        from starts to starts + counts - 1, padded to max_model_count.

        Position p of element b is model starts[b] + p if p < counts[b], and padding
        otherwise, which uses model 0 and is masked out. The length is max_model_count,
        not the batch's largest count, which would have to be copied from the device.
        """
        positions = torch.arange(self.max_model_count, device=input.device)
        mask = positions < counts.unsqueeze(1)
        models = self.models[torch.where(mask, starts.unsqueeze(1) + positions, 0)]
        # Row b of input indexed by the models of element b:
        # (batch, max_models, atoms_per_model) -> (batch, max_models) -> (batch,)
        batch = torch.arange(input.size(0), device=input.device).view(-1, 1, 1)
        products = NeurASPModule._product(input[batch, models])
        return (products * mask).sum(dim=1)

    def _projection(self) -> Tensor:
        """The projection matrix, (models, outputs), with a 1 in row m at the column of
        model m's output: sparse COO or dense, as chosen in __init__.

        Built as the (outputs, models) matrix and transposed, so that forward()'s
        matrix product is the one measured. The models are grouped by output, so model
        m's output repeats each output its count times; model_count gives that length
        on the host, so this never waits for the GPU.
        """
        device = self.model_counts.device
        shape = (len(self.model_counts), self.model_count)
        outputs = torch.repeat_interleave(
            torch.arange(shape[0], device=device),
            self.model_counts,
            output_size=self.model_count,
        )
        columns = torch.arange(self.model_count, device=device)
        ones = torch.ones(self.model_count, device=device)
        if not self.sparse:
            projection = torch.zeros(shape, device=device)
            projection[outputs, columns] = ones
            return projection.t()
        # One nonzero per column, sorted by row and then column: already coalesced.
        # The indices are valid by construction, so the invariant checks are off;
        # saying so explicitly also silences PyTorch's warning.
        return torch.sparse_coo_tensor(
            torch.stack((outputs, columns)),
            ones,
            shape,
            check_invariants=False,
            is_coalesced=True,
        ).t()

    @staticmethod
    def _product(atom_probabilities: Tensor) -> Tensor:
        """Product over the last dimension, (..., atoms_per_model) -> (...). Same as
        .prod(dim=-1), but prod's CUDA backward checks its input for zeros, which forces
        a GPU->CPU sync, and atom probabilities can be zero. Elementwise multiplications
        of the atoms_per_model columns have a sync-free backward.
        """
        return functools.reduce(mul, atom_probabilities.unbind(dim=-1))

    @staticmethod
    def _build_matrices(
        models: Mapping[int, Tensor | Sequence[Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the module's buffers from models (see __init__).

        Returns:
            The index tensor, (models, atoms_per_model) int32, of every model grouped
            by output in output order; and, per output, the int64 position in it of the
            output's first model and the number of its models, both (outputs,).
        """
        assert list(models) == list(
            range(len(models))
        ), "output keys must be 0, 1, ..., n - 1 in that order"
        # One (models, atoms_per_model) matrix per output, in output order.
        blocks = NeurASPModule._model_matrices(models)
        counts = torch.tensor([len(block) for block in blocks], dtype=torch.long)
        starts = counts.cumsum(0) - counts
        return torch.cat(blocks), starts, counts

    @staticmethod
    def _model_matrices(
        models: Mapping[int, Tensor | Sequence[Tensor]],
    ) -> list[Tensor]:
        """Each output's models as one int32 matrix, (models, atoms_per_model), in
        output order. PyTorch indexes with int32 as with int64, and the indices are
        input columns (208 for four cards), so int32 halves the index buffers.

        An output's models may already be such a matrix, as ModuleGenerator returns
        them, which is used as it is; or a sequence of one index tensor per model,
        which is stacked. An output with no models gets a (0, atoms_per_model) matrix.
        """
        matrices: list[Tensor | None] = []
        for output_models in models.values():
            if isinstance(output_models, Tensor):
                matrices.append(output_models.int())
            elif len(output_models) > 0:
                matrices.append(torch.stack(list(output_models)).int())
            else:
                matrices.append(None)
        width = next(m.size(1) for m in matrices if m is not None and len(m) > 0)
        return [
            (
                m
                if m is not None and len(m) > 0
                else torch.empty(0, width, dtype=torch.int32)
            )
            for m in matrices
        ]


def semantic_loss(probabilities: Tensor) -> Tensor:
    """NeurASP's semantic loss: the mean of -log P(observation) over a batch, given each
    element's probability of its observed output, as forward_single_output returns.

    It equals NLLLoss on the log of the full output distribution with the observed
    outputs as targets, without computing that distribution. Probabilities are clamped
    to 1e-12, so an impossible observation gives a large but finite loss.
    """
    return -probabilities.clamp_min(1e-12).log().mean()


class NeurASPProgram(nn.Module):
    """Neural networks followed by a NeurASPModule: the networks turn raw inputs
    (images) into atom probabilities, and the module turns those into output
    probabilities.
    """

    def __init__(
        self, networks: list[tuple[nn.Module, int]], module: NeurASPModule
    ) -> None:
        """Combine neural networks with a compiled NeurASPModule.

        Args:
            networks (list[tuple[nn.Module, int]]): (network, count) pairs; the network
                is applied to count consecutive inputs, so [(network, 2)] shares one
                network between two inputs.
            module (NeurASPModule): takes the networks' outputs, concatenated in input
                order.
        """
        super(NeurASPProgram, self).__init__()
        assert all(
            count > 0 for _, count in networks
        ), "every network must be used by at least one input"
        # ModuleList rather than a plain list so the networks are registered submodules and
        # show up in .parameters(), .to(device) and .train()/.eval().
        self.networks = nn.ModuleList(network for network, _ in networks)
        self.network_input_count = [count for _, count in networks]
        # Index in forward()'s inputs of each network's first input.
        self.network_input_starts = list(
            itertools.accumulate(self.network_input_count, initial=0)
        )[:-1]
        self.module = module

    def forward(
        self,
        inputs: list[Tensor],
        *,
        expected_outputs: Tensor | None = None,
        model_total: int | None = None,
    ) -> Tensor:
        """Run the networks on their inputs and the module on their outputs.

        Args:
            inputs (list[Tensor]): one input per network use, sum of the counts in all,
                in the order of the (network, count) pairs, all with the same batch
                size. Each network is called once on its inputs concatenated along the
                batch dimension, so layers with batch statistics (BatchNorm) see all of
                them together.
            expected_outputs (Tensor | None): if given, one output per batch element,
                evaluated alone with NeurASPModule.forward_single_output; best on the
                CPU, as a DataLoader gives them (see there).
            model_total (int | None): with expected_outputs, passed on to
                forward_single_output (see NeurASPModule.model_total).
        Returns:
            Tensor: the output probabilities, of shape (batch, outputs); with
                expected_outputs, only the expected output's, of shape (batch,).
        """
        assert len(inputs) == sum(
            self.network_input_count
        ), f"expected {sum(self.network_input_count)} inputs, got {len(inputs)}"
        batch_size = inputs[0].size(0)
        network_outputs: list[Tensor] = []
        for network, start, count in zip(
            self.networks, self.network_input_starts, self.network_input_count
        ):
            # One call per network: its inputs stacked along the batch dimension,
            # (count * batch, ...), then split back into one output per input.
            output = network(torch.cat(inputs[start : start + count]))
            network_outputs.extend(output.split(batch_size))
        input = torch.cat(network_outputs, dim=1)
        if expected_outputs is None:
            return self.module(input)
        return self.module.forward_single_output(input, expected_outputs, model_total)
