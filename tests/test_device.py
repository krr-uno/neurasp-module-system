"""Device selection and tiny CPU/CUDA probability and gradient checks."""

from contextlib import nullcontext
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from neurasp.device import autocast, memory_format, resolve_device, synchronize
from neurasp.module import NeurASPModule, NeurASPProgram


def binary_addition_models() -> dict[int, list[torch.Tensor]]:
    """Models of the sum of two binary digits, inputs [a0, a1, b0, b1]."""
    return {
        0: [torch.tensor([0, 2])],
        1: [torch.tensor([0, 3]), torch.tensor([1, 2])],
        2: [torch.tensor([1, 3])],
    }


class DeviceSelectionTests(unittest.TestCase):
    """Device resolution and synchronization, without needing a GPU."""

    def test_cpu_does_not_require_cuda_and_gpu_failure_is_explicit(self) -> None:
        """CPU works without CUDA; cuda and gpu fail loudly without it; synchronize does
        nothing on CPU.
        """
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
            for name in ("cuda", "gpu"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    resolve_device(name)
        with patch("torch.cuda.synchronize") as cuda_sync:
            synchronize(torch.device("cpu"))
        cuda_sync.assert_not_called()

    def test_gpu_alias_and_invalid_device(self) -> None:
        """gpu is an alias for cuda; an unknown device name is rejected."""
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("gpu"), torch.device("cuda"))
            self.assertEqual(resolve_device("cuda"), torch.device("cuda"))
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_reduced_precision_needs_cuda_and_a_known_name(self) -> None:
        """tf32 and bf16 are rejected on the CPU, as is an unknown precision; autocast
        only acts for bf16 on CUDA.
        """
        for precision in ("tf32", "bf16"):
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(ValueError, "needs a CUDA device"):
                    resolve_device("cpu", precision)
        with self.assertRaisesRegex(ValueError, "precision must be one of"):
            resolve_device("cpu", "fp8")
        cpu, cuda = torch.device("cpu"), torch.device("cuda")
        for device, precision in [(cpu, "ieee"), (cpu, "bf16"), (cuda, "tf32")]:
            with self.subTest(device=device, precision=precision):
                self.assertIsInstance(autocast(device, precision), nullcontext)
        self.assertIsInstance(autocast(cuda, "bf16"), torch.autocast)

    def test_channels_last_only_for_bf16_on_cuda(self) -> None:
        """memory_format gives channels_last for bf16 on CUDA and the default layout
        otherwise; a network moved to it keeps its outputs (checked on the CPU).
        """
        cpu, cuda = torch.device("cpu"), torch.device("cuda")
        self.assertEqual(memory_format(cuda, "bf16"), torch.channels_last)
        for device, precision in [(cpu, "ieee"), (cpu, "bf16"), (cuda, "ieee")]:
            with self.subTest(device=device, precision=precision):
                self.assertEqual(
                    memory_format(device, precision), torch.contiguous_format
                )
        convolution = nn.Conv2d(3, 4, 3)
        network = nn.Sequential(convolution, nn.Flatten(), nn.Linear(64, 2))
        images = torch.rand(2, 3, 6, 6)
        expected = network(images)
        network.to(memory_format=memory_format(cuda, "bf16"))
        self.assertTrue(
            convolution.weight.is_contiguous(memory_format=torch.channels_last)
        )
        torch.testing.assert_close(network(images), expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaParityTests(unittest.TestCase):
    """CPU and CUDA give the same results; skipped without CUDA."""

    def test_dense_and_sparse_outputs_input_gradients_and_buffers(self) -> None:
        """On CUDA, both layouts give the CPU outputs and input gradients, and all
        buffers move to and from the GPU; the projection that forward() builds is on
        the module's device.
        """
        probabilities = torch.tensor([[0.8, 0.2, 0.4, 0.6], [0.35, 0.65, 0.7, 0.3]])
        expected = torch.tensor([[0.32, 0.56, 0.12], [0.245, 0.56, 0.195]])
        weights = torch.tensor([[1.1, 0.7, -0.3], [0.2, -0.4, 0.8]])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                cpu = NeurASPModule(binary_addition_models(), sparse=sparse)
                gpu = copy.deepcopy(cpu).to("cuda")
                self.assertEqual(
                    set(dict(gpu.named_buffers())),
                    {"models", "model_starts", "model_counts"},
                )
                self.assertTrue(
                    all(buffer.device.type == "cuda" for buffer in gpu.buffers())
                )
                self.assertEqual(gpu.output.layout, cpu.output.layout)
                self.assertEqual(gpu.output.device.type, "cuda")
                torch.testing.assert_close(gpu.models.cpu(), cpu.models)
                cpu_input = probabilities.clone().requires_grad_()
                gpu_input = probabilities.to("cuda").requires_grad_()
                cpu_output, gpu_output = cpu(cpu_input), gpu(gpu_input)
                torch.testing.assert_close(cpu_output, expected)
                torch.testing.assert_close(gpu_output.cpu(), cpu_output)
                (cpu_output * weights).sum().backward()
                (gpu_output * weights.to("cuda")).sum().backward()
                assert gpu_input.grad is not None
                torch.testing.assert_close(gpu_input.grad.cpu(), cpu_input.grad)
                # The buffers must move back too.
                gpu.to("cpu")
                self.assertTrue(
                    all(buffer.device.type == "cpu" for buffer in gpu.buffers())
                )
                torch.testing.assert_close(gpu(probabilities), expected)

    def test_program_keeps_shared_network_and_matches_parameter_gradients(self) -> None:
        """A program with a shared network moves to CUDA with one set of parameters and
        matches the CPU parameter and input gradients.
        """
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                linear = nn.Linear(3, 2)
                network = nn.Sequential(linear, nn.Softmax(dim=1))
                with torch.no_grad():
                    linear.weight.copy_(
                        torch.tensor([[0.2, -0.3, 0.5], [-0.4, 0.7, 0.1]])
                    )
                    linear.bias.copy_(torch.tensor([0.1, -0.2]))
                cpu = NeurASPProgram(
                    [(network, 2)],
                    NeurASPModule(binary_addition_models(), sparse=sparse),
                )
                gpu = copy.deepcopy(cpu).to("cuda")
                self.assertEqual(len(gpu.networks), 1)
                self.assertEqual(len(list(gpu.parameters())), 2)
                self.assertTrue(
                    all(
                        parameter.device.type == "cuda"
                        for parameter in gpu.parameters()
                    )
                )
                self.assertTrue(
                    all(buffer.device.type == "cuda" for buffer in gpu.buffers())
                )
                cpu_inputs = [
                    torch.tensor(
                        [[0.1, 0.2, -0.4], [0.6, -0.5, 0.3]], requires_grad=True
                    ),
                    torch.tensor(
                        [[0.7, -0.2, 0.1], [-0.3, 0.8, 0.4]], requires_grad=True
                    ),
                ]
                gpu_inputs = [
                    value.detach().to("cuda").requires_grad_() for value in cpu_inputs
                ]
                weights = torch.tensor([[1.1, 0.7, -0.3], [0.2, -0.4, 0.8]])
                cpu_output, gpu_output = cpu(cpu_inputs), gpu(gpu_inputs)
                torch.testing.assert_close(gpu_output.cpu(), cpu_output)
                (cpu_output * weights).sum().backward()
                (gpu_output * weights.to("cuda")).sum().backward()
                for cpu_parameter, gpu_parameter in zip(
                    cpu.parameters(), gpu.parameters()
                ):
                    assert cpu_parameter.grad is not None
                    assert gpu_parameter.grad is not None
                    self.assertGreater(cpu_parameter.grad.abs().sum().item(), 0)
                    torch.testing.assert_close(
                        gpu_parameter.grad.cpu(),
                        cpu_parameter.grad,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                for cpu_input, gpu_input in zip(cpu_inputs, gpu_inputs):
                    assert gpu_input.grad is not None
                    torch.testing.assert_close(
                        gpu_input.grad.cpu(), cpu_input.grad, rtol=1e-5, atol=1e-6
                    )


if __name__ == "__main__":
    unittest.main()
