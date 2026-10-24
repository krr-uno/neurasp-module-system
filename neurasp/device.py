"""Shared device selection, precision and timing for the native pipeline examples."""

from contextlib import AbstractContextManager, nullcontext

import torch

# Arithmetic precision on CUDA: "ieee" full float32 (the default, comparable with CPU
# runs); "tf32" lets float32 matrix products and convolutions use TF32; "bf16" also
# runs the forward passes under bfloat16 autocast (see autocast) with convolution
# networks in the channels_last layout (see memory_format).
PRECISIONS = ("ieee", "tf32", "bf16")


def resolve_device(name: str, precision: str = "ieee") -> torch.device:
    """Resolve the CLI device and set its float32 precision, refusing to silently fall
    back from GPU to CPU. Raises ValueError for an unknown device or precision, or a
    precision other than ieee without CUDA.
    """
    if name not in ("cpu", "cuda", "gpu"):
        raise ValueError("device must be cpu, cuda, or gpu")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")
    name = "cuda" if name == "gpu" else name
    if name != "cuda" and precision != "ieee":
        raise ValueError(f"precision {precision} needs a CUDA device")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("cuda not found")
        # By default full float32, including convolutions, so CPU and GPU runs agree.
        fp32_precision = "tf32" if precision == "tf32" else "ieee"
        torch.backends.cuda.matmul.fp32_precision = fp32_precision
        # cudnn.conv is added at runtime (CudnnModule), so the stubs do not list it.
        torch.backends.cudnn.conv.fp32_precision = fp32_precision  # type: ignore[attr-defined]
    return torch.device(name)


def autocast(device: torch.device, precision: str) -> AbstractContextManager[object]:
    """A context for a training step's forward pass and loss: bfloat16 autocast on
    CUDA for precision bf16, which keeps precision-sensitive operations (softmax,
    losses, the NeurASP products) in float32; otherwise it does nothing. Run the
    backward pass outside it.
    """
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def memory_format(device: torch.device, precision: str) -> torch.memory_format:
    """The memory layout for a network's weights: channels_last on CUDA for precision
    bf16, otherwise the default contiguous layout. Pass it to network.to(...,
    memory_format=...); it changes how 4-dimensional tensors (convolution weights) are
    stored, not their shapes or values.

    Under bf16 autocast cuDNN runs convolutions on Tensor Cores, whose kernels want
    each pixel's channels adjacent; in the default layout every convolution transposes
    its data and back. cuDNN uses channels_last when the weights do, so the inputs
    need no conversion. Measured on card sum_3 (batch 16): 67 to 53 ms per step. In
    full float32 it gave nothing, so it is used only with bf16.
    """
    if precision == "bf16" and device.type == "cuda":
        return torch.channels_last
    return torch.contiguous_format


def synchronize(device: torch.device) -> None:
    """Wait at timing boundaries so asynchronous GPU work is included."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_metadata(device: torch.device, precision: str = "ieee") -> dict[str, object]:
    """Describe device for the metrics files: its name, the CUDA version and compute
    capability, the precision and the float32 precision used.
    """
    name, capability = "CPU", None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        name, capability = properties.name, f"{properties.major}.{properties.minor}"
    return {
        "device": str(device),
        "device_name": name,
        "cuda_version": torch.version.cuda,
        "cuda_compute_capability": capability,
        "precision": precision,
        "fp32_precision": "tf32" if precision == "tf32" else "ieee",
    }
