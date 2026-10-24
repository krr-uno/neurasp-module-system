"""Explicit device selection for the original NeurASP examples."""
import torch


def resolve_device(name):
    if name not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    if name == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise ValueError('GPU requested but CUDA is unavailable. Activate the neurasp5-gpu environment.')
    # Full float32 for comparison with the pipeline; no mixed precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
