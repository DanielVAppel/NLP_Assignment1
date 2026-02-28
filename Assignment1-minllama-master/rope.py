from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(q, k, head_dim, max_seq_len, base=10000.0):
    device = q.device
    bs, seqlen, _, _ = q.shape

    pos = torch.arange(seqlen, device=device, dtype=torch.float32)

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    freqs = torch.einsum("s,d->sd", pos, inv_freq)  # (seqlen, head_dim/2)

    # IMPORTANT: broadcast against q_even (head_dim/2), not q (head_dim)
    q_even_ref = q[..., 0::2]
    cos = reshape_for_broadcast(freqs.cos(), q_even_ref)
    sin = reshape_for_broadcast(freqs.sin(), q_even_ref)

    q_even = q[..., 0::2]
    q_odd  = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd  = k[..., 1::2]

    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd  = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd  = k_even * sin + k_odd * cos

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_out[..., 0::2] = q_rot_even
    q_out[..., 1::2] = q_rot_odd
    k_out[..., 0::2] = k_rot_even
    k_out[..., 1::2] = k_rot_odd

    return q_out, k_out