import torch
import torch.nn.functional as F

def spec_rmse_loss(estimate, sources, stft_config):

    _, _, _, lenc = estimate.shape
    spec_estimate = estimate.view(-1, lenc)
    spec_sources = sources.view(-1, lenc)

    spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)


    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
    spec_estimate = spec_estimate.view(*new_shape)
    spec_sources = spec_sources.view(*new_shape)


    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')


    dims = tuple(range(2, loss.dim()))
    loss = loss.mean(dims).sqrt().mean(dim=(0, 1))  

    return loss


# ------------------------------------------------------------
# Masked variant: only compute loss on active sources defined by `mask`.
# `mask` shape: (batch, sources) with 1 for active, 0 for absent.
# This avoids penalising the model for stems that are not present in the mix.
def spec_rmse_loss_masked(estimate, sources, stft_config, mask, absent_weight=0.05, eps=1e-8):
    """RMSE on complex STFT magnitude, averaged only over active sources.

    Args:
        estimate (Tensor): (B, S, C, T) separated waveforms
        sources (Tensor):  same shape, ground-truth stems
        stft_config (dict): kwargs to torch.stft
        mask (Tensor): (B, S) binary tensor, 1 if source present in mixture
    Returns:
        torch.Tensor: scalar loss averaged over active (B,S) positions
    """
    B, S, C, L = estimate.shape

    est_flat = estimate.view(-1, L)  # (B*S*C, L) after view but C kept inside L? Wait dims. We'll treat C in shape.
    src_flat = sources.view(-1, L)

    spec_est = torch.stft(est_flat, **stft_config, return_complex=True)
    spec_src = torch.stft(src_flat, **stft_config, return_complex=True)

    spec_est = torch.view_as_real(spec_est)
    spec_src = torch.view_as_real(spec_src)

    new_shape = estimate.shape[:-1] + spec_est.shape[-3:]
    spec_est = spec_est.view(*new_shape)  # (B, S, C, F, T, 2)
    spec_src = spec_src.view(*new_shape)

    loss = F.mse_loss(spec_est, spec_src, reduction='none')

    dims = tuple(range(2, loss.dim()))  # average over (C, F, T, 2)
    loss = (loss.mean(dims) + eps).sqrt()  # (B, S)

    # Build weights: present → 1, absent → absent_weight
    weights = mask + absent_weight * (1 - mask)
    masked_loss = loss * weights
    denom = weights.sum()
    if denom < 1:
        # No active sources in batch (should not happen), default to zero to avoid NaN
        return masked_loss.sum() * 0.0
    return masked_loss.sum() / denom

