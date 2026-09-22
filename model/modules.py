"""SpikeRain building blocks mapped to WACV 2026 paper sections.

This module groups the core architectural components described in the paper:
- Section 3.2: Dense Spiking Residual Block (DSRB)
- Section 3.3: Multi Dimensional Spiking Attention (MDSA)
- Section 3.4: Temporal Fusion (TF)
- Section 3.5: Adaptive Residual Feature Enhancement (ARFE)

Utility layers used by the main network (overlap embedding and sampling) are
kept here for clarity and reuse.
"""

from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn
import torch.nn.functional as F

v_th = 0.15


class ThresholdDependentBatchNorm2d(nn.Module):
    """BatchNorm2d wrapper that supports 4D and 5D spike tensors.

    Inputs:
        x: (B, C, H, W) or (T, B, C, H, W)
    Outputs:
        Tensor with the same shape as x.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch norm over 4D/5D inputs."""
        if x.dim() == 5:
            return functional.seq_to_ann_forward(x, self.bn)
        if x.dim() == 4:
            return self.bn(x)
        raise ValueError(f"Unexpected input shape: {x.shape}")


# ----------------------------------------------------------------------------
# Shared utility layers (used across sections)
# ----------------------------------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """Overlap patch embedding used at the network input.

    Inputs:
        x: (T, B, C, H, W) or (B, C, H, W)
    Outputs:
        Tensor with embed_dim channels and matching spatial dimensions.
    """

    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif", LayerNorm_type='WithBias', bias=False):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """Project the input into the embedding space."""
        x = self.proj(x)
        return x


class DownSampling(nn.Module):
    """Downsample by 2x with spiking activation and convolution.

    Inputs:
        x: (T, B, C, H, W)
    Outputs:
        Tensor with 2x channels and half spatial resolution.
    """

    def __init__(self, dim):
        super(DownSampling, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.maxpool_conv = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode='m', bias=False),
            ThresholdDependentBatchNorm2d(dim * 2),
        )

    def forward(self, x):
        """Apply spiking downsampling."""
        return self.maxpool_conv(x)


class UpSampling(nn.Module):
    """Upsample by 2x with bilinear interpolation and spiking convolution.

    Inputs:
        input: (T, B, C, H, W)
    Outputs:
        Tensor with half channels and doubled spatial resolution.
    """

    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode='m', bias=False),
            ThresholdDependentBatchNorm2d(dim // 2),
        )

    def forward(self, input):
        """Upsample each timestep and apply spiking convolution."""
        # Allocated on the input's own device/dtype so the block also runs on
        # CPU (needed by model_complexity.py --gpu_ids -1); behaviour on GPU is
        # unchanged.
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor),
                           device=input.device, dtype=input.dtype)
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)


# ----------------------------------------------------------------------------
# Section 3.2: Dense Spiking Residual Block (DSRB)
# ----------------------------------------------------------------------------
class DSRB(nn.Module):
    """Dense Spiking Residual Block with local feature fusion.

    Inputs:
        x: (T, B, C, H, W)
    Outputs:
        Tensor with the same shape as x.

    Key steps:
        - Dense spiking convolutions
        - Local feature fusion via 1x1 projection
        - MDSA-enhanced residual connection
    """

    def __init__(self, dim, growth_rate=24, num_layers=4):
        super(DSRB, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.in_channels = dim
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        channels = dim

        for _ in range(num_layers):
            layer_i = nn.Sequential(
                LIFNode(v_threshold=0.15, backend='torch', step_mode='m', decay_input=False),
                layer.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False, step_mode='m'),
                ThresholdDependentBatchNorm2d(growth_rate),
            )
            self.layers.append(layer_i)
            channels += growth_rate

        self.lff = nn.Sequential(
            LIFNode(v_threshold=0.15, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(channels, dim, kernel_size=1, bias=False, step_mode='m'),
        )
        self.attn = MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x):
        """Apply dense spiking layers, fuse, then add MDSA residual."""
        inputs = [x]
        for layer_i in self.layers:
            out = layer_i(torch.cat(inputs, dim=2))
            inputs.append(out)

        dense_out = torch.cat(inputs, dim=2)
        out = self.lff(dense_out)
        out = self.attn(out) + x
        return out


# ----------------------------------------------------------------------------
# Section 3.3: Multi Dimensional Spiking Attention (MDSA)
# ----------------------------------------------------------------------------
class MultiDimensionalAttention(nn.Module):
    """Multi-dimensional attention over temporal, channel, and spatial axes.

    Inputs:
        x: (T, B, C, H, W)
    Outputs:
        Tensor with the same shape as x.
    """

    def __init__(self, T, C, reduction_t=4, reduction_c=16, kernel_size=3):
        super().__init__()
        self.T = T
        self.C = C

        self.temporal_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(C, C // reduction_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(C // reduction_c, C, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size // 2, kernel_size // 2)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Compute temporal, channel, and spatial attention and reweight x."""
        T, B, C, H, W = x.shape
        x_perm = x.permute(1, 2, 0, 3, 4).contiguous()

        temp_att = self.temporal_fc(x_perm.mean(1, keepdim=True))
        x_temp = x_perm * temp_att

        chn_att = self.channel_fc(x_temp)
        x_chn = x_temp * chn_att

        spatial_pool = x_chn.mean(1, keepdim=True)
        spatial_att = self.spatial_conv(spatial_pool)
        x_spatial = x_chn * spatial_att

        out = x_spatial.permute(2, 0, 1, 3, 4).contiguous()
        return out


# ----------------------------------------------------------------------------
# Section 3.4: Temporal Fusion (TF)
# ----------------------------------------------------------------------------
class TemporalFusion(nn.Module):
    """Weighted fusion over T timesteps.

    Inputs:
        x: (T, B, C, H, W)
    Outputs:
        Tensor of shape (B, C, H, W).
    """

    def __init__(self, T):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(T) / T)

    def forward(self, x):
        """Fuse temporal features with learned softmax weights."""
        weights = F.softmax(self.weights, dim=0)[:, None, None, None, None]
        fused = (x * weights).sum(0)
        return fused


# ----------------------------------------------------------------------------
# Section 3.5: Adaptive Residual Feature Enhancement (ARFE)
# ----------------------------------------------------------------------------
class ARFE(nn.Module):
    """Adaptive Residual Feature Enhancement for decoder refinement.

    Inputs:
        x: (B, C, H, W)
    Outputs:
        Tensor with the same shape as x.
    """

    def __init__(self, channel, reduction):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 5, padding=2),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Combine channel/spatial attention with gated residual."""
        ca_out = self.ca(x) * x
        sa_out = self.sa(x) * x

        fusion = torch.cat([x, ca_out, sa_out], dim=1)
        gate = self.gate(fusion)
        out = gate * ca_out + (1 - gate) * sa_out + x
        return out
