from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardConvBlock(nn.Module):
    """Standard Conv-BN-SiLU block used by the baseline BiFPN variants."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (DWConv + PWConv + BN + SiLU)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FastNormalizedFusion(nn.Module):
    """BiFPN-style learned scalar fusion weights."""

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        return [weights[i].to(device=inputs[0].device, dtype=inputs[0].dtype).view(1, 1, 1, 1) for i in range(len(inputs))]


class ContextAwareWeight(nn.Module):
    """Context-aware weighting module for dynamic feature fusion."""

    def __init__(self, channels: int, num_inputs: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.num_inputs = num_inputs
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * num_inputs, hidden, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, num_inputs, kernel_size=1, bias=True)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        context = torch.cat([self.pool(x) for x in inputs], dim=1)
        hidden = self.act(self.fc1(context))
        logits = self.fc2(hidden)
        weights = F.softmax(logits, dim=1)
        return [weights[:, i : i + 1] for i in range(self.num_inputs)]


class TALFFN(nn.Module):
    """TAL-FFN: Task-driven Asymmetric Lightweight Feature Fusion Network.
    
    A specialized feature pyramid network for tiny object detection in agricultural scenarios.
    This architecture implements three key innovations proposed in the AgriYOLO paper:
    
    1. **ADSA (Asymmetric Depth Strategy Allocation)**: 
       Allocates more computational resources to shallow layers (P2/P3) by using asymmetric
       depth configuration (heavy_levels, heavy_repeats, light_repeats).
    
    2. **CADFM (Context-Aware Dynamic Fusion Mechanism)**: 
       Dynamically computes fusion weights based on input content using a lightweight
       context-aware weight network (use_cawn, reduction).
    
    3. **DSConv (Depthwise Separable Convolution)**: 
       All convolutional blocks use depthwise separable convolution for lightweight feature extraction.

    Args:
        out_channels: Output channel number for all feature levels
        fpn_sizes: List of input channel numbers for each feature level
        heavy_levels: Number of levels to use heavy_repeats (typically 2 for P2/P3)
        heavy_repeats: Number of DSConv blocks for heavy levels (typically 2)
        light_repeats: Number of DSConv blocks for light levels (typically 1)
        use_cawn: Whether to use Context-Aware Weight Network for dynamic fusion (CADFM)
        use_adsa: Whether to use Asymmetric Depth Strategy Allocation (ADSA, default True)
        reduction: Reduction ratio for CAWN hidden channels (default 4)
        
    Example:
        >>> fpn_sizes = [128, 256, 512, 1024]  # P2, P3, P4, P5
        >>> talffn = TALFFN(out_channels=128, fpn_sizes=fpn_sizes, heavy_levels=2, 
        ...                 heavy_repeats=2, light_repeats=1, use_cawn=True)
        >>> inputs = [torch.randn(1, c, h, w) for c, h, w in zip(fpn_sizes, [160, 80, 40, 20])]
        >>> outputs = talffn(inputs)  # Returns [P2, P3, P4, P5] features
    """


    def __init__(
        self,
        out_channels: int,
        fpn_sizes: list[int],
        heavy_levels: int = 2,
        heavy_repeats: int = 2,
        light_repeats: int = 1,
        use_cawn: bool = False,
        use_adsa: bool = False,
        use_dsconv: bool = False,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        assert len(fpn_sizes) >= 2, "TALFFN expects at least two feature levels."

        self.num_levels = len(fpn_sizes)
        self.out_channels = out_channels
        self.use_cawn = use_cawn
        self.use_adsa = use_adsa
        self.use_dsconv = use_dsconv
        heavy_levels = max(0, min(heavy_levels, self.num_levels))

        if self.use_adsa:
            repeats_per_level = [heavy_repeats if i < heavy_levels else light_repeats for i in range(self.num_levels)]
        else:
            repeats_per_level = [max(light_repeats, 1)] * self.num_levels

        self.input_projs = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False) for in_c in fpn_sizes]
        )
        self.input_bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in fpn_sizes])

        self.top_down_levels = list(range(self.num_levels - 2, -1, -1))
        self.bottom_up_levels = list(range(1, self.num_levels))

        self.top_down_blocks = nn.ModuleList(
            self._make_block(repeats_per_level[level]) for level in self.top_down_levels
        )
        self.bottom_up_blocks = nn.ModuleList(
            self._make_block(repeats_per_level[level]) for level in self.bottom_up_levels
        )

        weight_module = ContextAwareWeight if self.use_cawn else FastNormalizedFusion

        self.td_weight_modules = nn.ModuleList(
            weight_module(out_channels, 2, reduction) if self.use_cawn else weight_module(2) for _ in self.top_down_levels
        )

        self.bu_num_inputs = [3 if idx < len(self.bottom_up_levels) - 1 else 2 for idx in range(len(self.bottom_up_levels))]
        self.bu_weight_modules = nn.ModuleList(
            weight_module(out_channels, num_inputs, reduction) if self.use_cawn else weight_module(num_inputs)
            for num_inputs in self.bu_num_inputs
        )

    def _make_block(self, repeats: int) -> nn.Sequential:
        repeats = max(repeats, 1)
        block_cls = DepthwiseSeparableConv if self.use_dsconv else StandardConvBlock
        return nn.Sequential(*(block_cls(self.out_channels) for _ in range(repeats)))

    def _project_inputs(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        projected: list[torch.Tensor] = []
        for x, proj, bn in zip(inputs, self.input_projs, self.input_bns):
            x = proj(x)
            x = bn(x)
            projected.append(x)
        return projected

    @staticmethod
    def _fuse(features: list[torch.Tensor], weights: list[torch.Tensor]) -> torch.Tensor:
        weighted = [w * f for w, f in zip(weights, features)]
        return sum(weighted)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(inputs) == self.num_levels, "Unexpected number of feature levels for TALFFN."

        projected = self._project_inputs(inputs)

        td_features: list[torch.Tensor | None] = [None] * self.num_levels
        td_features[-1] = projected[-1]

        for idx, level in enumerate(self.top_down_levels):
            upsampled = F.interpolate(td_features[level + 1], size=projected[level].shape[-2:], mode="nearest")
            weights = self.td_weight_modules[idx]([projected[level], upsampled])
            fused = self._fuse([projected[level], upsampled], weights)
            td_features[level] = self.top_down_blocks[idx](fused)

        outputs: list[torch.Tensor | None] = [None] * self.num_levels
        outputs[0] = td_features[0]

        for idx, level in enumerate(self.bottom_up_levels):
            prev_out = outputs[level - 1]
            assert prev_out is not None
            target_hw = td_features[level].shape[-2:]
            downsampled = F.max_pool2d(prev_out, kernel_size=3, stride=2, padding=1)
            if downsampled.shape[-2:] != target_hw:
                downsampled = F.interpolate(downsampled, size=target_hw, mode="nearest")
            num_inputs = self.bu_num_inputs[idx]
            if num_inputs == 3:
                fusion_inputs = [projected[level], td_features[level], downsampled]
            else:
                fusion_inputs = [td_features[level], downsampled]
            weights = self.bu_weight_modules[idx](fusion_inputs)
            outputs[level] = self.bottom_up_blocks[idx](self._fuse(fusion_inputs, weights))

        return [x for x in outputs if x is not None]


class FeatureSelect(nn.Module):
    """Selects a single feature map from a list based on the given index."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("FeatureSelect expects a list or tuple input")
        if not (0 <= self.index < len(inputs)):
            raise IndexError(f"FeatureSelect index {self.index} out of range for inputs of length {len(inputs)}")
        return inputs[self.index]



# Backward compatibility alias
DynamicWeightedBiFPN = TALFFN

__all__ = ("TALFFN", "DynamicWeightedBiFPN", "FeatureSelect")
