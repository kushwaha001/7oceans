import torch
import torch.nn as nn
from modules.mamba_block import BidirectionalMamba2Block


class MacroEncoder(nn.Module):
    """
    Component 1 — Macro Encoder.

    Reads the fleet-level daily traffic state as a sequence of grid cell tokens
    and compresses it into a single fixed-size context vector Z.

    Forward pass dimensions:
        input  : [N_c, 10]   (N_c occupied 0.5°x0.5° grid cells, 10 features each)
        output : [256]       context vector Z (fleet-level summary for that day)
    """

    def __init__(
        self,
        in_features: int = 10,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Input projection: 10 → d_model (no bias, no activation per architecture spec)
        self.input_proj = nn.Linear(in_features, d_model, bias=True)
        # n_layers stacked bidirectional Mamba2 blocks
        self.blocks = nn.ModuleList([
            BidirectionalMamba2Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
    
    def forward(self, macro_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            macro_tokens: [N_c, 10]  float32 — one tensor per training sample

        Returns:
            Z: [d_model]  fleet context vector
        """
        # Add batch dim for Mamba2: [N_c, 10] → [1, N_c, 10]
        
        x = macro_tokens.unsqueeze(0)

        # Input projection: [1, N_c, 10] → [1, N_c, d_model]
        x = self.input_proj(x)

        # 4x bidirectional Mamba2 blocks: [1, N_c, d_model] → [1, N_c, d_model]
        for block in self.blocks:
            x = block(x)

        # Mean pool over all cell positions → Z: [1, N_c, d_model] → [d_model]
        # Mean pool chosen over last token: sequence is spatial, not temporal,
        # so every cell contributes equally
        Z = Z_elaborate(x.squeeze(0).mean(dim=0)) 

        return Z
