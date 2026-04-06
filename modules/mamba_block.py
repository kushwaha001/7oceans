import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2


class CausalMamba2Block(nn.Module):
    """
    Single causal Mamba2 block for the micro decoder.
    Processes sequence left-to-right only (default Mamba2 behaviour).

    Pre-norm → Mamba2 SSM → gate → residual
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Linear(d_model, d_model)
        self.out  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        x_norm  = self.norm(x)
        y       = self.ssm(x_norm)                      # causal SSM
        gate    = torch.sigmoid(self.gate(x_norm))
        y_gated = y * gate
        return x + self.out(y_gated)                     # residual


class BidirectionalMamba2Block(nn.Module):
    """
    Single bidirectional Mamba2 block for the macro encoder.
    Runs two causal SSMs — one forward, one on the reversed sequence — then sums.

    Pre-norm → SSM_fwd + SSM_bwd (reversed) → sum → gate → residual
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.ssm_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate    = nn.Linear(d_model, d_model)
        self.out     = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]  — for macro encoder B=1 always
        Returns:
            [B, L, d_model]
        """
        x_norm   = self.norm(x)
        y_fwd    = self.ssm_fwd(x_norm)                 # north → south
        y_bwd    = self.ssm_bwd(x_norm.flip(1)).flip(1) # south → north
        y_bidi   = y_fwd + y_bwd
        gate     = torch.sigmoid(self.gate(x_norm))
        y_gated  = y_bidi * gate
        return x + self.out(y_gated)                     # residual
