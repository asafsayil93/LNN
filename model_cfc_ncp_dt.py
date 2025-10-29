# model_cfc_ncp_dt.py
import torch
import torch.nn as nn
from typing import Optional, Union
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCNCPWrapper(nn.Module):
    """
    CfC with AutoNCP wiring (+tanh output saturation) and explicit dt support.

    Output range
    ------------
    • The output activation defaults to tanh, so y ∈ [-1, 1], which matches
      imitation targets y = u/10.

    Mode & dt handling
    ------------------
    • mode: "default", "no_gate", "pure" (and other modes supported by ncps.CfC).
    • dt: None | float/int | Tensor
        - None     → timespans=None (CfC assumes 1.0 internally)
        - scalar   → timespans = full((B,T,H), dt)
        - 1D (T,)  → broadcast across batch → (B,T,H)
        - 2D (B,T) → expand last dim → (B,T,H)
        - 3D (B,T,1) → expand to H; (B,T,H) → used as-is
    """

    def __init__(
        self,
        in_dim: int = 7,
        out_dim: int = 1,
        units: int = 32,
        sparsity_level: float = 0.5,
        mode: str = "default",
        mixed_memory: bool = False,
        seed: int = 22222,
        output_activation: Optional[nn.Module] = None,  # override if needed
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        wiring = AutoNCP(
            units=units,
            output_size=out_dim,
            sparsity_level=sparsity_level,
            seed=seed,
        )
        self.wiring = wiring

        self.core = CfC(
            input_size=in_dim,
            units=wiring,                # pass the Wiring
            return_sequences=True,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,                   # "default" | "no_gate" | "pure" | ...
        )

        # Default output activation: tanh ([-1,1] scale)
        self.out_act = output_activation if output_activation is not None else nn.Tanh()

    # ---- helpers: dt -> timespans (B,T,H) or None ----
    @property
    def state_size(self) -> int:
        return self.core.state_size  # H

    def _make_timespans_for_cfc(self, x: torch.Tensor, dt) -> torch.Tensor | None:
        """Produce CfC 'timespans' in shape (B, T, H), or None if dt is not provided."""
        if dt is None:
            return None  # CfC internally assumes ts=1.0

        B, T, _ = x.shape
        H = self.state_size
        dev, dty = x.device, x.dtype

        if not isinstance(dt, torch.Tensor):
            # scalar -> copy across (B,T,H)
            return torch.full((B, T, H), float(dt), device=dev, dtype=dty)

        dt = dt.to(device=dev, dtype=dty)
        if dt.ndim == 0:                        # ()
            return dt.view(1,1,1).expand(B,T,H).contiguous()
        if dt.ndim == 1 and dt.shape[0] == T:   # (T,)
            return dt.view(1,T,1).expand(B,T,H).contiguous()
        if dt.ndim == 2 and dt.shape == (B, T): # (B,T)
            return dt.unsqueeze(-1).expand(B,T,H).contiguous()
        if dt.ndim == 3:
            if dt.shape == (B, T, 1):           # (B,T,1) → expand to H
                return dt.expand(B,T,H).contiguous()
            if dt.shape == (B, T, H):           # already correct
                return dt.contiguous()

        raise ValueError(
            f"Unsupported dt shape {tuple(dt.shape)}; expected scalar, (T,), (B,T), (B,T,1) or (B,T,H)."
        )

    def forward(self, x: torch.Tensor, dt, h=None):
        """Forward with explicit 'timespans' built from dt; returns activation-scaled output."""
        ts = self._make_timespans_for_cfc(x, dt)      # (B,T,H) or None
        y_raw, h_out = self.core(x, hx=h, timespans=ts)
        y = self.out_act(y_raw)  # tanh by default -> [-1,1]
        return y, h_out


    # ---------- Wiring visualization (optional) ----------
    def draw_wiring(self,
                    layout: str = "shell",
                    draw_labels: bool = False,
                    save_path: Optional[str] = None):
        """
        Draw the AutoNCP wiring. Builds the wiring on first call if needed.
        """
        import matplotlib.pyplot as plt

        if not self.wiring.is_built():
            self.wiring.build(self.in_dim)

        neuron_colors = {"command": "tab:cyan"}
        legend = self.wiring.draw_graph(
            layout=layout,
            neuron_colors=neuron_colors,
            draw_labels=draw_labels,
        )

        title = (f"NCP wiring | units={self.state_size}  "
                 f"motor={self.wiring.output_dim}  "
                 f"synapses={int(self.wiring.synapse_count)}  "
                 f"sensory_syn={int(self.wiring.sensory_synapse_count)}")
        plt.title(title)
        if legend:
            plt.legend(handles=legend, loc="upper center", bbox_to_anchor=(1.12, 1.0))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        return legend
