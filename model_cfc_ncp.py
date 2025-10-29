# model_cfc_ncp.py
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class CfCNCPWrapper(nn.Module):
    """
    CfC with AutoNCP wiring (+tanh output saturation).

    Notes
    -----
    • Output is scaled to [-1, 1] to match imitation targets (u/10).
    • Keeps a forward signature (x, dt, h) for drop-in compatibility with
      'dt-aware' variants, although dt is unused here (timespans=None).
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
    ):
        super().__init__()
        self.in_dim = in_dim  # used when building the wiring for drawing

        wiring = AutoNCP(
            units=units,
            output_size=out_dim,
            sparsity_level=sparsity_level,
            seed=seed,
        )
        self.wiring = wiring
        self.core = CfC(
            input_size=in_dim,
            units=wiring,                # critical: pass the Wiring
            return_sequences=True,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, dt: float, h=None):
        """CfC forward with no explicit timespans; returns tanh-scaled output in [-1, 1]."""
        y_raw, h_out = self.core(x, hx=h, timespans=None)  # (B,T,out_dim)
        y_scaled = self.tanh(y_raw)                        # [-1,1]
        return y_scaled, h_out

    @property
    def state_size(self) -> int:
        return self.core.state_size

    # ---------- Wiring visualization (optional) ----------
    def draw_wiring(self,
                    layout: str = "shell",
                    draw_labels: bool = False,
                    save_path: str | None = None):
        """
        Draw the AutoNCP wiring. Builds the wiring on first call if needed.
        """
        import matplotlib.pyplot as plt

        if not self.wiring.is_built():
            self.wiring.build(self.in_dim)

        # colors (sensory=olive, inter=blue, command=cyan, motor=orange)
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
