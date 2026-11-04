"""One place that controls figure quality for every script.

Environment variables (optional):
  BBBCP_FIG_DPI      raster resolution for PNG output (default 300; use 600 for print)
  BBBCP_FIG_FORMATS  comma-separated formats (default "png,pdf"; pdf/svg are vector)
"""
import os
from pathlib import Path

import matplotlib

DPI = int(os.environ.get("BBBCP_FIG_DPI", "300"))
FORMATS = [f.strip().lower() for f in os.environ.get("BBBCP_FIG_FORMATS", "png,pdf").split(",") if f.strip()]

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType fonts (journal requirement)
    "svg.fonttype": "none",
    "savefig.dpi": DPI, "savefig.bbox": "tight",
})


def save(fig, path):
    """Save `fig` to every requested format next to `path` (suffix is replaced)."""
    path = Path(path)
    for fmt in FORMATS:
        fig.savefig(path.with_suffix("." + fmt), dpi=DPI, bbox_inches="tight")
