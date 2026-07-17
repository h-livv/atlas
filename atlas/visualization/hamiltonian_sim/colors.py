"""Colormap helpers for lattice node rendering.

Physics-agnostic display utilities. Defaults favor a diverging map suitable
for signed observables (negative → blue, zero → white, positive → red).
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import Colormap


# Built-in RdBu_r ends are relatively dark; use a brighter custom map so
# saturated values near ±1 read clearly as vivid blue / red.
DEFAULT_DIVERGING_CMAP = "atlas_diverging"


def _build_atlas_diverging() -> Colormap:
    """Bright blue → white → bright red diverging colormap."""

    return mcolors.LinearSegmentedColormap.from_list(
        DEFAULT_DIVERGING_CMAP,
        [
            (0.0, "#1E88E5"),  # bright blue at vmin (e.g. |1⟩ / <Z> = -1)
            (0.5, "#FFFFFF"),  # white at zero
            (1.0, "#F44336"),  # bright red at vmax (e.g. |0⟩ / <Z> = +1)
        ],
    )


def _ensure_registered() -> None:
    cmap = _build_atlas_diverging()
    try:
        if DEFAULT_DIVERGING_CMAP not in matplotlib.colormaps:
            matplotlib.colormaps.register(cmap, name=DEFAULT_DIVERGING_CMAP)
    except AttributeError:
        # Matplotlib < 3.5
        if DEFAULT_DIVERGING_CMAP not in matplotlib.cm.cmap_d:
            matplotlib.cm.register_cmap(name=DEFAULT_DIVERGING_CMAP, cmap=cmap)


_ensure_registered()


def diverging_norm(
    vmin: float,
    vmax: float,
    *,
    vcenter: float = 0.0,
) -> mcolors.TwoSlopeNorm | mcolors.Normalize:
    """Return a norm centered at ``vcenter`` when the range straddles it."""

    if vmin < vcenter < vmax:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def resolve_colormap(name: Optional[str] = None) -> Colormap:
    """Return a matplotlib colormap by name (default bright diverging)."""

    _ensure_registered()
    cmap_name = name or DEFAULT_DIVERGING_CMAP
    try:
        return matplotlib.colormaps[cmap_name]
    except AttributeError:
        # Matplotlib < 3.5
        return matplotlib.cm.get_cmap(cmap_name)


def map_values_to_colors(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    *,
    cmap: Optional[str] = None,
) -> np.ndarray:
    """Map scalar site values to RGBA colors with shape ``(n, 4)``."""

    norm = diverging_norm(vmin, vmax)
    colormap = resolve_colormap(cmap)
    return colormap(norm(np.asarray(values, dtype=float)))
