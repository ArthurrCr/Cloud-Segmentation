"""Shared styling constants and helpers for all visualizations."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# Canonical model order used across every figure.
MODEL_ORDER: List[str] = [
    "Unet + regnetz d8",
    "Unet + regnetz d8 (CE)",
    "CloudS2Mask ensemble",
    "CloudS2Mask Dice_1 (single)",
    "CloudS2Mask Dice_2 (single)",
    "Unet + MobilenetV2",
]

# Pastel palette aligned with MODEL_ORDER.
PASTEL_PALETTE: Dict[str, str] = {
    "Unet + regnetz d8": "#7CB9E8",
    "Unet + regnetz d8 (CE)": "#A3D9A5",
    "CloudS2Mask ensemble": "#F4A87C",
    "CloudS2Mask Dice_1 (single)": "#F7D98E",
    "CloudS2Mask Dice_2 (single)": "#F5A0A0",
    "Unet + MobilenetV2": "#B8B8D1",
}

# Display labels.
MODEL_LABELS: Dict[str, str] = {
    "Unet + regnetz d8": "Ours (UNet+RegNetZ-D8)",
    "Unet + regnetz d8 (CE)": "Ours (UNet+RegNetZ-D8 (CE))",
    "CloudS2Mask ensemble": "CloudS2Mask (ensemble)",
    "CloudS2Mask Dice_1 (single)": "CloudS2Mask Dice\u2081",
    "CloudS2Mask Dice_2 (single)": "CloudS2Mask Dice\u2082",
    "Unet + MobilenetV2": "UNetMobV2 (baseline)",
}

# Marker shapes.
MODEL_MARKERS: Dict[str, str] = {
    "Unet + regnetz d8": "o",
    "Unet + regnetz d8 (CE)": "o",
    "CloudS2Mask ensemble": "s",
    "CloudS2Mask Dice_1 (single)": "^",
    "CloudS2Mask Dice_2 (single)": "v",
    "Unet + MobilenetV2": "D",
}

# Fallback colors for unknown models.
_FALLBACK_COLORS = ["#A3D9A5", "#C5A3D9", "#A3D9D9", "#C9B8A3", "#A3B8C9"]


def get_style(model_name: str) -> Dict[str, str]:
    """Return label, color, and marker for a model with fallback defaults."""
    if model_name in MODEL_LABELS:
        return {
            "label": MODEL_LABELS[model_name],
            "color": PASTEL_PALETTE[model_name],
            "marker": MODEL_MARKERS[model_name],
        }
    idx = hash(model_name) % len(_FALLBACK_COLORS)
    return {"label": model_name, "color": _FALLBACK_COLORS[idx], "marker": "o"}


def sort_models(models: List[str]) -> List[str]:
    """Sort model names according to MODEL_ORDER; unknowns go at the end."""
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order_map.get(m, len(MODEL_ORDER)))


def clean_spines(ax: plt.Axes) -> None:
    """Remove top, right, and left spines, keeping only the bottom."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)


def save_and_show(
    fig: plt.Figure,
    save_path: Optional[str] = None,
) -> None:
    """Tight-layout, optionally save, and display a figure."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()