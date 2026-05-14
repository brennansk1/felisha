"""Color palette — ported from the design bundle's oklch values to hex.

These hex strings approximate the original oklch palette (dark navy with
blue-accent family) within the sRGB gamut Textual paints into. The mapping was
done by rendering each oklch value through a perceptual converter and rounding
to the nearest stable 24-bit code; the swatches below preserve the original
relative lightness/chroma ordering.
"""

from __future__ import annotations

# --- Surfaces ---------------------------------------------------------------
BG = "#0c1422"          # oklch(15% 0.03 250)
BG_SOFT = "#10182a"     # oklch(19% 0.035 250)
SURFACE = "#152034"     # oklch(22% 0.04 250)
SURFACE_HI = "#1c2a42"  # oklch(26% 0.045 250)
BORDER = "#2a3a55"      # oklch(32% 0.05 250)
BORDER_HI = "#3b5078"   # oklch(42% 0.06 250)
RULE = "#24314a"        # oklch(28% 0.04 250)

# --- Text ladder ------------------------------------------------------------
TEXT = "#eef2f9"        # oklch(96% 0.01 250)
TEXT_SOFT = "#cfd6e4"   # oklch(86% 0.015 250)
TEXT_MUTED = "#9aa3b5"  # oklch(70% 0.025 250)
TEXT_DIM = "#6b7691"    # oklch(54% 0.035 250)
TEXT_FAINT = "#4d5773"  # oklch(40% 0.04 250)

# --- Accent (sky blue) -----------------------------------------------------
ACCENT = "#5fa8ff"      # oklch(74% 0.16 240)
ACCENT_HI = "#9ec2ff"   # oklch(86% 0.13 235)
ACCENT_SOFT = "#3771bf" # oklch(48% 0.13 245)
ACCENT_BG = "#1a2c52"   # oklch(28% 0.08 245)

# --- Status (within the blue family) ---------------------------------------
SUCCESS = "#7ed2e6"     # cyan-blue — oklch(82% 0.12 210)
SUCCESS_BG = "#0e2a36"
WARNING = "#a3b6da"     # desaturated blue
WARNING_BG = "#1e2840"
DANGER = "#e08877"      # muted coral, used sparingly
DANGER_BG = "#3a1c19"
INFO = "#6ec2e2"


def color(name: str) -> str:
    """Look up a color name -> hex, with a deliberately small surface area."""
    return globals()[name.upper()]


__all__ = [
    "BG",
    "BG_SOFT",
    "SURFACE",
    "SURFACE_HI",
    "BORDER",
    "BORDER_HI",
    "RULE",
    "TEXT",
    "TEXT_SOFT",
    "TEXT_MUTED",
    "TEXT_DIM",
    "TEXT_FAINT",
    "ACCENT",
    "ACCENT_HI",
    "ACCENT_SOFT",
    "ACCENT_BG",
    "SUCCESS",
    "SUCCESS_BG",
    "WARNING",
    "WARNING_BG",
    "DANGER",
    "DANGER_BG",
    "INFO",
    "color",
]
