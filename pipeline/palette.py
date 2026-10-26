"""Plot colours: purple and black, with blue only where a third distinct series is unavoidable."""
from matplotlib.colors import to_rgb

PURPLE = "#4d1259"
BLACK = "#121c1f"
BLUE = "#0a3c8a"


def tint(color: str, amount: float) -> tuple:
    """Blend `color` towards white (amount 0 = unchanged, 1 = white)."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


# GeoSmart variants after the first: lighter purples, then blue and its tint (only needed with many variants)
VARIANT_COLORS = [PURPLE, BLUE, tint(PURPLE, 0.45), tint(BLUE, 0.45), tint(PURPLE, 0.7), tint(BLACK, 0.5)]
