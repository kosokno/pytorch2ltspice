# clock_sampling.py
# Utilities for sampling LTspice waveforms on clock edges.

from __future__ import annotations

from typing import Literal
import pandas as pd


Edge = Literal["falling", "rising", "both"]


def sample_on_clock(
    df: pd.DataFrame,
    clk: str = "V(ctrlclk)",
    threshold: float = 0.5,
    latch_edge: Edge = "falling",
) -> pd.DataFrame:
    """
    Sample (latch) dataframe rows at specified clock edges.

    Parameters
    ----------
    df : pandas.DataFrame
        LTspice transient result dataframe.
    clk : str
        Column name of the clock signal (e.g., 'V(ctrlclk)').
    threshold : float
        Threshold voltage for edge detection.
    latch_edge : {'falling', 'rising', 'both'}
        Which edge(s) to latch.

    Returns
    -------
    pandas.DataFrame
        Rows latched at specified edges, index reset.
    """
    if latch_edge not in ("falling", "rising", "both"):
        raise ValueError("latch_edge must be 'falling', 'rising', or 'both'")

    if clk not in df.columns:
        raise KeyError(f"Clock column not found: {clk}")

    clk_v = df[clk].values
    if len(clk_v) < 2:
        return df.iloc[0:0].copy()

    # Check if the clock starts at high level (same behavior as your original)
    if clk_v[0] > threshold:
        raise ValueError("ERROR: Clock started with Level Hi")

    indices: list[int] = []
    state = "LOW"  # FSM: LOW -> HIGH (rising), HIGH -> LOW (falling)

    for i in range(1, len(clk_v)):
        prev = clk_v[i - 1]
        curr = clk_v[i]

        # Rising edge: LOW -> HIGH
        if state == "LOW" and prev <= threshold and curr > threshold:
            if latch_edge in ("rising", "both"):
                indices.append(i)
            state = "HIGH"

        # Falling edge: HIGH -> LOW
        elif state == "HIGH" and prev > threshold and curr <= threshold:
            if latch_edge in ("falling", "both"):
                indices.append(i)
            state = "LOW"

    return df.iloc[indices].reset_index(drop=True)
