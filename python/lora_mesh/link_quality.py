"""
link_quality.py — bridges the MATLAB PHY/channel simulation into Python.

runLoRaDisasterSim.m exports lora_disaster_link_quality_dataset.csv with
columns: SF, SNR_dB, Fading, Collision, SymbolErrorRate, PDR_estimate.

This module loads that CSV (physics-grounded ground truth) and provides
PDR(snr_db, fading, collision) via 1-D interpolation per (fading,
collision) bucket. If the CSV hasn't been generated yet (e.g. no
MATLAB available on this machine), it falls back to a reasonable
analytic sigmoid so the rest of the pipeline still runs end-to-end —
swap in the real CSV any time by placing it in matlab/ and rerunning.
"""

from pathlib import Path
import numpy as np
import pandas as pd

_DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "matlab"
    / "lora_disaster_link_quality_dataset.csv"
)


def _analytic_fallback_pdr(snr_db, fading: str, collision: bool) -> float:
    """Logistic PDR-vs-SNR curve, degraded by fading/collision. Used only
    if the real MATLAB dataset CSV isn't present."""
    midpoint = {"none": -4.0, "rician": 0.0, "rayleigh": 3.0}.get(fading, 0.0)
    if collision:
        midpoint += 4.0
    pdr = 1.0 / (1.0 + np.exp(-(snr_db - midpoint) / 2.5))
    return float(np.clip(pdr, 0.0, 1.0))


class LinkQualityModel:
    def __init__(self, csv_path: Path | str | None = None):
        csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
        self.available = csv_path.exists()
        self._tables = {}
        if self.available:
            df = pd.read_csv(csv_path)
            for (fading, collision), grp in df.groupby(["Fading", "Collision"]):
                grp = grp.sort_values("SNR_dB")
                self._tables[(fading, bool(collision))] = (
                    grp["SNR_dB"].to_numpy(),
                    grp["PDR_estimate"].to_numpy(),
                )

    def pdr(self, snr_db: float, fading: str = "rician", collision: bool = False) -> float:
        key = (fading, collision)
        if self.available and key in self._tables:
            xs, ys = self._tables[key]
            return float(np.clip(np.interp(snr_db, xs, ys, left=ys[0], right=ys[-1]), 0, 1))
        return _analytic_fallback_pdr(snr_db, fading, collision)
