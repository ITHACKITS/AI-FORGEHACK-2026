"""
app/engine/scavenging.py

Pencucian partikel oleh hujan (Wet Deposition Scavenging).
Menghitung fraksi massa yang hilang dari udara dan terendapkan ke tanah akibat hujan.
Rumus: L = a * P^b, fraksi hilang = 1 - exp(-L * dt)
"""

import numpy as np

from app.engine.particles import ParticleArray


def apply_rain_scavenging(
    p: ParticleArray,
    rain_rate_mm_h: float,
    dt_s: float,
    a: float = 1.0e-4,
    b: float = 0.8
) -> np.ndarray:
    """
    Menerapkan pencucian hujan pada partikel aktif.
    Mengembalikan array massa yang tercuci (kg) per partikel aktif.
    """
    if rain_rate_mm_h <= 0.01:
        return np.zeros(0, dtype=float)

    active_idx = np.where(p.active)[0]
    if len(active_idx) == 0:
        return np.zeros(0, dtype=float)

    # Koefisien pencucian L (1/s)
    L = a * (rain_rate_mm_h ** b)
    loss_fraction = 1.0 - np.exp(-L * dt_s)
    loss_fraction = np.clip(loss_fraction, 0.0, 1.0)

    lost_mass = p.mass[active_idx] * loss_fraction
    p.mass[active_idx] -= lost_mass

    return lost_mass
