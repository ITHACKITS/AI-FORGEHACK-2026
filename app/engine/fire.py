"""
app/engine/fire.py

Pemodelan sumber kebakaran hutan dan asap PM2.5 (Wildfire Smoke Source Term).
Mengubah FRP satelit menjadi biomassa terbakar dan laju emisi PM2.5.
Mendukung penandaan partikel per titik api untuk fitur pemadaman (extinguish) tanpa simulasi ulang.
"""

from typing import Any, Dict, List

import numpy as np

from app.engine.particles import ParticleArray


def sample_frp_conversion_factor(rng: np.random.Generator) -> float:
    """
    D13: FRP ke bahan kering (kg/MJ).
    Distribusi segitiga: low=0.27, mode=0.37, high=0.45 (Wooster 2005, Li 2018).
    """
    return float(rng.triangular(left=0.27, mode=0.37, right=0.45))


def generate_fire_particles_per_step(
    p: ParticleArray,
    start_slot: int,
    hotspots: List[Dict[str, Any]],
    dt_s: float,
    particles_per_hotspot: int,
    frp_factor_kg_mj: float,
    emission_factor_g_kg: float = 15.0,
    mixed_layer_m: float = 1200.0,
    rng: np.random.Generator | None = None
) -> int:
    """
    Melepaskan partikel asap PM2.5 dari setiap titik api aktif pada langkah waktu saat ini.
    Massa PM2.5 (kg) = FRP (MW) * 1e6 J/s * dt (s) / 1e6 MJ * frp_factor (kg/MJ) * (ef / 1000)
    Sederhananya: Mass = FRP * dt_s * frp_factor_kg_mj * (emission_factor_g_kg / 1000.0)
    """
    active_hotspots = [h for h in hotspots if h.get("is_active", True)]
    n_active = len(active_hotspots)
    if n_active == 0:
        return 0

    slots_avail = len(p.active) - start_slot
    if slots_avail <= 0:
        return 0

    # Batasi partikel yang dimasukkan per langkah waktu (maksimal 400 titik teratas/teraktif)
    n_insert = min(n_active, 400, slots_avail)
    if n_insert <= 0:
        return 0

    chosen = active_hotspots[:n_insert]
    actual_insert = len(chosen)

    lats = np.fromiter((float(h["latitude"]) for h in chosen), dtype=float, count=actual_insert)
    lons = np.fromiter((float(h["longitude"]) for h in chosen), dtype=float, count=actual_insert)
    frps = np.fromiter((float(h.get("frp", 15.0)) for h in chosen), dtype=float, count=actual_insert)
    def _safe_cluster_id(raw_id, fallback_idx):
        try:
            parts = str(raw_id).split("_")
            for p in reversed(parts):
                if p.isdigit():
                    return int(p)
            return int(raw_id)
        except Exception:
            return fallback_idx

    ids = np.fromiter(
        (_safe_cluster_id(h.get("id"), i) for i, h in enumerate(chosen)),
        dtype=int,
        count=actual_insert
    )

    ef_kg_per_kg = emission_factor_g_kg / 1000.0
    masses = frps * dt_s * frp_factor_kg_mj * ef_kg_per_kg

    end_slot = start_slot + actual_insert
    sub_idx = np.arange(start_slot, end_slot)

    noise_lon = rng.normal(0, 0.003, size=actual_insert) if rng else 0.0
    noise_lat = rng.normal(0, 0.003, size=actual_insert) if rng else 0.0
    altitudes = rng.uniform(10.0, mixed_layer_m, size=actual_insert) if rng else 100.0

    p.lon[sub_idx] = lons + noise_lon
    p.lat[sub_idx] = lats + noise_lat
    p.z[sub_idx] = altitudes
    p.mass[sub_idx] = masses
    p.settling_vel[sub_idx] = 0.0
    p.cluster_id[sub_idx] = ids
    p.active[sub_idx] = True
    p.deposited[sub_idx] = False
    p.age_s[sub_idx] = 0.0

    return actual_insert
