"""
app/engine/volcano.py

Pemodelan sumber erupsi gunung berapi (Volcanic Ash Source Term).
Mengimplementasikan formulasi D24 Mastin dkk. 2009:
M = c * H^p dengan c = 140.85 kg/s, p = 4.15, H diukur di atas kawah (above vent).
Dilengkapi variasi lognormal per anggota ensemble dan distribusi ukuran butir.
"""

from dataclasses import dataclass

import numpy as np

from app.engine.particles import ParticleArray


@dataclass
class GrainClass:
    name: str
    fraction: float
    settling_vel_ms: float


GRAIN_CLASSES = [
    GrainClass(name="fine_ash", fraction=0.20, settling_vel_ms=0.04),
    GrainClass(name="medium_ash", fraction=0.40, settling_vel_ms=0.40),
    GrainClass(name="coarse_ash", fraction=0.40, settling_vel_ms=2.50),
]


def calculate_mer_mastin(h_km_above_vent: float) -> float:
    """
    Menghitung Laju Erupsi Massa (Mass Eruption Rate / MER) dalam kg/s
    sesuai Mastin dkk. 2009 (D24).
    Rumus: M = 140.85 * (H ^ 4.15)
    """
    h_safe = max(0.1, float(h_km_above_vent))
    return 140.85 * (h_safe ** 4.15)


def generate_volcano_particles_per_step(
    p: ParticleArray,
    start_slot: int,
    n_particles: int,
    vent_lat: float,
    vent_lon: float,
    vent_elev_m: float,
    plume_top_m: float,
    total_mass_step_kg: float,
    rng: np.random.Generator
) -> int:
    """
    Melepaskan partikel abu sepanjang kolom letusan pada satu langkah waktu simulasi.
    Mengembalikan jumlah partikel baru yang berhasil dimasukkan.
    """
    slots_avail = len(p.active) - start_slot
    n_insert = min(n_particles, slots_avail)
    if n_insert <= 0:
        return 0

    end_slot = start_slot + n_insert
    idx_range = np.arange(start_slot, end_slot)

    # Posisi awal dekat ventilasi
    p.lon[idx_range] = vent_lon + rng.normal(0, 0.002, size=n_insert)
    p.lat[idx_range] = vent_lat + rng.normal(0, 0.002, size=n_insert)

    # Distribusi ketinggian pelepasan (antara kawah hingga puncak kolom abu)
    # Jamin batas atas lebih tinggi dari dasar ventilasi
    safe_top_m = max(vent_elev_m + 500.0, plume_top_m)
    z_rand = rng.uniform(vent_elev_m, safe_top_m, size=n_insert)
    umbrella_mask = rng.uniform(0, 1, size=n_insert) < 0.3
    if np.any(umbrella_mask):
        umb_count = int(np.sum(umbrella_mask))
        umb_low = max(vent_elev_m + 100.0, 0.8 * safe_top_m)
        z_rand[umbrella_mask] = rng.uniform(umb_low, safe_top_m, size=umb_count)
    p.z[idx_range] = z_rand

    # Distribusi massa partikel merata
    mass_per_particle = total_mass_step_kg / n_insert
    p.mass[idx_range] = mass_per_particle

    # Kelas butir
    choices = rng.choice(len(GRAIN_CLASSES), size=n_insert, p=[g.fraction for g in GRAIN_CLASSES])
    settling_vels = np.array([GRAIN_CLASSES[c].settling_vel_ms for c in choices])
    p.settling_vel[idx_range] = settling_vels

    p.cluster_id[idx_range] = 1  # Volcano ID
    p.active[idx_range] = True
    p.deposited[idx_range] = False
    p.age_s[idx_range] = 0.0

    return n_insert
