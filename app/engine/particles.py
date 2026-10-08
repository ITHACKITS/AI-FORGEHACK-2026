"""
app/engine/particles.py

Model partikel Lagrangian untuk adveksi RK2, difusi horizontal, dan jatuhan vertikal.
Didesain tervektorisasi tinggi dengan NumPy untuk performa optimal.
"""

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6371000.0


@dataclass
class ParticleArray:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray  # Ketinggian di atas tanah (meter)
    mass: np.ndarray  # Massa partikel (kg)
    settling_vel: np.ndarray  # Kecepatan jatuh (m/s)
    cluster_id: np.ndarray  # ID cluster sumber untuk penandaan/pemadaman
    active: np.ndarray  # True jika masih melayang di udara
    deposited: np.ndarray  # True jika sudah mendarat di permukaan
    age_s: np.ndarray  # Umur partikel (detik)


def create_empty_particles(capacity: int) -> ParticleArray:
    return ParticleArray(
        lon=np.zeros(capacity, dtype=float),
        lat=np.zeros(capacity, dtype=float),
        z=np.zeros(capacity, dtype=float),
        mass=np.zeros(capacity, dtype=float),
        settling_vel=np.zeros(capacity, dtype=float),
        cluster_id=np.zeros(capacity, dtype=int),
        active=np.zeros(capacity, dtype=bool),
        deposited=np.zeros(capacity, dtype=bool),
        age_s=np.zeros(capacity, dtype=float),
    )


def advect_particles_rk2(
    p: ParticleArray,
    u_ms: float,
    v_ms: float,
    dt_s: float,
    diff_coeff_m2s: float = 50.0,
    rng: np.random.Generator | None = None
):
    """
    Adveksi Runge-Kutta orde 2 (RK2) untuk partikel aktif.
    Dilengkapi difusi horizontal acak (random walk) dan settling gravitasi.
    """
    active_idx = np.where(p.active)[0]
    if len(active_idx) == 0:
        return

    n_active = len(active_idx)
    lats = p.lat[active_idx]
    zs = p.z[active_idx]
    w_s = p.settling_vel[active_idx]

    # Faktor konversi meter ke derajat
    lat_rad = np.radians(lats)
    m_per_deg_lat = (np.pi * EARTH_RADIUS_M) / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(lat_rad)
    m_per_deg_lon = np.maximum(m_per_deg_lon, 100.0)  # Cegah pembagian nol di kutub

    # RK2 Tahap 1 (k1)
    k1_lon = (u_ms * dt_s) / m_per_deg_lon
    k1_lat = (v_ms * dt_s) / m_per_deg_lat

    # RK2 Tahap 2 (k2) dievaluasi di tengah langkah
    mid_lat_rad = np.radians(lats + 0.5 * k1_lat)
    mid_m_per_deg_lon = m_per_deg_lat * np.cos(mid_lat_rad)
    mid_m_per_deg_lon = np.maximum(mid_m_per_deg_lon, 100.0)

    k2_lon = (u_ms * dt_s) / mid_m_per_deg_lon
    k2_lat = (v_ms * dt_s) / m_per_deg_lat

    # Perpindahan adveksi rata-rata
    d_lon = 0.5 * (k1_lon + k2_lon)
    d_lat = 0.5 * (k1_lat + k2_lat)

    # Difusi horizontal random walk: sqrt(2 * Kh * dt)
    if diff_coeff_m2s > 0 and rng is not None:
        diff_sigma_m = np.sqrt(2.0 * diff_coeff_m2s * dt_s)
        rx = rng.normal(0, diff_sigma_m, size=n_active)
        ry = rng.normal(0, diff_sigma_m, size=n_active)
        d_lon += rx / m_per_deg_lon
        d_lat += ry / m_per_deg_lat

    # Update posisi horizontal
    p.lon[active_idx] += d_lon
    p.lat[active_idx] += d_lat
    p.age_s[active_idx] += dt_s

    # Settling gravitasi vertikal
    new_z = zs - (w_s * dt_s)
    p.z[active_idx] = new_z

    # Cek partikel yang mendarat di tanah (z <= 0)
    landed_mask = new_z <= 0
    if np.any(landed_mask):
        landed_sub_idx = active_idx[landed_mask]
        p.z[landed_sub_idx] = 0.0
        p.active[landed_sub_idx] = False
        p.deposited[landed_sub_idx] = True
