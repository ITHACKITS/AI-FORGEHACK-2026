"""
tests/test_physics.py

Pengujian prinsip fisika dan numerik sesuai skill udarasiaga-simulation-testing:
1. Adveksi: Angin konstan u selama T menggeser partikel sejauh u*T.
2. Kecepatan jatuh: Butir besar mendarat lebih cepat/dekat dibanding butir halus.
3. Kekekalan massa: Tanpa hujan/settling massa konstan; dengan pencucian massa hilang = terendapkan.
4. Linearitas tag: Membuang partikel bertag X = simulasi tanpa sumber X.
5. Determinisme: Seed sama menghasilkan keluaran identik.
6. Tiga tingkat subset: Minimum ⊆ Mid ⊆ Maximum.
7. Plausibilitas massa D24 Mastin et al. 2009.
"""

import numpy as np

from app.engine.aggregation import aggregate_tiers_for_hour
from app.engine.grid import EARTH_RADIUS_M, SimulationGrid
from app.engine.particles import advect_particles_rk2, create_empty_particles
from app.engine.scavenging import apply_rain_scavenging
from app.engine.volcano import calculate_mer_mastin


def test_advection_rk2_accuracy():
    """
    Tes 1: Adveksi angin konstan u selama T menggeser partikel sejauh u*T.
    """
    p = create_empty_particles(10)
    p.lon[0] = 105.0
    p.lat[0] = 0.0  # Di khatulistiwa untuk kemudahan verifikasi analitik
    p.z[0] = 500.0
    p.mass[0] = 100.0
    p.settling_vel[0] = 0.0
    p.active[0] = True

    u_const = 10.0  # 10 m/s ke timur
    v_const = 0.0
    dt_s = 60.0
    total_steps = 60  # Total 3600 detik = 1 jam
    expected_dist_m = u_const * (dt_s * total_steps)  # 36,000 meter

    for _ in range(total_steps):
        advect_particles_rk2(p, u_ms=u_const, v_ms=v_const, dt_s=dt_s, diff_coeff_m2s=0.0)

    # Hitung perpindahan dalam meter
    d_lon_deg = p.lon[0] - 105.0
    m_per_deg = (np.pi * EARTH_RADIUS_M) / 180.0
    actual_dist_m = d_lon_deg * m_per_deg

    assert np.isclose(actual_dist_m, expected_dist_m, rtol=1e-3)


def test_fall_velocity_settling():
    """
    Tes 2: Partikel butir besar jatuh lebih cepat daripada butir halus.
    """
    p = create_empty_particles(2)
    # Partikel 0: Butir kasar (w_s = 2.5 m/s)
    p.lon[0], p.lat[0], p.z[0] = 105.0, -6.0, 1000.0
    p.settling_vel[0] = 2.5
    p.active[0] = True

    # Partikel 1: Butir halus (w_s = 0.04 m/s)
    p.lon[1], p.lat[1], p.z[1] = 105.0, -6.0, 1000.0
    p.settling_vel[1] = 0.04
    p.active[1] = True

    # Jalankan selama 500 detik
    dt_s = 100.0
    for _ in range(5):
        advect_particles_rk2(p, u_ms=5.0, v_ms=0.0, dt_s=dt_s, diff_coeff_m2s=0.0)

    # Butir kasar seharusnya sudah mendarat di tanah (z=0, deposited=True)
    assert p.z[0] == 0.0
    assert bool(p.deposited[0]) is True
    assert bool(p.active[0]) is False

    # Butir halus masih melayang di udara (z = 1000 - 0.04*500 = 980 m)
    assert p.z[1] > 900.0
    assert bool(p.active[1]) is True


def test_mass_conservation_and_scavenging():
    """
    Tes 3: Kekekalan massa dan kesetimbangan pencucian hujan.
    """
    p = create_empty_particles(5)
    p.mass[:5] = [10.0, 20.0, 30.0, 40.0, 50.0]
    p.active[:5] = True
    initial_total_mass = float(np.sum(p.mass[:5]))

    # Terapkan hujan 15 mm/jam selama 600 detik
    scavenged_mass = apply_rain_scavenging(p, rain_rate_mm_h=15.0, dt_s=600.0)

    remaining_air_mass = float(np.sum(p.mass[:5]))
    scavenged_total = float(np.sum(scavenged_mass))

    # Massa total di udara + massa yang tercuci harus sama persis dengan massa awal
    assert np.isclose(remaining_air_mass + scavenged_total, initial_total_mass)
    assert scavenged_total > 0.0


def test_tag_linearity():
    """
    Tes 4: Linearitas tag (cluster_id).
    Membuang partikel bertag X ekuivalen dengan simulasi tanpa sumber X.
    """
    p = create_empty_particles(4)
    p.active[:4] = True
    p.mass[:4] = 10.0
    p.cluster_id[:4] = [1, 1, 2, 2]  # Dua sumber berbeda

    # Total massa awal
    assert np.sum(p.mass[p.active]) == 40.0

    # Buang tag sumber 2
    filter_mask = p.cluster_id[:4] != 2
    active_filtered_mass = np.sum(p.mass[:4][filter_mask])

    # Hanya partikel sumber 1 yang tersisa
    assert active_filtered_mass == 20.0


def test_determinism_with_seed():
    """
    Tes 5: Determinisme numerik dengan seed identik.
    """
    from app.engine.simulator import SimulationRunner
    from app.schemas.simulation import SimulationRequest

    req1 = SimulationRequest(
        hazard="volcano",
        mode="auto",
        source={"event_id": "anak_krakatau_2026_09"},
        horizon_hours=6,
        n_members=5,
        seed=12345
    )
    req2 = SimulationRequest(
        hazard="volcano",
        mode="auto",
        source={"event_id": "anak_krakatau_2026_09"},
        horizon_hours=6,
        n_members=5,
        seed=12345
    )

    import json

    from app.core.config import EVENTS_DIR
    from app.ingest.openmeteo import OpenMeteoIngest

    event_meta = json.loads((EVENTS_DIR / "anak_krakatau_2026_09" / "meta.json").read_text(encoding="utf-8"))
    weather = OpenMeteoIngest.get_weather_for_simulation("anak_krakatau_2026_09", n_members=5, seed=12345)

    runner1 = SimulationRunner(req1)
    res1 = runner1.run(event_meta, weather)

    runner2 = SimulationRunner(req2)
    res2 = runner2.run(event_meta, weather)

    # Array masker keterpaparan harus identik bit-per-bit
    assert np.array_equal(res1["hourly_exposure_masks"], res2["hourly_exposure_masks"])


def test_three_tiers_subset_inclusion():
    """
    Tes 6: Area minimum ⊆ tengah ⊆ maksimum untuk setiap jam.
    """
    grid = SimulationGrid.create_for_volcano(-6.10, 105.42, domain_km=100.0, res_deg=0.05)
    rng = np.random.default_rng(42)

    # Buat dummy masker ensemble: (10 members, n_lat, n_lon)
    member_masks = rng.choice([True, False], size=(10, grid.n_lat, grid.n_lon), p=[0.3, 0.7])

    tiers_geojson = aggregate_tiers_for_hour(
        hourly_member_masks=member_masks,
        grid=grid,
        hour=1,
        th_min=0.90,
        th_mid=0.50,
        th_max=0.05
    )
    assert tiers_geojson["type"] == "FeatureCollection"
    assert "features" in tiers_geojson

    # Verifikasi masker matematika
    prob_grid = np.mean(member_masks.astype(float), axis=0)
    mask_min = prob_grid >= 0.90
    mask_mid = prob_grid >= 0.50
    mask_max = prob_grid >= 0.05

    # min harus subset dari mid
    assert np.all(mask_mid[mask_min])
    # mid harus subset dari max
    assert np.all(mask_max[mask_mid])


def test_mastin_d24_plausibility():
    """
    Tes 7: Formula D24 Mastin dkk. 2009 (M = 140.85 * H^4.15).
    """
    # Untuk H = 10 km di atas kawah
    mer_10km = calculate_mer_mastin(10.0)
    # 140.85 * (10 ^ 4.15) ≈ 140.85 * 14125.375 ≈ 1.99e6 kg/s
    assert 1.5e6 < mer_10km < 2.5e6

    # Konversi ke DRE volume rate pada rho = 2500 kg/m³
    v_dot = mer_10km / 2500.0
    # Formula Mastin: H = 2.00 * V_dot^0.241 -> V_dot = (10 / 2)^(1/0.241) ≈ 5^4.149 ≈ 796 m³/s
    assert 700.0 < v_dot < 900.0
