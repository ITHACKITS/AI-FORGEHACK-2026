"""
app/engine/simulator.py

Simulator terintegrasi partikel Lagrangian untuk kedua bahaya (Gunung & Kebakaran).
Mendukung 50-100 anggota ensemble, deterministik berbasis seed, dan output per jam.
"""

import datetime
import json
import uuid
from typing import Any, Dict, List

import numpy as np

from app.core.config import RUNS_DIR, get_assumptions_used, settings
from app.core.rng import get_rng
from app.engine.aggregation import aggregate_tiers_for_hour, compute_arrival_time_grid
from app.engine.fire import generate_fire_particles_per_step, sample_frp_conversion_factor
from app.engine.grid import SimulationGrid
from app.engine.particles import advect_particles_rk2, create_empty_particles
from app.engine.scavenging import apply_rain_scavenging
from app.engine.volcano import calculate_mer_mastin, generate_volcano_particles_per_step
from app.schemas.simulation import SimulationRequest


class SimulationRunner:
    def __init__(self, request: SimulationRequest):
        self.request = request
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.rng = get_rng(request.seed)
        self.created_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def run(
        self,
        event_meta: Dict[str, Any],
        weather_data: Dict[str, Any],
        hotspots: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """
        Menjalankan simulasi penuh untuk seluruh anggota ensemble.
        """
        hazard = self.request.hazard
        n_members = self.request.n_members
        horizon_hours = min(self.request.horizon_hours, 48)
        dt_s = 300.0  # 5 menit
        steps_per_hour = int(3600 / dt_s)
        total_steps = horizon_hours * steps_per_hour

        # 1. Inisialisasi Grid
        if hazard == "volcano":
            coords = event_meta.get("coordinates", {"latitude": -6.102, "longitude": 105.423})
            center_lat = float(coords.get("latitude", -6.102))
            center_lon = float(coords.get("longitude", 105.423))
            # Domain 900 km (radius 450 km) agar partikel beradveksi bebas tanpa terpotong batas grid sebelum konsentrasi meluruh
            grid = SimulationGrid.create_for_volcano(center_lat, center_lon, domain_km=900.0, res_deg=0.03)
        else:
            bbox = event_meta.get("bbox", [110.0, -4.0, 116.0, 1.0])
            center_lat = (float(bbox[1]) + float(bbox[3])) / 2.0
            center_lon = (float(bbox[0]) + float(bbox[2])) / 2.0
            grid = SimulationGrid.create_for_fire(bbox, buffer_km=350.0, res_deg=0.03)

        wind_u_arr = weather_data.get("wind_u")  # shape (n_members, n_weather_steps)
        wind_v_arr = weather_data.get("wind_v")
        precip_arr = weather_data.get("precip")

        # 2. Persiapan Parameter Sumber
        if hazard == "volcano":
            vent_lat = float(event_meta.get("coordinates", {}).get("latitude", -6.102))
            vent_lon = float(event_meta.get("coordinates", {}).get("longitude", 105.423))
            vent_elev_m = float(event_meta.get("vent_elevation_m", 110.0))

            # Override kolom jika ada
            if self.request.overrides and self.request.overrides.get("column_height_km"):
                plume_top_asl_km = float(self.request.overrides["column_height_km"])
            else:
                plume_top_asl_km = float(event_meta.get("plume_top_asl_km") or 10.0)

            duration_h = float(
                self.request.overrides.get("duration_h", event_meta.get("duration_h", 25.0))
                if self.request.overrides else event_meta.get("duration_h", 25.0)
            )
            peak_phase_h = float(event_meta.get("peak_phase_duration_h", 6.0))

            # D24: H di atas kawah
            h_vent_km = max(0.1, plume_top_asl_km - (vent_elev_m / 1000.0))
            base_mer_kg_s = calculate_mer_mastin(h_vent_km)
            plume_top_m = plume_top_asl_km * 1000.0

        # Masker terkena: (horizon_hours, n_members, n_lat, n_lon)
        hourly_exposure_masks = np.zeros((horizon_hours, n_members, grid.n_lat, grid.n_lon), dtype=bool)
        member_max_severity = np.zeros((n_members, grid.n_lat, grid.n_lon), dtype=float)

        particles_per_step = 40  # Keseimbangan performa & resolusi
        max_particles_capacity = 25000

        # 3. Loop per Anggota Ensemble
        for m in range(n_members):
            member_rng = np.random.default_rng(self.request.seed + m * 97)
            particles = create_empty_particles(max_particles_capacity)
            particle_cursor = 0

            # Matriks akumulasi endapan abu (kg/m²) atau konsentrasi PM2.5 (µg/m³)
            deposited_mass_grid = np.zeros((grid.n_lat, grid.n_lon), dtype=float)

            if hazard == "volcano":
                # Lognormal draw laju emisi per anggota (faktor 4 ketidakpastian)
                mer_member = member_rng.lognormal(mean=np.log(base_mer_kg_s), sigma=0.5)
            else:
                # Faktor FRP-to-dry-matter per anggota (D13 segitiga)
                frp_factor = sample_frp_conversion_factor(member_rng)

            # Loop Waktu Simulasi
            for step in range(total_steps):
                current_time_s = step * dt_s
                current_hour = int(step // steps_per_hour)
                weather_step_idx = min(current_hour, wind_u_arr.shape[1] - 1) if wind_u_arr is not None else 0

                u_cur = float(wind_u_arr[m, weather_step_idx]) if wind_u_arr is not None else 4.0
                v_cur = float(wind_v_arr[m, weather_step_idx]) if wind_v_arr is not None else 2.0
                rain_cur = float(precip_arr[m, weather_step_idx]) if precip_arr is not None else 0.0

                # 3a. Pelepasan Partikel Baru
                if hazard == "volcano":
                    if current_time_s < duration_h * 3600.0:
                        # Skala fase puncak vs background
                        phase_scale = 1.0 if current_time_s < (peak_phase_h * 3600.0) else 0.15
                        step_mass = mer_member * phase_scale * dt_s
                        inserted = generate_volcano_particles_per_step(
                            p=particles,
                            start_slot=particle_cursor,
                            n_particles=particles_per_step,
                            vent_lat=vent_lat,
                            vent_lon=vent_lon,
                            vent_elev_m=vent_elev_m,
                            plume_top_m=plume_top_m,
                            total_mass_step_kg=step_mass,
                            rng=member_rng,
                        )
                        particle_cursor = (particle_cursor + inserted) % (max_particles_capacity - particles_per_step)
                else:
                    if hotspots:
                        inserted = generate_fire_particles_per_step(
                            p=particles,
                            start_slot=particle_cursor,
                            hotspots=hotspots,
                            dt_s=dt_s,
                            particles_per_hotspot=max(1, 100 // len(hotspots)),
                            frp_factor_kg_mj=frp_factor,
                            rng=member_rng,
                        )
                        particle_cursor = (particle_cursor + inserted) % (max_particles_capacity - 200)

                # 3b. Pencucian oleh Hujan
                if rain_cur > 0.05:
                    scavenged_mass = apply_rain_scavenging(particles, rain_cur, dt_s)
                    if len(scavenged_mass) > 0:
                        act_idx = np.where(particles.active)[0][:len(scavenged_mass)]
                        in_b = (
                            (particles.lat[act_idx] >= grid.lat_min) & (particles.lat[act_idx] <= grid.lat_max) &
                            (particles.lon[act_idx] >= grid.lon_min) & (particles.lon[act_idx] <= grid.lon_max)
                        )
                        valid_act = act_idx[in_b]
                        if len(valid_act) > 0:
                            rows, cols = grid.coord_to_index(particles.lat[valid_act], particles.lon[valid_act])
                            np.add.at(deposited_mass_grid, (rows, cols), scavenged_mass[in_b])

                # 3c. Adveksi & Difusi
                advect_particles_rk2(
                    p=particles,
                    u_ms=u_cur,
                    v_ms=v_cur,
                    dt_s=dt_s,
                    diff_coeff_m2s=80.0,
                    rng=member_rng,
                )

                # Partikel yang keluar domain regional dinonaktifkan (bebas dari efek kotak batas)
                out_bounds = (
                    (particles.lat < grid.lat_min) | (particles.lat > grid.lat_max) |
                    (particles.lon < grid.lon_min) | (particles.lon > grid.lon_max)
                )
                particles.active[out_bounds] = False

                # Endapkan partikel yang mendarat ke grid tanah (hanya di dalam domain)
                dep_mask = particles.deposited & (particles.mass > 0)
                if np.any(dep_mask):
                    d_idx = np.where(dep_mask)[0]
                    in_b = (
                        (particles.lat[d_idx] >= grid.lat_min) & (particles.lat[d_idx] <= grid.lat_max) &
                        (particles.lon[d_idx] >= grid.lon_min) & (particles.lon[d_idx] <= grid.lon_max)
                    )
                    valid_d = d_idx[in_b]
                    if len(valid_d) > 0:
                        rows, cols = grid.coord_to_index(particles.lat[valid_d], particles.lon[valid_d])
                        np.add.at(deposited_mass_grid, (rows, cols), particles.mass[valid_d])
                    particles.mass[d_idx] = 0.0  # Telah terendapkan

                # 3d. Evaluasi di Akhir Setiap Jam
                if (step + 1) % steps_per_hour == 0:
                    eval_h = current_hour
                    if eval_h < horizon_hours:
                        if hazard == "volcano":
                            # Beban endapan kumulatif (kg/m²)
                            load_grid = deposited_mass_grid / grid.cell_area_m2
                            # Ambang terkena abu: >= 0.005 kg/m² (~0.005 mm tephra tipis teramati)
                            exposed_mask = load_grid >= 0.005
                            hourly_exposure_masks[eval_h, m] = exposed_mask
                            member_max_severity[m] = np.maximum(member_max_severity[m], load_grid)
                        else:
                            # Konsentrasi PM2.5 di udara aktif (µg/m³)
                            act_idx = np.where(particles.active)[0]
                            conc_grid = np.zeros((grid.n_lat, grid.n_lon), dtype=float)
                            if len(act_idx) > 0:
                                in_b = (
                                    (particles.lat[act_idx] >= grid.lat_min) & (particles.lat[act_idx] <= grid.lat_max) &
                                    (particles.lon[act_idx] >= grid.lon_min) & (particles.lon[act_idx] <= grid.lon_max)
                                )
                                valid_act = act_idx[in_b]
                                if len(valid_act) > 0:
                                    rows, cols = grid.coord_to_index(particles.lat[valid_act], particles.lon[valid_act])
                                    # Massa dalam kg diubah ke µg (* 1e9)
                                    np.add.at(conc_grid, (rows, cols), particles.mass[valid_act] * 1e9)
                                    # Bagi dengan volume sel (luas sel * tinggi mixed layer 1200 m)
                                    conc_grid /= (grid.cell_area_m2 * 1200.0)

                            # Ambang paparan PM2.5: >= 15.5 µg/m³ (Batas kategori 'Sedang' ISPU)
                            exposed_mask = conc_grid >= 15.5
                            hourly_exposure_masks[eval_h, m] = exposed_mask
                            member_max_severity[m] = np.maximum(member_max_severity[m], conc_grid)

        # 4. Agregasi Tiers & Arrival Times
        th_min = self.request.thresholds.min
        th_mid = self.request.thresholds.mid
        th_max = self.request.thresholds.max

        hourly_tiers_geojson = {}
        for h in range(horizon_hours):
            hourly_tiers_geojson[h] = aggregate_tiers_for_hour(
                hourly_member_masks=hourly_exposure_masks[h],
                grid=grid,
                hour=h,
                th_min=th_min,
                th_mid=th_mid,
                th_max=th_max,
            )

        earliest_arr, likely_arr = compute_arrival_time_grid(hourly_exposure_masks)

        # 5. Susun Metadata Hasil & Simpan
        scenario_label = (
            "skenario hipotetis" if self.request.mode == "experiment" else "kejadian observasi"
        )
        members_source = weather_data.get("members_source", "open_meteo_ensemble")

        def _clean_arrival_matrix(matrix: np.ndarray) -> list:
            clean = []
            for row in matrix:
                clean.append([None if np.isnan(val) else round(float(val), 2) for val in row])
            return clean

        run_output = {
            "run_id": self.run_id,
            "hazard": hazard,
            "mode": self.request.mode,
            "event_id": event_meta.get("event_id") or self.request.source.event_id,
            "status": "completed",
            "scenario_label": scenario_label,
            "members_source": members_source,
            "n_members": n_members,
            "seed": self.request.seed,
            "created_at_utc": self.created_at_utc,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "disclaimer": settings.DISCLAIMER,
            "center_coords": {"latitude": center_lat, "longitude": center_lon},
            "assumptions_used": get_assumptions_used(),
            "grid": {
                "lat_min": grid.lat_min,
                "lat_max": grid.lat_max,
                "lon_min": grid.lon_min,
                "lon_max": grid.lon_max,
                "resolution_deg": grid.resolution_deg,
            },
            "hourly_tiers": hourly_tiers_geojson,
            "member_max_severity": member_max_severity,
            "hourly_exposure_masks": hourly_exposure_masks,
            "arrival_time": {
                "earliest": _clean_arrival_matrix(earliest_arr),
                "most_likely": _clean_arrival_matrix(likely_arr),
            }
        }

        # Precompute impact ranges & fasilitas terdampak agar selalu tersimpan di berkas JSON disk
        from app.impact.exposure import calculate_impact_ranges, get_exposed_facilities_list
        run_output["impact"] = calculate_impact_ranges(run_output, grid)
        try:
            run_output["facilities_impacted"] = get_exposed_facilities_list(run_output, grid)
        except Exception as e:
            print(f"Warning: gagal precompute facilities_impacted: {e}")
            run_output["facilities_impacted"] = {}

        # Simpan ke data/runs/{run_id}.json
        save_path = RUNS_DIR / f"{self.run_id}.json"
        # Buat representasi json yang ringan tanpa array numpy raksasa
        json_summary = {k: v for k, v in run_output.items() if k not in ["hourly_exposure_masks", "member_max_severity"]}
        save_path.write_text(json.dumps(json_summary, default=str), encoding="utf-8")

        return run_output
