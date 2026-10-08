"""
app/impact/comparative.py

Fitur komparasi, analisis efek hujan, sensitivitas skenario, dan saran tindakan (Lapis 2).
Mematuhi batasan bahwa saran tindakan abu HANYA mencantumkan entri terverifikasi (D16).
"""

from typing import Any, Dict

from app.engine.grid import SimulationGrid
from app.impact.exposure import calculate_impact_ranges


def calculate_rain_effect(
    run_output: Dict[str, Any],
    grid: SimulationGrid
) -> Dict[str, Any]:
    """
    Menghitung pengaruh hujan pada sebaran abu atau asap.
    Membandingkan kondisi saat hujan turun vs tanpa pencucian hujan.
    """
    hazard = run_output["hazard"]
    base_impact = calculate_impact_ranges(run_output, grid)

    # Estimasi fraksi pencucian rata-rata (hujan menurunkan area jangkauan jauh ~15-30%)
    scavenging_reduction_pct = 0.22

    def scale_delta(range_metric: Dict[str, float], factor: float) -> Dict[str, float]:
        return {
            "min": round(range_metric["min"] * factor, 1),
            "median": round(range_metric["median"] * factor, 1),
            "max": round(range_metric["max"] * factor, 1),
        }

    pop_reduced = scale_delta(base_impact["population_exposed"], scavenging_reduction_pct)
    area_reduced = scale_delta(base_impact["area_exposed_km2"], scavenging_reduction_pct)

    if hazard == "volcano":
        caveat_text = (
            "Hujan pencuci mempercepat jatuhnya partikel abu halus lebih dekat ke kawah. "
            "PERINGATAN: Abu basah jauh lebih berat di atap bangunan dan dapat memicu aliran lahar dingin di sungai."
        )
    else:
        caveat_text = (
            "Hujan ringan hanya membersihkan partikel PM2.5 di udara untuk sementara. "
            "Api gambut di bawah permukaan tanah memerlukan hujan deras berdurasi panjang agar benar-benar padam."
        )

    return {
        "run_id": run_output["run_id"],
        "hazard": hazard,
        "population_exposure_reduced": pop_reduced,
        "area_footprint_reduced_km2": area_reduced,
        "scavenging_reduction_percentage": round(scavenging_reduction_pct * 100.0, 1),
        "hazard_specific_warning": caveat_text,
        "scenario_label": run_output["scenario_label"],
        "disclaimer": run_output["disclaimer"]
    }


def compare_two_runs(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
    grid: SimulationGrid = None,
    grid2: SimulationGrid = None
) -> Dict[str, Any]:
    """
    Membandingkan dua simulasi secara berdampingan (side-by-side comparison).
    Masing-masing run dievaluasi dengan geometrinya sendiri agar terbebas dari dimension mismatch.
    """
    if grid is None and "grid" in run1:
        g1 = run1["grid"]
        grid = SimulationGrid(g1["lat_min"], g1["lat_max"], g1["lon_min"], g1["lon_max"], g1["resolution_deg"])
    if grid2 is None:
        if "grid" in run2:
            g2 = run2["grid"]
            grid2 = SimulationGrid(g2["lat_min"], g2["lat_max"], g2["lon_min"], g2["lon_max"], g2["resolution_deg"])
        else:
            grid2 = grid

    imp1 = calculate_impact_ranges(run1, grid)
    imp2 = calculate_impact_ranges(run2, grid2)

    def diff_metric(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        return {
            "min_diff": round(m2["min"] - m1["min"], 1),
            "median_diff": round(m2["median"] - m1["median"], 1),
            "max_diff": round(m2["max"] - m1["max"], 1),
        }

    return {
        "run_a": {
            "run_id": run1["run_id"],
            "mode": run1["mode"],
            "scenario_label": run1["scenario_label"],
            "impact": imp1
        },
        "run_b": {
            "run_id": run2["run_id"],
            "mode": run2["mode"],
            "scenario_label": run2["scenario_label"],
            "impact": imp2
        },
        "deltas": {
            "population_exposed_diff": diff_metric(imp1["population_exposed"], imp2["population_exposed"]),
            "area_exposed_diff_km2": diff_metric(imp1["area_exposed_km2"], imp2["area_exposed_km2"]),
            "schools_exposed_diff": diff_metric(imp1["schools_exposed"], imp2["schools_exposed"]),
        },
        "disclaimer": run1["disclaimer"]
    }


def run_parameter_sensitivity(
    base_run_output: Dict[str, Any],
    param_name: str,
    grid: SimulationGrid
) -> Dict[str, Any]:
    """
    Menghasilkan sapuan kurva sensitivitas parameter (mis. tinggi kolom 5 s/d 18 km).
    """
    base_impact = calculate_impact_ranges(base_run_output, grid)
    base_median_pop = base_impact["population_exposed"]["median"]
    base_median_area = base_impact["area_exposed_km2"]["median"]

    sweep_curve = []
    if param_name == "column_height_km":
        steps = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0]
        # Mengikuti laju Mastin H^4.15 secara proporsional
        for h in steps:
            factor = (h / 10.0) ** 1.8  # Skala dispersi area terhadap laju emisi
            sweep_curve.append({
                "parameter_value": h,
                "unit": "km",
                "median_population_exposed": round(base_median_pop * factor, 1),
                "median_area_km2": round(base_median_area * factor, 1),
            })
    elif param_name == "rain_intensity_mm_h":
        steps = [0.0, 5.0, 10.0, 20.0, 30.0]
        for r_int in steps:
            # Pengurangan area akibat pencucian hujan L = a * P^0.8
            reduction = min(0.6, (r_int ** 0.8) * 0.04)
            sweep_curve.append({
                "parameter_value": r_int,
                "unit": "mm/h",
                "median_population_exposed": round(base_median_pop * (1.0 - reduction), 1),
                "median_area_km2": round(base_median_area * (1.0 - reduction), 1),
            })
    else:
        # Kurva baseline untuk parameter tunggal
        sweep_curve = [{"parameter_value": 1.0, "median_population_exposed": base_median_pop, "median_area_km2": base_median_area}]

    return {
        "run_id": base_run_output["run_id"],
        "parameter_swept": param_name,
        "sensitivity_curve": sweep_curve,
        "sweep_results": sweep_curve,
        "caveat": "Kurva sensitivitas menunjukkan elastisitas dampak terhadap perubahan satu parameter.",
        "disclaimer": base_run_output["disclaimer"]
    }


def get_action_advice(run_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    D16: Saran tindakan terverifikasi.
    Abu: Hanya entri bersumber (Badan Geologi 7 Sept), selebihnya available: false.
    Asap: Indeks paparan relatif.
    """
    hazard = run_output["hazard"]
    disclaimer = run_output["disclaimer"]

    if hazard == "volcano":
        return {
            "hazard": "volcano",
            "sourced_advice": [
                {
                    "action": "Mengurangi aktivitas luar rumah dan memakai masker penutup hidung dan mulut",
                    "target_audience": "Masyarakat di wilayah terdampak hujan abu lebat",
                    "status": "sourced",
                    "source": "Laporan Badan Geologi Kementerian ESDM tanggal 7 September 2026",
                    "source_url": "https://geologi.esdm.go.id/media-center/perkembangan-erupsi-gunungapi-anak-krakatau-tanggal-7-september-2026",
                    "verified_on": "2026-10-05"
                }
            ],
            "additional_guidance": {
                "available": False,
                "reason": "Saran tindakan mitigasi lanjutan (evakuasi, penutupan bandara) memerlukan arahan otoritas resmi BPBD/PVMBG dan tidak digeneralisasi otomatis oleh simulasi."
            },
            "disclaimer": disclaimer
        }
    else:
        return {
            "hazard": "fire",
            "sourced_advice": [
                {
                    "action": "Membatasi aktivitas luar ruang untuk kelompok sensitif (anak-anak dan lansia)",
                    "target_audience": "Wilayah dengan perkiraan paparan asap pekat",
                    "status": "sourced_secondary",
                    "source": "Pedoman Kualitas Udara ISPU KLHK (Permen LHK P.14/2020)",
                    "verified_on": "2026-10-05"
                }
            ],
            "additional_guidance": {
                "available": False,
                "reason": "Kategori ISPU definitif menunggu tersedianya faktor emisi PM2.5 berstatus sourced* (Prinsip 11 AGENTS.md)."
            },
            "disclaimer": disclaimer
        }
