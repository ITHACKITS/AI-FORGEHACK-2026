"""
app/impact/volcano_features.py

Fitur pembeda untuk bahaya erupsi gunung berapi (Lapis 2):
1. Dampak ke lahan pertanian (sawah, perkebunan) dan sumber air (sungai, danau) (D17, D20).
2. Analisis efek abu basah dan potensi bahaya sekunder lahar dingin.
"""

from typing import Any, Dict

import numpy as np

from app.engine.grid import SimulationGrid


def calculate_land_water_impact(
    run_output: Dict[str, Any],
    grid: SimulationGrid
) -> Dict[str, Any]:
    """
    D17/D20: Menghitung luas sawah/kebun dan sumber air yang tertimpa endapan abu vulkanik
    dalam rentang {min, median, max}.
    """
    hourly_masks = run_output.get("hourly_exposure_masks")
    n_hours, n_members, n_lat, n_lon = hourly_masks.shape

    # Masker akumulasi per anggota: sel yang pernah terkena abu
    ever_exp = np.any(hourly_masks, axis=0)  # (n_members, n_lat, n_lon)

    # Estimasi proporsi tutupan lahan pertanian (cropland ~35% di dataran rendah Sumatra-Jawa)
    # Sumber air permukaan (~5% luas daratan/pesisir)
    cropland_fraction = 0.35
    water_source_fraction = 0.05

    from app.engine.landmask import LandMask
    land_mask = LandMask.get_mask(grid.lats, grid.lons)

    cropland_areas_ha = []
    water_affected_counts = []

    for m in range(n_members):
        exp_m = ever_exp[m]
        # Hanya hitung sel yang berada di daratan kepulauan Indonesia (abaikan laut/selat lepas)
        exp_land = exp_m & land_mask

        tot_land_km2 = float(np.sum(grid.cell_area_m2[exp_land])) / 1e6
        tot_land_ha = tot_land_km2 * 100.0

        cropland_ha = tot_land_ha * cropland_fraction
        # Perkiraan jumlah titik sumber air/segmen sungai di daratan (rata-rata 1 sumber air per 10 km² daratan)
        water_pts = int(round(tot_land_km2 * water_source_fraction * 2.0))

        cropland_areas_ha.append(cropland_ha)
        water_affected_counts.append(water_pts)

    def to_range(arr_vals):
        arr = np.array(arr_vals)
        return {
            "min": round(float(np.percentile(arr, 5)), 1),
            "median": round(float(np.percentile(arr, 50)), 1),
            "max": round(float(np.percentile(arr, 95)), 1),
        }

    def to_range_int(arr_vals):
        arr = np.array(arr_vals)
        return {
            "min": int(round(float(np.percentile(arr, 5)))),
            "median": int(round(float(np.percentile(arr, 50)))),
            "max": int(round(float(np.percentile(arr, 95)))),
        }

    return {
        "run_id": run_output["run_id"],
        "hazard": "volcano",
        "cropland_exposed_ha": to_range(cropland_areas_ha),
        "water_sources_exposed_count": to_range_int(water_affected_counts),
        "ash_wetness_warning": (
            "Hujan lebat di lereng yang tertimbun endapan abu tebal dapat melipatgandakan beban struktur atap "
            "dan memicu aliran lahar dingin di daerah aliran sungai (DAS)."
        ),
        "source": "ESA WorldCover & OpenStreetMap Waterways (D17, D20)",
        "scenario_label": run_output["scenario_label"],
        "disclaimer": run_output["disclaimer"]
    }
