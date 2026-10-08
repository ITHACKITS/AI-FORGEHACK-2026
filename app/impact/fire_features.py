"""
app/impact/fire_features.py

Fitur pembeda untuk bahaya kebakaran hutan (Lapis 2):
1. Peringkat klaster titik api menurut kontribusi paparan penduduk (ranking).
2. Fitur padamkan titik api (extinguish) berbasis linearitas tag partikel tanpa simulasi ulang.
3. Estimasi emisi gas karbon (CO2e) dalam rentang, membedakan gambut dan non-gambut.
"""

from typing import Any, Dict, List

import numpy as np
from scipy.spatial import cKDTree

from app.engine.grid import SimulationGrid
from app.ingest.population import PopulationIngest


def rank_hotspot_clusters(
    hotspots: List[Dict[str, Any]],
    run_output: Dict[str, Any],
    grid: SimulationGrid,
    eps_deg: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Mengelompokkan titik api menggunakan cKDTree spasial dan memberi peringkat
    berdasarkan kontribusi terhadap paparan penduduk.
    """
    if not hotspots:
        return []

    coords = np.array([[float(h["latitude"]), float(h["longitude"])] for h in hotspots])
    n_points = len(coords)

    # Pengelompokan berbasis jarak spasial menggunakan cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=eps_deg)

    # Disjoint Set / Connected Components
    parent = list(range(n_points))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i, j in pairs:
        union(i, j)

    labels = [find(i) for i in range(n_points)]

    # Ambil grid populasi
    pop_data = PopulationIngest.get_population_grid(
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max, grid.resolution_deg,
        shape=(grid.n_lat, grid.n_lon)
    )
    pop_grid = pop_data["total"]

    # Ambil masker paparan kumulatif
    hourly_masks = run_output.get("hourly_exposure_masks")  # (n_hours, n_members, n_lat, n_lon)
    if hourly_masks is not None:
        mean_prob_grid = np.mean(np.any(hourly_masks, axis=0).astype(float), axis=0)
    else:
        mean_prob_grid = np.zeros((grid.n_lat, grid.n_lon))

    clusters_dict = {}
    for idx, label in enumerate(labels):
        cid = int(label)
        if cid not in clusters_dict:
            clusters_dict[cid] = {
                "cluster_id": cid,
                "hotspot_indices": [],
                "frp_total": 0.0,
                "lats": [],
                "lons": [],
            }
        clusters_dict[cid]["hotspot_indices"].append(idx)
        clusters_dict[cid]["frp_total"] += float(hotspots[idx].get("frp", 25.0))
        clusters_dict[cid]["lats"].append(float(hotspots[idx]["latitude"]))
        clusters_dict[cid]["lons"].append(float(hotspots[idx]["longitude"]))

    ranked_list = []
    for cid, cinfo in clusters_dict.items():
        center_lat = float(np.mean(cinfo["lats"]))
        center_lon = float(np.mean(cinfo["lons"]))
        r, c = grid.coord_to_index(center_lat, center_lon)

        # Skor kedekatan dengan populasi terancam
        # Ambil jendela 30x30 sel di sekitar titik api
        r_min, r_max = max(0, r - 15), min(grid.n_lat, r + 15)
        c_min, c_max = max(0, c - 15), min(grid.n_lon, c + 15)
        local_pop = np.sum(pop_grid[r_min:r_max, c_min:c_max])
        local_prob = np.mean(mean_prob_grid[r_min:r_max, c_min:c_max])

        exposure_score = float(cinfo["frp_total"] * (1.0 + local_prob) * (local_pop / 1000.0 + 1.0))

        ranked_list.append({
            "cluster_id": cid,
            "center": {"latitude": round(center_lat, 4), "longitude": round(center_lon, 4)},
            "hotspot_count": len(cinfo["hotspot_indices"]),
            "total_frp_mw": round(cinfo["frp_total"], 1),
            "exposure_contribution_score": round(exposure_score, 1),
            "hotspot_ids": [hotspots[i]["id"] for i in cinfo["hotspot_indices"]],
        })

    # Urutkan dari kontribusi terbesar
    ranked_list.sort(key=lambda x: x["exposure_contribution_score"], reverse=True)
    for rank, item in enumerate(ranked_list, start=1):
        item["rank"] = rank

    return ranked_list


def simulate_extinguish_delta(
    run_output: Dict[str, Any],
    extinguish_cluster_ids: List[int],
    grid: SimulationGrid,
    hotspots: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Menghitung penurunan dampak jika klaster titik api tertentu dipadamkan.
    Linearitas tag partikel: dampak berkurang proporsional terhadap kontribusi FRP klaster yang padam.
    """
    # Hitung total FRP awal dan FRP yang dipadamkan
    cluster_ranks = rank_hotspot_clusters(hotspots, run_output, grid)
    extinguish_set = set(extinguish_cluster_ids)

    total_frp = sum(c["total_frp_mw"] for c in cluster_ranks) or 1.0
    extinguished_frp = sum(c["total_frp_mw"] for c in cluster_ranks if c["cluster_id"] in extinguish_set)

    reduction_fraction = min(1.0, extinguished_frp / total_frp)

    # Ambil dampak dasar dari run_output
    from app.impact.exposure import calculate_impact_ranges
    base_impact = calculate_impact_ranges(run_output, grid)

    # Hitung pengurangan populasi, anak, dan sekolah dalam rentang
    def scale_range(range_metric: Dict[str, float], factor: float) -> Dict[str, float]:
        return {
            "min": round(range_metric["min"] * factor, 1),
            "median": round(range_metric["median"] * factor, 1),
            "max": round(range_metric["max"] * factor, 1),
        }

    pop_reduced = scale_range(base_impact["population_exposed"], reduction_fraction)
    child_reduced = scale_range(base_impact["children_exposed"], reduction_fraction)
    schools_saved = scale_range(base_impact["schools_exposed"], reduction_fraction)

    # Penurunan emisi karbon
    carbon_base = calculate_carbon_emissions(hotspots, duration_hours=48.0)
    carbon_reduced = scale_range(carbon_base["total_co2e_tonnes"], reduction_fraction)

    return {
        "extinguished_clusters": extinguish_cluster_ids,
        "extinguished_frp_mw": round(extinguished_frp, 1),
        "reduction_percentage": round(reduction_fraction * 100.0, 1),
        "population_reduced": pop_reduced,
        "children_protected": child_reduced,
        "schools_saved": schools_saved,
        "carbon_reduced_co2e_tonnes": carbon_reduced,
        "caveat": "Perhitungan cepat berbasis linearitas tag massa partikel tanpa simulasi ulang penuh."
    }


def calculate_carbon_emissions(
    hotspots: List[Dict[str, Any]],
    duration_hours: float = 48.0,
    peat_fraction: float = 0.40
) -> Dict[str, Any]:
    """
    D12/D13: Estimasi gas karbon (CO2e) dalam rentang {min, median, max}.
    Massa biomassa terbakar (kg) = FRP (MW) * waktu (s) * faktor konversi (0.27 - 0.45 kg/MJ).
    Emisi karbon: ~0.47 fraksi C dalam biomassa kering * (44/12) faktor konversi ke CO2.
    Gambut memiliki densitas karbon dan faktor emisi lebih tinggi (pengali ~1.6).
    """
    total_frp = sum(float(h.get("frp", 25.0)) for h in hotspots if h.get("is_active", True))
    duration_s = duration_hours * 3600.0

    # Faktor Wooster (0.27 / 0.37 / 0.45 kg/MJ)
    frp_factors = {"min": 0.27, "median": 0.37, "max": 0.45}
    carbon_fraction = 0.47
    co2_factor = carbon_fraction * (44.0 / 12.0)  # ≈ 1.72 kg CO2 per kg dry matter

    results_total = {}
    results_peat = {}
    results_non_peat = {}

    for k in ["min", "median", "max"]:
        factor = frp_factors[k]
        # Total biomassa terbakar (kg)
        dry_matter_kg = total_frp * duration_s * factor
        # Emisi CO2 dalam metrik ton (kg / 1000)
        co2_tonnes = (dry_matter_kg * co2_factor) / 1000.0

        # Pembagian gambut (faktor emisi smoldering lebih tinggi 1.6x)
        peat_tonnes = co2_tonnes * peat_fraction * 1.6
        non_peat_tonnes = co2_tonnes * (1.0 - peat_fraction)

        results_total[k] = round(peat_tonnes + non_peat_tonnes, 1)
        results_peat[k] = round(peat_tonnes, 1)
        results_non_peat[k] = round(non_peat_tonnes, 1)

    return {
        "total_co2e_tonnes": results_total,
        "peat_emissions_tonnes": results_peat,
        "non_peat_emissions_tonnes": results_non_peat,
        "unit": "metric_tonnes_co2e",
        "duration_hours": duration_hours,
        "source": "Wooster dkk. 2005 & Li dkk. 2018 (D13)",
        "caveat": "Rentang dihitung dari variasi faktor FRP-ke-biomassa 0.27-0.45 kg/MJ dan fraksi gambut."
    }
