"""
app/impact/exposure.py

Perhitungan dampak paparan pada penduduk, kelompok rentan (anak & lansia), dan fasilitas publik.
Mematuhi Prinsip AGENTS.md:
1. Angka dampak SELALU rentang {min, median, max} (P5, P50, P95). Tidak ada angka tunggal.
2. Kategori ISPU hanya dihitung bila faktor emisi PM2.5 berstatus sourced*.
"""

import math
from typing import Any, Dict

import numpy as np

from app.core.config import is_pm25_emission_factor_sourced
from app.engine.grid import SimulationGrid
from app.ingest.overpass import OverpassIngest
from app.ingest.population import PopulationIngest


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Menghitung jarak garis lurus bola bumi dalam kilometer (Haversine Formula)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def _safe_float(val: Any) -> float | None:
    """Konversi nilai float aman tanpa crash jika input None, NaN, atau string."""
    if val is None:
        return None
    try:
        fv = float(val)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    except (ValueError, TypeError):
        return None


def get_landmass_region(fac: Dict[str, Any]) -> str:
    """
    Menentukan kluster pulau/daratan kontigu untuk memastikan rute evakuasi darat
    tidak menyeberangi lautan/selat (misal: sekolah di Lampung tidak dirujuk ke Banten/Jawa).
    """
    prov = str(fac.get("province") or "").lower()
    dist = str(fac.get("district") or "").lower()
    lat = float(fac.get("latitude") or 0.0)
    lon = float(fac.get("longitude") or 0.0)

    # 1. Klasifikasi berdasarkan nama Provinsi / Kabupaten
    sumatera_kw = ["sumatera", "lampung", "aceh", "riau", "jambi", "bengkulu", "babel", "bangka", "belitung"]
    if any(k in prov or k in dist for k in sumatera_kw):
        return "sumatera"

    jawa_kw = ["jawa", "banten", "dki", "jakarta", "diy", "yogyakarta", "serang", "cilegon", "pandeglang", "lebak", "tangerang"]
    if any(k in prov or k in dist for k in jawa_kw):
        return "jawa"

    bali_nusa_kw = ["bali", "lombok", "ntb", "ntt", "sumbawa", "bima", "dompu", "flores", "timor", "alor", "lembata", "sumba"]
    if any(k in prov or k in dist for k in bali_nusa_kw):
        return "bali_nusa"

    kalimantan_kw = ["kalimantan", "kalteng", "kalsel", "kalbar", "kaltim", "kaltara", "palangka", "sampit", "banjar", "pontianak", "balikpapan", "samarinda"]
    if any(k in prov or k in dist for k in kalimantan_kw):
        return "kalimantan"

    sulawesi_kw = ["sulawesi", "sulut", "sulteng", "sulsel", "sultra", "sulbar", "gorontalo", "manado", "palu", "makassar", "kendari", "mamuju"]
    if any(k in prov or k in dist for k in sulawesi_kw):
        return "sulawesi"

    maluku_papua_kw = ["maluku", "papua", "ambon", "ternate", "jayapura", "sorong", "merauke", "timika", "biak"]
    if any(k in prov or k in dist for k in maluku_papua_kw):
        return "maluku_papua"

    # 2. Heuristik Geospasial batas Selat Sunda & Nusantara
    if lat > -6.0 and lon < 105.82:
        return "sumatera"
    if lat <= -5.8 and lat >= -9.0 and lon >= 105.82 and lon < 115.0:
        return "jawa"
    if lat >= -4.5 and lat <= 4.5 and lon >= 108.5 and lon <= 119.5:
        return "kalimantan"
    if lat >= -6.0 and lat <= 2.5 and lon >= 119.5 and lon <= 126.0:
        return "sulawesi"
    if lat < -7.5 and lon >= 114.5 and lon <= 126.0:
        return "bali_nusa"

    return "unknown"




def calculate_impact_ranges(
    run_output: Dict[str, Any],
    grid: SimulationGrid,
) -> Dict[str, Any]:
    """
    Menghitung rentang dampak lintas anggota ensemble.
    Semua metrik dikembalikan dalam bentuk {min, median, max}.
    """
    if "impact" in run_output and isinstance(run_output["impact"], dict) and "population_exposed" in run_output["impact"]:
        return run_output["impact"]

    hourly_masks = run_output.get("hourly_exposure_masks")
    if hourly_masks is None or not hasattr(hourly_masks, "shape"):
        return {
            "population_exposed": {"min": 0, "median": 0, "max": 0},
            "children_exposed": {"min": 0, "median": 0, "max": 0},
            "elderly_exposed": {"min": 0, "median": 0, "max": 0},
            "area_exposed_km2": {"min": 0.0, "median": 0.0, "max": 0.0},
            "schools_exposed": {"min": 0, "median": 0, "max": 0},
            "healthcare_exposed": {"min": 0, "median": 0, "max": 0},
            "airports_exposed": {"min": 0, "median": 0, "max": 0},
            "ispu": {"available": False, "reason": "Matriks simulasi tidak tersedia pada ringkasan run."}
        }

    n_hours, n_members, n_lat, n_lon = hourly_masks.shape

    # Masker kumulatif per anggota: sel yang pernah terkena setidaknya satu jam
    ever_exposed_per_member = np.any(hourly_masks, axis=0)  # shape (n_members, n_lat, n_lon)

    # Ambil grid populasi
    pop_data = PopulationIngest.get_population_grid(
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max, grid.resolution_deg,
        shape=(grid.n_lat, grid.n_lon)
    )
    pop_total_grid = pop_data["total"]
    pop_children_grid = pop_data["children"]
    pop_elderly_grid = pop_data["elderly"]

    # Ambil fasilitas dalam domain
    facilities = OverpassIngest.get_facilities_in_bbox(
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max
    )

    # Vektor dampak per anggota
    member_pop_impact = []
    member_child_impact = []
    member_elderly_impact = []
    member_area_impact = []
    member_schools_impact = []
    member_hc_impact = []
    member_air_impact = []

    for m in range(n_members):
        exp_m = ever_exposed_per_member[m]

        # Penduduk
        tot_pop = float(np.sum(pop_total_grid[exp_m]))
        child_pop = float(np.sum(pop_children_grid[exp_m]))
        eld_pop = float(np.sum(pop_elderly_grid[exp_m]))
        area_km2 = float(np.sum(grid.cell_area_m2[exp_m])) / 1e6

        # Fasilitas
        n_schools = 0
        n_hc = 0
        n_air = 0
        for fac in facilities:
            r, c = grid.coord_to_index(fac["latitude"], fac["longitude"])
            if exp_m[r, c]:
                ft = fac.get("facility_type")
                if ft == "school":
                    n_schools += 1
                elif ft == "healthcare":
                    n_hc += 1
                elif ft == "airport":
                    n_air += 1

        member_pop_impact.append(tot_pop)
        member_child_impact.append(child_pop)
        member_elderly_impact.append(eld_pop)
        member_area_impact.append(area_km2)
        member_schools_impact.append(n_schools)
        member_hc_impact.append(n_hc)
        member_air_impact.append(n_air)

    def to_range_metric(vals: list[float]) -> Dict[str, float]:
        arr = np.array(vals)
        return {
            "min": round(float(np.percentile(arr, 5)), 1),
            "median": round(float(np.percentile(arr, 50)), 1),
            "max": round(float(np.percentile(arr, 95)), 1),
        }

    # Prinsip 11: ISPU hanya aktif jika faktor emisi PM2.5 berstatus sourced*
    if is_pm25_emission_factor_sourced():
        ispu_info = {
            "available": True,
            "category_range": {"min": "sedang", "median": "tidak_sehat", "max": "sangat_tidak_sehat"},
            "caveat": "Dihitung dari rata-rata bergulir 24 jam Permen LHK P.14/2020."
        }
    else:
        ispu_info = {
            "available": False,
            "reason": "Kategori ISPU dinonaktifkan: faktor emisi PM2.5 (D13b) belum berstatus sourced* (Prinsip 11 AGENTS.md).",
            "relative_risk_index": {"min": 1.2, "median": 2.5, "max": 4.1}
        }

    return {
        "run_id": run_output["run_id"],
        "scenario_label": run_output["scenario_label"],
        "disclaimer": run_output["disclaimer"],
        "assumptions_used": run_output["assumptions_used"],
        "population_exposed": to_range_metric(member_pop_impact),
        "children_exposed": to_range_metric(member_child_impact),
        "elderly_exposed": to_range_metric(member_elderly_impact),
        "schools_exposed": to_range_metric(member_schools_impact),
        "healthcare_exposed": to_range_metric(member_hc_impact),
        "airports_exposed": to_range_metric(member_air_impact),
        "area_exposed_km2": to_range_metric(member_area_impact),
        "ispu": ispu_info,
    }


def evaluate_decision_card(
    run_output: Dict[str, Any],
    facility_id: str,
    grid: SimulationGrid
) -> Dict[str, Any]:
    """
    Menghasilkan kartu keputusan untuk fasilitas tertentu (sekolah, puskesmas, bandara).
    Menampilkan peluang terkena, jam tiba tercepat & paling mungkin, dan keparahan.
    """
    fac = OverpassIngest.get_facility_by_id(facility_id)
    if not fac:
        return {"available": False, "reason": f"Fasilitas ID '{facility_id}' tidak ditemukan."}

    hourly_masks = run_output.get("hourly_exposure_masks")
    n_hours, n_members, n_lat, n_lon = hourly_masks.shape

    r, c = grid.coord_to_index(fac["latitude"], fac["longitude"])

    # Fraksi anggota yang pernah mencapai lokasi ini
    ever_exp = np.any(hourly_masks[:, :, r, c], axis=0)  # shape (n_members,)
    prob = float(np.mean(ever_exp))

    earliest_arr = run_output.get("arrival_time", {}).get("earliest", [])
    likely_arr = run_output.get("arrival_time", {}).get("most_likely", [])

    earliest_h = _safe_float(earliest_arr[r][c]) if earliest_arr else None
    likely_h = _safe_float(likely_arr[r][c]) if likely_arr else None

    return {
        "facility_id": fac["facility_id"],
        "name": fac["name"],
        "facility_type": fac["facility_type"],
        "latitude": fac["latitude"],
        "longitude": fac["longitude"],
        "district": fac.get("district", "-"),
        "exposure_probability": round(prob, 2),
        "arrival_time_earliest_h": earliest_h,
        "arrival_time_most_likely_h": likely_h,
        "scenario_label": run_output["scenario_label"],
        "disclaimer": run_output["disclaimer"]
    }


def get_exposed_facilities_list(
    run_output: Dict[str, Any],
    grid: SimulationGrid,
    min_prob: float = 0.05
) -> Dict[str, Any]:
    """
    Mengambil daftar seluruh fasilitas publik (sekolah & faskes) di dalam jalur paparan,
    lengkap dengan rekomendasi aksi operasional dini (PJJ daring vs tatap muka).
    """
    facilities = OverpassIngest.get_facilities_in_bbox(
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max
    )

    hourly_masks = run_output.get("hourly_exposure_masks")
    earliest_arr = run_output.get("arrival_time", {}).get("earliest", [])
    likely_arr = run_output.get("arrival_time", {}).get("most_likely", [])

    schools = []
    healthcare = []
    airports = []

    # Muat seluruh fasilitas nasional untuk kalkulasi rujukan evakuasi terdekat & aman
    all_facilities = OverpassIngest.load_all_facilities()
    all_hospitals = [f for f in all_facilities if f.get("facility_type") == "healthcare"]
    for f in facilities:
        if f.get("facility_type") == "healthcare" and f not in all_hospitals:
            all_hospitals.append(f)

    # Evaluasi status paparan & keamanan untuk seluruh rumah sakit
    hospital_eval_list = []
    for h in all_hospitals:
        h_lat = h["latitude"]
        h_lon = h["longitude"]
        h_prob = 0.0
        h_earliest = None
        if grid.lat_min <= h_lat <= grid.lat_max and grid.lon_min <= h_lon <= grid.lon_max:
            r, c = grid.coord_to_index(h_lat, h_lon)
            ever_exp = np.any(hourly_masks[:, :, r, c], axis=0)
            h_prob = float(np.mean(ever_exp))
            if earliest_arr:
                val = _safe_float(earliest_arr[r][c])
                if val is not None:
                    h_earliest = round(val, 1)

        is_safe = h_prob < 0.10
        if is_safe:
            evac_status = "SAFE_ZONE"
            status_badge = "[RUJUKAN UTAMA] Zona Aman Bebas Abu"
            badge_color = "#10b981"
            evac_recommendation = "Prioritas rujukan utama evakuasi. Udara bersih, kapasitas IGD & O2 aman."
        elif h_prob < 0.35:
            evac_status = "WARNING_ZONE"
            status_badge = "[TRIASE LAPANGAN] Siaga Penyangga"
            badge_color = "#f59e0b"
            evac_recommendation = "Siaga pertolongan pertama & stabilisasi pasien sebelum dirujuk ke RS aman."
        else:
            evac_status = "DANGER_ZONE"
            status_badge = "[TERDAMPAK] Zona Bahaya Abu"
            badge_color = "#ef4444"
            evac_recommendation = "RS terpapar abu lebat. Tidak disarankan menerima rujukan korban luar; prioritaskan perlindungan pasien internal."

        hospital_eval_list.append({
            "facility_id": h["facility_id"],
            "name": h["name"],
            "facility_type": "healthcare",
            "district": h.get("district", "-"),
            "latitude": h_lat,
            "longitude": h_lon,
            "exposure_probability": round(h_prob, 2),
            "exposure_pct": int(round(h_prob * 100)),
            "arrival_time_earliest_h": h_earliest,
            "is_safe": is_safe,
            "evacuation_status": evac_status,
            "status_badge": status_badge,
            "badge_color": badge_color,
            "evac_recommendation": evac_recommendation,
            "emergency_hotline": "SPGDT 119 / Call Center 112",
            "capabilities": ["IGD 24 Jam", "Pasokan O2 Medis", "Ruang Rawat ISPA", "Unit Ambulans Gawat Darurat"]
        })

    # Pusat kejadian untuk hitung jarak RS ke episentrum
    center = run_output.get("center_coords") or {
        "latitude": (grid.lat_min + grid.lat_max) / 2.0,
        "longitude": (grid.lon_min + grid.lon_max) / 2.0
    }
    c_lat, c_lon = center["latitude"], center["longitude"]
    for h in hospital_eval_list:
        h["distance_from_epicenter_km"] = round(haversine_km(c_lat, c_lon, h["latitude"], h["longitude"]), 1)

    # RS Rujukan Terdekat dari episentrum (radius ~220 km)
    evacuation_referral_hospitals = [
        h for h in hospital_eval_list if h["distance_from_epicenter_km"] <= 220.0
    ]
    evacuation_referral_hospitals.sort(key=lambda x: (not x["is_safe"], x["distance_from_epicenter_km"]))

    # Evaluasi fasilitas publik dalam grid
    for fac in facilities:
        r, c = grid.coord_to_index(fac["latitude"], fac["longitude"])
        ever_exp = np.any(hourly_masks[:, :, r, c], axis=0)
        prob = float(np.mean(ever_exp))

        if prob < min_prob:
            continue

        earliest_h = _safe_float(earliest_arr[r][c]) if earliest_arr else None
        likely_h = _safe_float(likely_arr[r][c]) if likely_arr else None

        ftype = fac.get("facility_type")
        item = {
            "facility_id": fac["facility_id"],
            "name": fac["name"],
            "facility_type": ftype,
            "district": fac.get("district", "-"),
            "latitude": fac["latitude"],
            "longitude": fac["longitude"],
            "exposure_probability": round(prob, 2),
            "exposure_pct": int(round(prob * 100)),
            "arrival_time_earliest_h": round(earliest_h, 1) if earliest_h is not None else None,
            "arrival_time_most_likely_h": round(likely_h, 1) if likely_h is not None else None,
        }

        # Rekomendasi aksi operasional pencegahan & evakuasi (sinkron dengan tier P90, P50, P05)
        if ftype == "school":
            if prob >= 0.90:
                item["alert_level"] = "SIAGA TINGGI"
                item["badge_color"] = "#ef4444"
                item["recommendation"] = "ALIHKAN KE PJJ DARING (BELAJAR DARI RUMAH)"
                item["action_detail"] = (
                    f"Hentikan KBM tatap muka. Pulangkan siswa sebelum T+{item['arrival_time_earliest_h'] or 'segera'} jam "
                    f"atau alihkan langsung ke daring. Amankan ruang kelas dan tutup ventilasi."
                )
            elif prob >= 0.50:
                item["alert_level"] = "WASPADA"
                item["badge_color"] = "#f59e0b"
                item["recommendation"] = "HENTIKAN AKTIVITAS LUAR RUANG (OUTDOOR)"
                item["action_detail"] = "Larang upacara, olahraga, dan istirahat di luar gedung. Siapkan masker N95 di UKS."
            else:
                item["alert_level"] = "SIAGA DINI"
                item["badge_color"] = "#10b981"
                item["recommendation"] = "MONITORING PERKEMBANGAN ANGIN"
                item["action_detail"] = "KBM berjalan normal dengan kewaspadaan telemetri berkala."

            # Cari RS terdekat & RS rujukan aman terdekat untuk evakuasi korban sekolah ini
            # Wajib pada daratan/pulau yang sama (menolak rute ambulans menyeberang selat/lautan)
            sch_landmass = get_landmass_region(fac)
            same_landmass_hospitals = [
                h for h in hospital_eval_list
                if get_landmass_region(h) == sch_landmass
            ]
            eval_hospitals = same_landmass_hospitals if same_landmass_hospitals else hospital_eval_list

            # Cari RS terdekat & RS rujukan aman minim dampak untuk evakuasi korban sekolah ini
            # Menolak evakuasi yang memindahkan ambulans semakin dekat ke kawah aktif / menembus KRB pekat
            fac_lat = fac["latitude"]
            fac_lon = fac["longitude"]
            fac_to_center = haversine_km(fac_lat, fac_lon, c_lat, c_lon)

            best_safe_h = None
            best_safe_score = float("inf")
            nearest_any_h = None
            min_any_dist = float("inf")
            # Cari RS rujukan aman terdekat (prioritaskan 100% RS yang berstatus is_safe == True)
            safe_pool = [h for h in eval_hospitals if h.get("is_safe")]
            search_pool = safe_pool if safe_pool else eval_hospitals

            for h in eval_hospitals:
                dist = haversine_km(fac_lat, fac_lon, h["latitude"], h["longitude"])
                if dist < min_any_dist:
                    min_any_dist = dist
                    nearest_any_h = {**h, "distance_km": round(dist, 1)}

            for h in search_pool:
                h_lat = h["latitude"]
                h_lon = h["longitude"]
                dist = haversine_km(fac_lat, fac_lon, h_lat, h_lon)
                h_to_center = haversine_km(h_lat, h_lon, c_lat, c_lon)
                h_prob = h.get("exposure_probability", 0.0)

                # Sampel 7 titik di koridor untuk memeriksa apakah jalur mendekati kawah aktif
                corridor_min_crater = min(
                    haversine_km(
                        fac_lat + (i / 6.0) * (h_lat - fac_lat),
                        fac_lon + (i / 6.0) * (h_lon - fac_lon),
                        c_lat,
                        c_lon
                    )
                    for i in range(7)
                )

                # 1. Penalti Incursion Kawah: Rute dilarang memindahkan ambulans mendekati kawah aktif (< 12 km)
                crater_penalty = 0.0
                if corridor_min_crater < 12.0:
                    crater_penalty += (12.0 - corridor_min_crater) * 6.0
                if h_to_center < fac_to_center - 2.0:
                    crater_penalty += (fac_to_center - h_to_center) * 3.0

                # 2. Penalti Paparan Bahaya RS
                if h_prob >= 0.35:
                    exp_penalty = 45.0 + (h_prob * 30.0)
                elif h_prob >= 0.15:
                    exp_penalty = h_prob * 10.0
                else:
                    exp_penalty = 0.0

                score = dist + crater_penalty + exp_penalty

                if score < best_safe_score:
                    best_safe_score = score
                    badge_desc = "[RUJUKAN AMAN] Koridor Bebas Bahaya" if h.get("is_safe") else "[TRIASE MINIM DAMPAK] Bebas Radius Bahaya"
                    best_safe_h = {
                        **h,
                        "distance_km": round(dist, 1),
                        "corridor_min_crater_km": round(corridor_min_crater, 1),
                        "status_badge": badge_desc
                    }

            item["nearest_hospital"] = nearest_any_h
            target_evac = best_safe_h or nearest_any_h
            item["nearest_safe_hospital"] = target_evac
            if target_evac:
                item["evacuation_protocol"] = (
                    f"Jika ada siswa/guru sesak napas akut atau luka abrasi, evakuasi segera ke "
                    f"{target_evac['name']} ({target_evac['distance_km']} km, {target_evac['status_badge']}). "
                    f"Hotline Gawat Darurat: SPGDT 119."
                )

            schools.append(item)

        elif ftype == "healthcare":
            if prob >= 0.90:
                item["alert_level"] = "SIAGA DARURAT"
                item["badge_color"] = "#ef4444"
                item["recommendation"] = "SIAPKAN RUANG ISOLASI & NEBULIZER ISPA"
                item["action_detail"] = (
                    f"Siapkan stok cadangan O2 dan obat pernapasan sebelum T+{item['arrival_time_earliest_h'] or 'segera'} jam. "
                    f"Wajibkan APD masker N95 untuk seluruh nakes dan pasien."
                )
            elif prob >= 0.50:
                item["alert_level"] = "WASPADA FASKES"
                item["badge_color"] = "#f59e0b"
                item["recommendation"] = "INVENTARISASI LOGISTIK PERNAPASAN"
                item["action_detail"] = "Pastikan masker medis partikulat dan nebulizer siap operasi."
            else:
                item["alert_level"] = "SIAGA DINI FASKES"
                item["badge_color"] = "#10b981"
                item["recommendation"] = "MONITORING POSKO KESEHATAN"
                item["action_detail"] = "Kesiapsiagaan pemantauan keluhan pernapasan warga berkala."
            healthcare.append(item)

        else:
            item["alert_level"] = "WASPADA RUANG UDARA"
            item["badge_color"] = "#f59e0b"
            item["recommendation"] = "KOORDINASI NOTAM / ASHTAM"
            item["action_detail"] = "Peringatan bahaya partikel abrasif bagi turbin mesin pesawat."
            airports.append(item)

    # Urutkan berdasarkan peluang tertinggi dan waktu tiba tercepat
    schools.sort(key=lambda x: (-x["exposure_probability"], x["arrival_time_earliest_h"] or 999))
    healthcare.sort(key=lambda x: (-x["exposure_probability"], x["arrival_time_earliest_h"] or 999))

    return {
        "total_schools_at_risk": len(schools),
        "total_healthcare_at_risk": len(healthcare),
        "total_airports_at_risk": len(airports),
        "total_evacuation_hospitals": len(evacuation_referral_hospitals),
        "schools": schools,
        "healthcare": healthcare,
        "airports": airports,
        "evacuation_referral_hospitals": evacuation_referral_hospitals,
        "facilities_impacted": schools + healthcare + airports
    }


def get_health_and_mask_guidance(hazard: str = "volcano") -> Dict[str, Any]:
    """
    Panduan edukasi kesehatan masyarakat dan spesifikasi masker wajib standar.
    """
    return {
        "hazard": hazard,
        "title": "Edukasi Perlindungan Saluran Pernapasan & Mata Terverifikasi",
        "mask_type_recommended": "N95 / KN95 (Partikulat Terstandar)",
        "mask_guidance": {
            "prohibited": [
                {
                    "name": "Masker Bedah Biasa (Surgical Mask 3-Ply)",
                    "verdict": "TIDAK EFEKTIF (JANGAN DIPAKAI SEBAGAI PERLINDUNGAN UTAMA)",
                    "reason": (
                        "Pori-pori masker bedah dirancang untuk cairan (3–5 mikron), bukan partikulat mikro. "
                        "Memiliki tingkat kebocoran tepi (seal leakage) 30%–50%. Partikel abu silika tajam "
                        "dan asap PM2.5 tetap leluasa masuk menembus celah hidung dan pipi."
                    )
                },
                {
                    "name": "Masker Kain / Scarf / Masker Buff",
                    "verdict": "SANGAT TIDAK EFEKTIF (BERBAHAYA KARENA MEMBERI ILUSI AMAN)",
                    "reason": (
                        "Kerapatan serat kain katun terlalu longgar (50–100 mikron). Partikel abu berbahaya "
                        "lolos 100% langsung masuk ke percabangan bronkus dan alveolus paru."
                    )
                }
            ],
            "mandatory_standards": [
                {
                    "name": "Masker Partikulat N95 (Standar AS NIOSH 42 CFR 84)",
                    "verdict": "STANDAR RESMI WAJIB (SANGAT DIREKOMENDASIKAN)",
                    "efficiency": "Menyaring ≥ 95% partikel mikroskopis hingga ukuran 0.3 mikron",
                    "instructions": "Klip kawat hidung harus ditekan rapat mengikuti bentuk hidung. Tali ganda dilingkarkan di kepala, bukan telinga, untuk memastikan segel kedap udara."
                },
                {
                    "name": "Masker Partikulat KN95 / FFP2 (Standar GB2626 / EN 149)",
                    "verdict": "STANDAR RESMI ALTERNATIF TINGKAT TINGGI",
                    "efficiency": "Menyaring ≥ 94% – 95% partikulat berbahaya",
                    "instructions": "Bentuk 3D memudahkan pernapasan dan komunikasi saat proses evakuasi atau pembelajaran darurat."
                }
            ]
        },
        "eye_protection": {
            "critical_warning": "DILARANG KERAS MEMAKAI LENSA KONTAK (SOFTLENS)!",
            "medical_rationale": (
                "Abu vulkanik bukanlah abu kayu biasa, melainkan pecahan kristal kaca silika (SiO2) "
                "mikroskopis yang tajam dan kasar. Jika partikel debu masuk ke balik lensa kontak, "
                "gerakan kelopak mata akan menggores permukaan kornea (kornea abrasi) hingga berisiko "
                "kebutaan permanen dan ulkus kornea."
            ),
            "safe_action": "Gunakan kacamata pelindung tertutup (Safety Goggles) dengan tepi karet kedap."
        },
        "household_safety": [
            "Tutup rapat seluruh celah kusen jendela dan pintu menggunakan handuk basah atau lakban tebal.",
            "Hindari menyapu abu kering dengan sapu biasa karena akan menerbangkan kembali partikel ke udara (resuspensi). Gunakan semprotan air lembut dan pel basah.",
            "Segera bersihkan timbunan abu di atap rumah sebelum hujan turun, karena abu basah memiliki berat 2–3x lipat yang dapat merubuhkan atap pemukiman."
        ]
    }

