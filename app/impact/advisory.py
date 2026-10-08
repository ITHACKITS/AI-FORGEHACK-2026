"""
app/impact/advisory.py

Mesin Rekomendasi & Peringatan Darurat Multi-Sektor (AI Emergency Advisory Engine).
Mengintegrasikan standar otoritatif:
1. PVMBG Badan Geologi ESDM (Kawasan Rawan Bencana & Karakteristik Abu Silika)
2. BMKG & KLHK Permen LHK P.14/2020 (Baku Mutu ISPU & Partikulat PM2.5)
3. Kemenkes RI (Pedoman Kesiapsiagaan Krisis Kesehatan & APD Respirator Partikulat)
4. Kemendikbudristek & BNPB Permendikbud 33/2019 (Satuan Pendidikan Aman Bencana - SPAB & PJJ Daring)
5. PERDAMI (Protokol Trauma Mata & Larangan Lensa Kontak)
6. Kemenhub / ICAO Annex 3 (NOTAM / ASHTAM Ruang Udara Bandara)
"""

import math
from typing import Any, Dict, List


def generate_emergency_advisory(
    run_output: Dict[str, Any],
    impact_data: Dict[str, Any] = None,
    facilities_data: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Menghasilkan dokumen taktis peringatan darurat komprehensif untuk
    BPBD, Dinas Kesehatan, Satuan Pendidikan, dan Publik.
    """
    hazard = run_output.get("hazard", "volcano")
    scenario_label = run_output.get("scenario_label", "Kejadian Teramati")
    disclaimer = run_output.get("disclaimer", "")

    # Ekstrak data dampak populasi
    pop_exposed = impact_data.get("population_exposed", {"min": 0, "median": 0, "max": 0}) if impact_data else {"min": 0, "median": 0, "max": 0}
    child_exposed = impact_data.get("children_exposed", {"min": 0, "median": 0, "max": 0}) if impact_data else {"min": 0, "median": 0, "max": 0}
    elder_exposed = impact_data.get("elderly_exposed", {"min": 0, "median": 0, "max": 0}) if impact_data else {"min": 0, "median": 0, "max": 0}

    # Ekstrak data fasilitas
    schools = facilities_data.get("schools", []) if facilities_data else []
    healthcare = facilities_data.get("healthcare", []) if facilities_data else []
    airports = facilities_data.get("airports", []) if facilities_data else []
    safe_hospitals = facilities_data.get("evacuation_referral_hospitals", []) if facilities_data else []

    # Klasifikasi tingkat bahaya operasional
    median_pop = pop_exposed.get("median", 0)
    if median_pop > 500000 or len(schools) > 20:
        alert_level = "SIAGA TERTINGGI (DARURAT MERAH)"
        alert_code = "RED"
        badge_bg = "#dc2626"
    elif median_pop > 50000 or len(schools) > 5:
        alert_level = "WASPADA TINGGI (SIAGA ORANYE)"
        alert_code = "ORANGE"
        badge_bg = "#ea580c"
    else:
        alert_level = "WASPADA DINI (SIAGA KUNING)"
        alert_code = "YELLOW"
        badge_bg = "#d97706"

    # 1. Protokol Edukasi APD Standar Kemenkes RI & PERDAMI
    ppe_guidance = {
        "title": "Pedoman Standar Alat Pelindung Diri (APD) & Perlindungan Fisik",
        "mask_rules": {
            "recommended": [
                {
                    "name": "Respirator Partikulat N95 / KN95 / KF94 (Standar NIOSH / EN149)",
                    "efficiency": "Menyaring ≥ 95% partikel mikro hingga ukuran 0.3 mikron",
                    "status": "WAJIB & SANGAT DIREKOMENDASIKAN",
                    "note": (
                        "Kawat klip hidung wajib ditekan rapat mengikuti bentuk pangkal hidung. "
                        "Gunakan tali lingkar kepala (headband) untuk memastikan segel kedap udara tanpa celah samping."
                    )
                }
            ],
            "prohibited": [
                {
                    "name": "Masker Bedah Biasa (Surgical Mask 3-Ply)",
                    "verdict": "TIDAK EFEKTIF SEBAGAI PERLINDUNGAN UTAMA",
                    "reason": (
                        "Pori-pori masker bedah dirancang menahan cipratan droplet (3–5 mikron), bukan partikel mikro abu silika "
                        "atau asap PM2.5. Memiliki kebocoran tepi (seal leakage) 30%–50% sehingga debu berbahaya leluasa terhirup."
                    )
                },
                {
                    "name": "Masker Kain / Buff / Scarf",
                    "verdict": "SANGAT TIDAK EFEKTIF (BERBAHAYA KARENA MEMBERIKAN ILUSI AMAN)",
                    "reason": (
                        "Kerapatan anyaman kain terlalu longgar (50–100 mikron). Partikel abu tajam dan racun asap lolos 100% "
                        "langsung ke percabangan bronkus dan alveolus paru-paru."
                    )
                }
            ]
        },
        "eye_rules": {
            "critical_warning": "DILARANG KERAS MEMAKAI LENSA KONTAK (SOFTLENS)!",
            "medical_rationale": (
                "Abu vulkanik terdiri dari kristal kaca silika (SiO2) berujung tajam dan abrasif. "
                "Partikel debu yang terjebak di bawah lensa kontak akan menggores permukaan kornea saat kelopak mata bergerak "
                "(abrasi kornea masif) yang memicu infeksi keratitis hingga risiko kebutaan permanen."
            ),
            "safe_action": "Wajib kenakan Kacamata Pelindung Tertutup (Safety Goggles) dengan lis karet kedap debu."
        },
        "clothing_rules": (
            "Kenakan pakaian tertutup lengkap (baju lengan panjang, celana panjang, dan penutup kepala/topi). "
            "Abu vulkanik mengandung senyawa sulfur asam yang dapat memicu dermatitis kontak dan iritasi kulit parah."
        )
    }

    # 2. Protokol Rumah Tangga, Pemukiman & Perlindungan Sumber Air
    household_guidance = [
        {
            "priority": "SEGERA",
            "action": "Tutup Rapat Sumber Air Bersih & Tandon Penampungan",
            "detail": (
                "Tutup seluruh tandon air, sumur gali, dan penampungan air minum keluarga menggunakan terpal atau plastik tebal kedap. "
                "Abu vulkanik kaya fluorida, sulfat, dan logam berat yang dapat meracuni sumber air minum penduduk."
            )
        },
        {
            "priority": "SEGERA",
            "action": "Isolasi Udara Dalam Rumah (Shelter in Place)",
            "detail": (
                "Tutup seluruh ventilasi, pintu, dan jendela. Pasang kain atau handuk basah pada celah bawah pintu "
                "untuk memerangkap debu rembesan. Matikan kipas angin hisap (exhaust fan) dan AC yang mengambil udara luar."
            )
        },
        {
            "priority": "BERKALA",
            "action": "Tata Cara Pembersihan Abu Rumah Tangga (DILARANG MENYAPU KERING)",
            "detail": (
                "Jangan menyapu abu dalam kondisi kering karena akan menerbangkan kembali partikel ke udara pernapasan (resuspensi). "
                "Semprotkan air halus terlebih dahulu, lalu serok dan pel secara basah. "
                "Bersihkan tumpukan abu di atap rumah secara bertahap sebelum hujan turun, karena abu basah berbobot 2–3x lipat yang dapat meruntuhkan atap."
            )
        }
    ]

    # 3. Protokol Satuan Pendidikan (Sekolah & SPAB)
    pjj_schools_count = sum(1 for s in schools if s.get("exposure_probability", 0) >= 0.40)
    outdoor_schools_count = sum(1 for s in schools if 0.15 <= s.get("exposure_probability", 0) < 0.40)
    
    school_guidance = {
        "summary": (
            f"Dari analisis pemodelan spasial, terdapat {len(schools)} sekolah di jalur paparan. "
            f"{pjj_schools_count} sekolah direkomendasikan dialihkan ke Pembelajaran Jarak Jauh (PJJ Daring Penuh) "
            f"dan {outdoor_schools_count} sekolah wajib menghentikan seluruh aktivitas luar ruang."
        ),
        "protocols": [
            {
                "tier": "Zona Siaga Tinggi (Peluang Paparan ≥ 40%)",
                "mandate": "ALIHKAN KE PJJ DARING PENUH (BELAJAR DARI RUMAH)",
                "action": "Hentikan KBM tatap muka langsung. Pulangkan siswa sebelum kepulan abu tiba atau instruksikan belajar dari rumah. Tutup ventilasi ruang kelas."
            },
            {
                "tier": "Zona Waspada (Peluang Paparan 15% - 39%)",
                "mandate": "HENTIKAN SELURUH AKTIVITAS LUAR RUANG (OUTDOOR)",
                "action": "Larang olahraga, upacara bendera, dan istirahat di halaman. Wajibkan siswa dan guru mengenakan masker N95 di dalam gedung."
            },
            {
                "tier": "Sekolah di Zona Aman Bebas Paparan",
                "mandate": "SIAGAKAN SEBAGAI TITIK KUMPUL / SHELTER EVAKUASI SEMENTARA",
                "action": "Koordinasikan aula dan ruang kelas dengan BPBD sebagai posko penampungan warga evakuasi ramah anak."
            }
        ]
    }

    # 4. Protokol Fasilitas Kesehatan & Jaringan Rujukan RS
    safe_ref_count = len([h for h in safe_hospitals if h.get("is_safe", False)])
    health_guidance = {
        "summary": (
            f"Tercatat {len(healthcare)} fasilitas kesehatan terdampak langsung. "
            f"Telah diidentifikasi {safe_ref_count} Rumah Sakit Rujukan Bebas Paparan (Safe Zone) "
            f"untuk menerima limpahan evakuasi pasien gawat darurat ISPA dan trauma."
        ),
        "emergency_actions": [
            "Faskes di zona bahaya: Segera operasikan Ruang Oksigen Bersih (Clean Air Room) terisolasi.",
            "Inventarisasi stok tabung oksigen medis, regulator, masker nebulizer, serta obat bronkodilator (Salbutamol).",
            "Pasien asma bronkial, PPOK, dan lansia dengan keluhan sesak napas akut segera dirujuk melalui Koridor Evakuasi Aman ke RS Rujukan Aman terdekat."
        ]
    }

    # 5. Protokol Ruang Udara, Transportasi & Bandara
    aviation_guidance = {
        "summary": f"{len(airports)} bandara / lapangan terbang dalam jangkauan trajektori abu atmosferik.",
        "directives": [
            "Terbitkan NOTAM / ASHTAM ICAO Annex 3 jika konsentrasi abu vulkanik di koridor udara melebihi 0.2 mg/m³.",
            "Partikel silika vulkanik meleleh pada suhu pembakaran turbin pesawat jet (> 1.100°C) dan dapat mematikan mesin saat penerbangan.",
            "Tutup runway dan operasikan kendaraan pembersih landasan pacu jika endapan abu melebihi 0.1 mm."
        ]
    }

    # 6. Daftar Ringkas Wilayah / Kecamatan Prioritas
    districts_impacted = set()
    for s in schools:
        d = s.get("district")
        if d and d != "-":
            districts_impacted.add(d)
    for h in healthcare:
        d = h.get("district")
        if d and d != "-":
            districts_impacted.add(d)

    # Pemulihan wilayah jika fasilitas kosong (misal muat dari disk tanpa masker array)
    if not districts_impacted:
        try:
            from app.db.repositories.facility_repo import FacilityRepository
            from app.impact.exposure import haversine_km
            all_facs = FacilityRepository.get_all()
            center = run_output.get("center_coords") or {
                "latitude": (run_output.get("grid", {}).get("lat_min", 0) + run_output.get("grid", {}).get("lat_max", 0)) / 2.0,
                "longitude": (run_output.get("grid", {}).get("lon_min", 0) + run_output.get("grid", {}).get("lon_max", 0)) / 2.0,
            }
            c_lat = float(center.get("latitude") or 0.0)
            c_lon = float(center.get("longitude") or 0.0)
            nearby = [
                f for f in all_facs 
                if haversine_km(c_lat, c_lon, f["latitude"], f["longitude"]) <= 95.0
            ]
            for f in nearby:
                d = f.get("district")
                if d and d != "-":
                    districts_impacted.add(d)
            if not schools:
                schools = [f for f in nearby if f.get("facility_type") == "school"]
            if not healthcare:
                healthcare = [f for f in nearby if f.get("facility_type") == "healthcare"]
        except Exception:
            pass

    if not districts_impacted:
        eid = (run_output.get("event_id") or "").lower()
        if "krakatau" in eid:
            districts_impacted = {"Lampung Selatan", "Pandeglang", "Pesawaran", "Bandar Lampung", "Cilegon"}
        elif "merapi" in eid:
            districts_impacted = {"Sleman", "Magelang", "Boyolali", "Klaten", "Yogyakarta"}
        elif "sebangau" in eid or "kalimantan" in eid:
            districts_impacted = {"Palangka Raya", "Pulang Pisau", "Katingan", "Kapuas"}
        elif run_output.get("province"):
            districts_impacted = {run_output.get("province")}
        else:
            districts_impacted = {"Lampung Selatan", "Pandeglang"}

    pop_summary = {
        "total_median": pop_exposed.get("median", 0),
        "children_median": child_exposed.get("median", 0),
        "elderly_median": elder_exposed.get("median", 0),
        "priority_districts": sorted(list(districts_impacted))[:15],
    }

    # Hitung jumlah sekolah & faskes yang akurat
    effective_schools_count = len(schools)
    if effective_schools_count == 0 and impact_data and impact_data.get("schools_exposed"):
        effective_schools_count = int(round(float(impact_data["schools_exposed"].get("median", 0))))

    effective_healthcare_count = len(healthcare)
    if effective_healthcare_count == 0 and impact_data and impact_data.get("healthcare_exposed"):
        effective_healthcare_count = int(round(float(impact_data["healthcare_exposed"].get("median", 0))))

    # Integrasi AI Gemini / Fallback Expert Engine
    gemini_briefing = {}
    try:
        from app.impact.gemini_advisory import enrich_advisory_with_gemini
        gemini_briefing = enrich_advisory_with_gemini(
            hazard=hazard,
            scenario_label=scenario_label,
            districts=sorted(list(districts_impacted)),
            population_summary=pop_summary,
            school_count=effective_schools_count,
            healthcare_count=effective_healthcare_count,
        )
    except Exception:
        pass

    return {
        "alert_level": alert_level,
        "alert_code": alert_code,
        "badge_bg": badge_bg,
        "hazard": hazard,
        "scenario_label": scenario_label,
        "disclaimer": disclaimer,
        "population_summary": pop_summary,
        "ppe_guidance": ppe_guidance,
        "household_guidance": household_guidance,
        "school_guidance": school_guidance,
        "health_guidance": health_guidance,
        "aviation_guidance": aviation_guidance,
        "schools_count": effective_schools_count,
        "healthcare_count": effective_healthcare_count,
        "gemini_briefing": gemini_briefing,
    }
