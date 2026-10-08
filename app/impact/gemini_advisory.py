"""
app/impact/gemini_advisory.py

Modul Pengayaan Narasi Bencana Menggunakan Google Gemini API (Hybrid LLM Enhancer).
Menghasilkan narasi resmi, berwibawa, dan berstandar otoritatif Pusdalops BNPB & Kemenkes RI:
1. Siaran Pers Resmi Siaga Bencana BPBD (Pers Release Media)
2. Pesan Siaran Darurat Warga (Format Buletin Resmi Tanpa Emoji Kasual)
3. Rekomendasi Mitigasi Taktis & Perlindungan Organ

Dilengkapi mekanisme fallback deterministik (Zero-Failure Architecture)
berstandar Pusdalops BNPB jika jaringan offline atau kuota API terbatas.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


def get_gemini_api_key() -> str:
    """Mengambil kunci API Gemini dari environment atau file .env."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        from app.core.config import BACKEND_DIR
        env_file = BACKEND_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key


def enrich_advisory_with_gemini(
    hazard: str,
    scenario_label: str,
    districts: list,
    population_summary: Dict[str, Any],
    school_count: int,
    healthcare_count: int,
) -> Dict[str, Any]:
    """
    Memanggil Google Gemini API untuk menghasilkan narasi siaran pers dan broadcast warga.
    Jika offline/gagal, mengembalikan narasi terstruktur resmi berstandar Pusdalops BNPB secara deterministik.
    """
    api_key = get_gemini_api_key()
    total_pop = population_summary.get("total_median", 0)
    children = population_summary.get("children_median", 0)
    elderly = population_summary.get("elderly_median", 0)
    districts_str = ", ".join(districts[:6]) if districts else "Wilayah Lingkar Bahaya Episentrum"

    hazard_label = "Erupsi Gunung Api (Hujan Abu Silika)" if hazard == "volcano" else "Kebakaran Hutan & Lahan (Asap PM2.5)"

    prompt_text = f"""
Anda adalah Kepala Pusat Data, Informasi, dan Komunikasi Kebencanaan Badan Nasional Penanggulangan Bencana (BNPB) bekerja sama dengan Kementerian Kesehatan RI.
Tugas: Buat dokumen peringatan taktis resmi dalam Bahasa Indonesia baku untuk media massa dan masyarakat terdampak.

DATA TELEMETRI DAN SITUASI AKTUAL:
- Kejadian Bahaya: {hazard_label}
- Skenario Operasional: {scenario_label}
- Wilayah Prioritas Pengawasan: {districts_str}
- Estimasi Penduduk Terpapar: {total_pop:,.0f} jiwa (Kelompok Rentan: {children:,.0f} anak-anak <15 th, {elderly:,.0f} lansia >=65 th)
- Infrastruktur Kritis di Koridor Risiko: {school_count} sekolah, {healthcare_count} fasilitas kesehatan

PEDOMAN FORMAT DAN GAYA BAHASA (WAJIB DIPATUHI):
1. GAYA BAHASA: Sangat resmi, profesional, tegas, berwibawa, dan berbasis data ilmiah kebencanaan (gaya bahasa resmi Pusdalops BNPB).
2. DILARANG KERAS MENGGUNAKAN EMOJI APAPUN. DILARANG menggunakan bahasa santai atau kasual.
3. KONTEN SIARAN PERS (press_release_body):
   - Paragraf 1: Pengumuman situasi kedaruratan resmi, nama kejadian dan lokasinya, arah pergerakan trajektori atmosferik, dan rincian kuantitatif warga rentan terpapar.
   - Paragraf 2: Instruksi taktis multi-sektor: pengalihan KBM sekolah ke PJJ Daring (mandat SPAB Kemendikbud-BNPB), kesiapsiagaan ruang isolasi udara bersih di puskesmas/RSUD, serta pengamanan sumber air minum.
4. KONTEN BROADCAST WARGA (whatsapp_broadcast):
   - Format surat edaran / buletin telegram darurat resmi berstruktur (Nomor, Perihal, Wilayah, Poin Instruksi 1-5, Kontak Darurat).
   - Tanpa emoji sama sekali. Gunakan penomoran rapi (1., 2., 3.).
   - Wajib cantumkan larangan keras pemakaian lensa kontak (softlens) karena mikro-kristal silika tajam (SiO2) berisiko abrasi kornea dan kebutaan permanen.
   - Wajib cantumkan spesifikasi respirator N95/KN95/KF94 (larangan masker kain dan masker bedah karena kebocoran samping 30-50%).
   - Wajib cantumkan penutupan tandon air minum/sumur gali dari pencemaran fluorida dan asam sulfur.
   - Cantumkan nomor kontak resmi: Pusdalops BPBD (Call Center 112) dan Kegawatdaruratan Medis (SPGDT 119).

Kembalikan HANYA format JSON valid tanpa tanda pembungkus markdown (```json):
{{
  "press_release_title": "SIARAN PERS PUSDALOPS BPBD: PERINGATAN DINI KESIAPSIAGAAN ANCAMAN ...",
  "press_release_body": "Teks resmi siaran pers 2 paragraf padat instruksi...",
  "whatsapp_broadcast": "Teks resmi buletin darurat warga bernomor 1 s.d. 5 tanpa emoji...",
  "key_callout": "Pernyataan kunci kedaruratan untuk pemangku kebijakan daerah"
}}
"""

    is_offline = os.getenv("UDARASIAGA_OFFLINE") == "1"
    if api_key and not is_offline:
        models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                }
            }
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload).encode("utf-8")
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    if resp.status == 200:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if raw_text.startswith("```"):
                            raw_text = raw_text.split("\n", 1)[1]
                            if raw_text.endswith("```"):
                                raw_text = raw_text.rsplit("\n", 1)[0]
                        parsed = json.loads(raw_text)
                        return {
                            "engine": f"Google Gemini AI ({model})",
                            "status": "ai_generated",
                            "press_release_title": parsed.get("press_release_title", "SIARAN PERS RESMI TANGGAP DARURAT PUSDALOPS BPBD"),
                            "press_release_body": parsed.get("press_release_body", "").replace("🚨", "").replace("😷", "").replace("🚫", ""),
                            "whatsapp_broadcast": parsed.get("whatsapp_broadcast", "").replace("🚨", "").replace("😷", "").replace("🚫", ""),
                            "key_callout": parsed.get("key_callout", "Gunakan masker respirator N95, dilarang memakai lensa kontak, dan amankan sumber air minum."),
                        }
            except Exception:
                continue

    # Fallback Deterministik Cerdas (Standar Resmi Pusdalops BNPB / Kemenkes RI)
    if school_count > 0:
        school_action_text = (
            f"Menindaklanjuti potensi bahaya tersebut, sebanyak {school_count} satuan pendidikan di wilayah koridor terdampak "
            f"diinstruksikan segera mengalihkan kegiatan belajar mengajar ke Pembelajaran Jarak Jauh (PJJ Daring) sesuai kerangka SPAB Kemendikbud-BNPB. "
        )
        wa_school_point = f"5. Perlindungan Siswa Sekolah: Sebanyak {school_count} satuan pendidikan di zona terdampak dialihkan ke Pembelajaran Jarak Jauh (PJJ Daring). Siswa diimbau tetap berada di dalam ruangan tertutup."
    else:
        school_action_text = (
            "Seluruh satuan pendidikan di wilayah koridor penyangga diimbau meningkatkan kewaspadaan, membatasi kegiatan luar ruang, "
            "dan bersiap mengalihkan kegiatan ke PJJ Daring jika terjadi eskalasi sebaran partikel. "
        )
        wa_school_point = "5. Kesiapsiagaan Satuan Pendidikan: Sekolah di wilayah penyangga diimbau meniadakan aktivitas luar ruang (olahraga/upacara) dan memantau perkembangan arah angin."

    if healthcare_count > 0:
        health_action_text = (
            f"Fasilitas pelayanan kesehatan ({healthcare_count} faskes) disiagakan dengan Ruang Udara Bersih dan pasokan oksigen medis "
            f"guna mengantisipasi lonjakan kasus sindrom pernapasan akut."
        )
    else:
        health_action_text = (
            "Fasilitas pelayanan kesehatan di wilayah terdekat disiagakan dengan Ruang Udara Bersih dan pasokan oksigen medis "
            "untuk mengantisipasi potensi lonjakan keluhan pernapasan warga kelompok rentan."
        )

    default_pr_title = f"SIARAN PERS PUSDALOPS BPBD: PERINGATAN DINI KESIAPSIAGAAN ANCAMAN {hazard_label.upper()} ({scenario_label})"
    default_pr_body = (
        f"Pusat Pengendalian Operasi Badan Penanggulangan Bencana Daerah (Pusdalops BPBD) bersama Dinas Kesehatan "
        f"menerbitkan peringatan kesiapsiagaan dini menyusul dinamika aktivitas {hazard_label} pada skenario {scenario_label}. "
        f"Berdasarkan hasil pemodelan dispersi partikulat atmosferik, sebaran partikel terdeteksi mengarah ke koridor "
        f"{districts_str} dengan estimasi populasi berisiko terpapar mencapai {total_pop:,.0f} jiwa, yang mencakup {children:,.0f} anak-anak "
        f"dan {elderly:,.0f} lansia berisiko tinggi.\n\n"
        f"{school_action_text}{health_action_text}"
    )
    default_wa = (
        f"PEMBERITAHUAN RESMI PUSDALOPS BPBD & DINAS KESEHATAN\n"
        f"Nomor: 360/PUSDALOPS/DARURAT/{scenario_label.replace(' ', '_').upper()}\n"
        f"Perihal: Arahan Kesiapsiagaan Dini Paparan {hazard_label}\n\n"
        f"Wilayah Prioritas Pengawasan: {districts_str}\n"
        f"Estimasi Warga Terpapar: {total_pop:,.0f} jiwa (Mencakup {children:,.0f} anak-anak dan {elderly:,.0f} lansia)\n\n"
        f"Instruksi Perlindungan Diri dan Lingkungan:\n"
        f"1. Standar Perlindungan Pernapasan: Wajib menggunakan respirator partikulat N95/KN95/KF94 dengan klip hidung tertutup rapat. Masker kain dan masker bedah dilarang sebagai pelindung utama karena memiliki kebocoran samping hingga 50%.\n"
        f"2. Perlindungan Mata: Dilarang keras memakai lensa kontak (softlens). Debu abu silika (SiO2) bersifat tajam dan abrasif yang dapat merobek kornea mata (abrasi kornea) dan memicu kebutaan permanen. Gunakan kacamata pelindung tertutup (Safety Goggles).\n"
        f"3. Sanitasi Sumber Air Minum: Tutup rapat seluruh tandon air minum, penampungan, dan sumur gali keluarga menggunakan terpal kedap guna menghindari kontaminasi senyawa asam sulfur dan racun fluorida.\n"
        f"4. Isolasi Ruang Tinggal: Tutup ventilasi rumah dan pasang kain basah pada celah pintu/jendela. Jangan menyapu abu dalam kondisi kering karena akan memicu resuspensi partikel ke udara pernapasan; gunakan semprotan air halus dan pel basah.\n"
        f"{wa_school_point}\n\n"
        f"Pusat Bantuan & Komando Darurat:\n"
        f"- Layanan Gawat Darurat Medis: SPGDT 119\n"
        f"- Posko Terpadu Pusdalops BPBD: Call Center 112"
    )

    return {
        "engine": "UdaraSiaga Deterministic Expert AI Synthesizer (Standard BNPB/Kemenkes)",
        "status": "expert_fallback",
        "press_release_title": default_pr_title,
        "press_release_body": default_pr_body,
        "whatsapp_broadcast": default_wa,
        "key_callout": "Gunakan respirator N95, dilarang memakai lensa kontak, dan tutup sumber air keluarga.",
    }
