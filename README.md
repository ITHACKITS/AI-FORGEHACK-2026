# 🌪️ UdaraSiaga AI & Atmospheric Physics Engine — ITHACKITS FORGEHACK 2026

**Repository Resmi: Numerical Physics Dispersion Models & Generative AI Disaster Synthesizer**

Repository ini menyediakan modul komputasi ilmiah dan kecerdasan buatan untuk pemodelan dispersi partikulat atmosfer (abu vulkanik & asap kebakaran hutan dan lahan/karhutla) serta sintesis darurat berbasis Large Language Model (Google Gemini 2.5 Flash & Sistem Pakar Deterministik BNPB/Kemenkes).

---

## 🔬 Fondasi Saintifik & Model Fisika

1. **Pemodelan Erupsi Gunung Api (Mastin et al., 2009)**
   - Perumusan empiris laju erupsi massa (*Mass Eruption Rate* / MER):
     $$H = 2.0 \cdot \dot{V}^{0.241}$$
   - Profil distribusi konsentrasi vertikal Suzuki.
   - Spektrum ukuran butir (*Grain Size Distribution*) dari partikel kasar hingga debu respirabel ($PM_{2.5}$ & $PM_{10}$).

2. **Pemodelan Karhutla (Byram, 1959)**
   - *Fireline Intensity* & laju pelepasan kalor (*Heat Release Rate* / HRR).
   - Injeksi asap termal dan emisi $PM_{2.5}$ berdasarkan FRP (*Fire Radiative Power*) satelit NASA FIRMS.

3. **Dinamika Partikel & Transpor Atmosferik**
   - Transpor Eulerian-Lagrangian dengan adveksi vektor angin (integrasi numerik Runge-Kutta Orde-2 / RK2).
   - Difusi turbulen atmosferik vertikal & horizontal.
   - *Stokes Settling Velocity* dilengkapi faktor koreksi slip Cunningham (*Cunningham Slip Correction Factor*) untuk partikel mikron sub-mikron.
   - *Wet Scavenging*: Pencucian partikel oleh presipitasi hujan ($\Lambda$ scavenging coefficient).

4. **Masking Spasial & Integritas Daratan (LandMask Engine)**
   - Indeks spasial poligon daratan Indonesia vektor Shapely 2.1 (`indonesia_coastline.geojson`).
   - Mencegah artefak populasi atau fasilitas fiktif di perairan/laut.

5. **AI Emergency Briefing & LLM Advisory Synthesizer**
   - **Google Gemini 2.5 Flash**: Menghasilkan draf siaran pers resmi PUSDALOPS BPBD, broadcast darurat WhatsApp warga, dan protokol medis APD respirasi.
   - **Deterministic Fallback Synthesizer**: Menjamin ketersediaan rekomendasi terstandarisasi Permenkes & BNPB bahkan saat jaringan offline.

---

## 🚀 Panduan Memulai Cepat (Quickstart)

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Environment
```bash
cp .env.example .env
# Masukkan GEMINI_API_KEY Anda di file .env
```

### 3. Menjalankan AI Microservice
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
Aplikasi akan aktif di **http://localhost:8001**.
Dokumentasi interaktif OpenAPI / Swagger dapat diakses di **http://localhost:8001/docs**.

### 4. Menjalankan Suite Pengujian Fisika
```bash
pytest tests -v
```
Seluruh 9 pengujian numerik dan sintesis darurat terverifikasi 100% lulus.

---

## 📁 Struktur Direktori

```text
AI-FORGEHACK-2026/
├── app/
│   ├── engine/             # Komputasi numerik dispersi atmosfer
│   │   ├── simulator.py    # Orkestrator simulasi Lagrangian-Eulerian
│   │   ├── volcano.py      # Mastin 2009 MER & distribusi partikel erupsi
│   │   ├── fire.py         # Byram 1959 Fireline intensity & emisi karhutla
│   │   ├── particles.py    # RK2 adveksi partikel & Stokes settling
│   │   ├── grid.py         # Grid konsentrasi & IDW interpolation
│   │   ├── landmask.py     # Shapely 2.1 vectorized spatial index daratan
│   │   ├── scavenging.py   # Rain washout kinetics
│   │   └── aggregation.py  # Agregasi probabilitas P90/P50/P05
│   ├── impact/             # Analisis dampak & sintesis AI
│   │   ├── gemini_advisory.py # Google Gemini 2.5 Flash LLM synthesizer
│   │   ├── advisory.py     # Deterministic expert synthesizer BNPB/Kemenkes
│   │   ├── exposure.py     # Matriks kerentanan & populasi terpapar
│   │   ├── volcano_features.py # Dampak sawah & sumber air daratan
│   │   ├── fire_features.py# Dampak lahan gambut & kawasan konservasi
│   │   ├── comparative.py  # Analisis delta skenario multi-bahaya
│   │   └── validation.py   # Validasi konsistensi saintifik
│   ├── core/               # Konfigurasi, RNG, offline mode guards
│   ├── schemas/            # Pydantic schemas kontrak request & response
│   ├── ingest/             # Klien meteorologi Open-Meteo & FIRMS
│   └── main.py             # FastAPI entry point microservice (Port 8001)
├── data/
│   ├── layers/             # Data geospasial garis pantai & gunung api
│   └── events/             # Presets observasi lapangan (Anak Krakatau, Kalimantan)
├── tests/
│   ├── test_physics.py     # Uji kekekalan massa, adveksi RK2, Mastin MER
│   └── test_advisory.py    # Uji sintesis rekomendasi darurat AI
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🔗 Keterhubungan Komponen ForgeHack 2026

- **Frontend Command Center:** [FE-FORGEHACK-2026](https://github.com/ITHACKITS/FE-FORGEHACK-2026)
- **Backend API Gateway:** [BE-FORGEHACK-2026](https://github.com/ITHACKITS/BE-FORGEHACK-2026)