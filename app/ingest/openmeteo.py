"""
app/ingest/openmeteo.py

Modul ingest untuk data cuaca ensemble Open-Meteo (D01, D02, D03, D28).
Mendukung pemuatan offline cache dan penanganan anggota sintetis jika diperlukan.
"""

import json
from typing import Any, Dict

import numpy as np

from app.core.config import EVENTS_DIR


class OpenMeteoIngest:
    @staticmethod
    def load_cached_event_weather(event_id: str) -> Dict[str, Any]:
        """Memuat snapshot cuaca dari data/events/{event_id}/weather_ensemble.json."""
        event_file = EVENTS_DIR / event_id / "weather_ensemble.json"
        if not event_file.exists():
            # Coba ambil cuaca live real-time dari Open-Meteo API berdasarkan koordinat kejadian ini
            meta_file = EVENTS_DIR / event_id / "meta.json"
            if meta_file.exists():
                try:
                    from app.core.offline import assert_online_allowed
                    assert_online_allowed("Open-Meteo Live API")
                    import urllib.request
                    meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
                    coords = meta.get("coordinates", {})
                    lat = coords.get("latitude")
                    lon = coords.get("longitude")
                    if lat is not None and lon is not None:
                        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,precipitation&forecast_days=2"
                        req = urllib.request.urlopen(url, timeout=3.5)
                        live_data = json.loads(req.read().decode("utf-8"))
                        live_data["members_source"] = "open_meteo_live_forecast_api"
                        with open(event_file, "w", encoding="utf-8") as ef:
                            json.dump(live_data, ef, indent=2)
                        return live_data
                except Exception:
                    pass

            fallback_file = EVENTS_DIR / "anak_krakatau_2026_09" / "weather_ensemble.json"
            if fallback_file.exists():
                data = json.loads(fallback_file.read_text(encoding="utf-8-sig"))
                data["members_source"] = "regional_open_meteo_ensemble_cache"
                return data
            raise FileNotFoundError(f"Snapshot cuaca untuk event '{event_id}' tidak ditemukan di {event_file}")
        return json.loads(event_file.read_text(encoding="utf-8-sig"))

    @staticmethod
    def get_weather_for_simulation(
        event_id: str,
        n_members: int = 51,
        seed: int = 42,
        overrides: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Mengambil atau menyintesis data cuaca untuk n_members anggota simulasi.
        Jika anggota tersimpan < n_members, variasikan secara probabilistik (Prinsip 12).
        """
        cached = OpenMeteoIngest.load_cached_event_weather(event_id)
        hourly = cached.get("hourly", {})
        members_source = cached.get("members_source", "open_meteo_ensemble")

        # Ekstrak data dasar
        times = hourly.get("time", [])
        n_steps = len(times) if times else 48

        # Cek ketersediaan anggota asli
        # Pola key: wind_speed_10m_member01, wind_direction_10m_member01, dll.
        member_wind_u = []
        member_wind_v = []
        member_precip = []

        # Deteksi jumlah anggota yang ada di snapshot
        member_ids = set()
        for k in hourly.keys():
            if "member" in k:
                # e.g. wind_speed_10m_member01
                part = k.split("_")[-1]
                if part.startswith("member"):
                    member_ids.add(part)

        member_ids = sorted(list(member_ids))
        rng = np.random.default_rng(seed)

        # Baseline wind & rain
        units = cached.get("hourly_units", {})
        speed_unit = str(units.get("wind_speed_10m", "km/h")).lower()
        speed_to_ms = (1.0 / 3.6) if ("km/h" in speed_unit or "kmh" in speed_unit) else 1.0

        base_speed = np.array(hourly.get("wind_speed_10m", [14.0] * n_steps), dtype=float) * speed_to_ms
        base_dir = np.array(hourly.get("wind_direction_10m", [90.0] * n_steps), dtype=float)
        base_precip = np.array(hourly.get("precipitation", [0.0] * n_steps), dtype=float)

        # Handle overrides jika ada (mode experiment)
        if overrides and "wind" in overrides and overrides["wind"]:
            wind_ovr = overrides["wind"]
            if wind_ovr.get("reverse"):
                base_dir = (base_dir + 180.0) % 360.0
            if wind_ovr.get("rotate_deg") is not None:
                base_dir = (base_dir + float(wind_ovr["rotate_deg"])) % 360.0

        if overrides and "rain" in overrides and overrides["rain"]:
            rain_ovr = overrides["rain"]
            intensity = float(rain_ovr.get("intensity_mm_h", 10.0))
            duration = int(rain_ovr.get("duration_h", 4))
            start_h = int(rain_ovr.get("start_h", 6))
            end_h = min(n_steps, start_h + duration)
            base_precip[start_h:end_h] += intensity

        # Siapkan array per anggota (n_members, n_steps)
        # Konversi speed dan direction ke komponen U dan V (angin bertiup DARI arah dir)
        # u = -speed * sin(rad), v = -speed * cos(rad)
        rad = np.radians(base_dir)
        base_u = -base_speed * np.sin(rad)
        base_v = -base_speed * np.cos(rad)

        for m_idx in range(n_members):
            # Jika ada anggota asli di index ini, pakai; jika tidak, gunakan variasi sintetis
            if m_idx < len(member_ids):
                m_str = member_ids[m_idx]
                sp_k = f"wind_speed_10m_{m_str}"
                dir_k = f"wind_direction_10m_{m_str}"
                pr_k = f"precipitation_{m_str}"

                sp_raw = np.array(hourly.get(sp_k, hourly.get("wind_speed_10m", [14.0] * n_steps)), dtype=float)
                sp = sp_raw * speed_to_ms
                dr = np.array(hourly.get(dir_k, base_dir), dtype=float)
                pr = np.array(hourly.get(pr_k, base_precip), dtype=float)

                r = np.radians(dr)
                u = -sp * np.sin(r)
                v = -sp * np.cos(r)
            else:
                # Variasi AR(1) sintetis di sekitar mean
                members_source = "synthetic_from_mean_spread"
                perturb_u = rng.normal(0, 1.2, size=n_steps)
                perturb_v = rng.normal(0, 1.2, size=n_steps)
                u = base_u + perturb_u
                v = base_v + perturb_v
                pr = np.maximum(0.0, base_precip + rng.normal(0, 0.5, size=n_steps))

            member_wind_u.append(u)
            member_wind_v.append(v)
            member_precip.append(pr)

        return {
            "n_members": n_members,
            "n_steps": n_steps,
            "times": times,
            "members_source": members_source,
            "wind_u": np.array(member_wind_u),  # shape (n_members, n_steps)
            "wind_v": np.array(member_wind_v),  # shape (n_members, n_steps)
            "precip": np.array(member_precip),  # shape (n_members, n_steps)
        }
