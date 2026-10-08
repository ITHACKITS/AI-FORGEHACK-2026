"""
app/ingest/firms.py

Modul ingest untuk titik api dan FRP dari NASA FIRMS (D09).
Mendukung pemuatan CSV lokal dan penyesuaian (tambah/hapus) di mode experiment.
"""

from typing import Any, Dict, List

import pandas as pd

from app.core.config import EVENTS_DIR


class FirmsIngest:
    @staticmethod
    def load_event_hotspots(event_id: str | None = None, overrides: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        # Mode Nasional: Kumpulkan titik api dari seluruh kawasan karhutla se-Indonesia
        if not event_id or event_id in ["national", "all", "indonesia"]:
            records = []
            for d in sorted(EVENTS_DIR.iterdir()):
                if d.is_dir() and (d.name.startswith("karhutla_") or d.name.startswith("hutan_") or d.name == "live_kalimantan"):
                    csv_path = d / "hotspots.csv"
                    if csv_path.exists() and csv_path.stat().st_size > 0:
                        try:
                            df = pd.read_csv(csv_path)
                            for idx, row in df.iterrows():
                                lat = row.get("latitude")
                                lon = row.get("longitude")
                                if pd.isna(lat) or pd.isna(lon):
                                    continue
                                frp_val = row.get("frp", 25.0)
                                frp_float = float(frp_val) if not pd.isna(frp_val) else 25.0
                                records.append({
                                    "id": f"{d.name}_{idx}",
                                    "latitude": float(lat),
                                    "longitude": float(lon),
                                    "frp": frp_float,
                                    "confidence": str(row.get("confidence", "nominal")),
                                    "acq_date": str(row.get("acq_date", "")),
                                    "acq_time": str(row.get("acq_time", "")),
                                    "is_active": True,
                                })
                        except Exception:
                            continue
            return records

        csv_path = EVENTS_DIR / event_id / "hotspots.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return []

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return []
        except Exception:
            return []
        # Standarisasi kolom
        records = []
        for idx, row in df.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            frp_val = row.get("frp", 25.0)
            frp_float = float(frp_val) if not pd.isna(frp_val) else 25.0
            rec = {
                "id": f"hotspot_{idx}",
                "latitude": float(lat),
                "longitude": float(lon),
                "frp": frp_float,
                "confidence": str(row.get("confidence", "nominal")),
                "acq_date": str(row.get("acq_date", "")),
                "acq_time": str(row.get("acq_time", "")),
                "is_active": True,
            }
            records.append(rec)

        # Proses overrides (tambah / padamkan titik api)
        if overrides and "hotspots" in overrides and overrides["hotspots"]:
            h_ovr = overrides["hotspots"]
            remove_ids = set(h_ovr.get("remove", []))
            if remove_ids:
                records = [r for r in records if r["id"] not in remove_ids]

            for new_pt in h_ovr.get("add", []):
                records.append({
                    "id": f"added_{len(records)}",
                    "latitude": float(new_pt["latitude"]),
                    "longitude": float(new_pt["longitude"]),
                    "frp": float(new_pt.get("frp", 30.0)),
                    "confidence": "high",
                    "acq_date": "experiment",
                    "acq_time": "0000",
                    "is_active": True,
                })

        return records
