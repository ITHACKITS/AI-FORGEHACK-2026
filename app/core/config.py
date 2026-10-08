"""
app/core/config.py

Konfigurasi aplikasi dan pemuat file asumsi & bahaya.
Menghindari kebocoran kunci API dan memastikan status asumsi tercatat.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BACKEND_DIR / "config"
DATA_DIR = BACKEND_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
RUNS_DIR = DATA_DIR / "runs"
DOCS_DIR = BACKEND_DIR / "docs"

# Buat folder penting
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    PROJECT_NAME: str = "UdaraSiaga"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    DESCRIPTION: str = (
        "Simulator sebaran abu gunung meletus dan asap kebakaran hutan "
        "untuk Hackathon JOINTS UGM 2026. Aplikasi ini simulasi untuk kesiapsiagaan, BUKAN peringatan resmi."
    )
    OFFLINE: bool = os.environ.get("UDARASIAGA_OFFLINE", "1") == "1"
    FIRMS_MAP_KEY: str | None = os.environ.get("FIRMS_MAP_KEY")
    OPENAQ_API_KEY: str | None = os.environ.get("OPENAQ_API_KEY")

    DISCLAIMER: str = (
        "Sistem ini adalah platform estimasi sains untuk kesiapsiagaan dini & protokol aksi pencegahan bencana. "
        "Untuk koordinasi evakuasi otoritatif di lapangan, tetap rujuk arahan resmi PVMBG Badan Geologi, BMKG, dan BPBD."
    )


settings = Settings()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


_assumptions_cache = None
_hazards_cache = None


def get_assumptions() -> Dict[str, Any]:
    global _assumptions_cache
    if _assumptions_cache is None:
        _assumptions_cache = load_yaml(CONFIG_DIR / "assumptions.yaml")
    return _assumptions_cache


def get_hazards() -> Dict[str, Any]:
    global _hazards_cache
    if _hazards_cache is None:
        _hazards_cache = load_yaml(CONFIG_DIR / "hazards.yaml")
    return _hazards_cache


def get_assumptions_used(keys: list[str] | None = None) -> list[Dict[str, Any]]:
    """
    Mengembalikan daftar parameter yang dipakai yang belum berstatus 'sourced' murni,
    atau seluruh parameter terkait jika keys ditentukan.
    """
    assumptions = get_assumptions()
    result = []
    target_keys = keys if keys is not None else list(assumptions.keys())

    for k in target_keys:
        item = assumptions.get(k)
        if item and isinstance(item, dict):
            status = item.get("status", "assumption_placeholder")
            result.append({
                "parameter": k,
                "value": item.get("value"),
                "status": status,
                "source": item.get("source"),
                "caveat": item.get("caveat"),
            })
    return result


get_assumptions_list = get_assumptions_used


def is_pm25_emission_factor_sourced() -> bool:
    """
    Prinsip 11 AGENTS.md:
    Kategori ISPU hanya dihitung bila faktor emisi PM2.5 (D13b) berstatus sourced*.
    """
    assumptions = get_assumptions()
    ef_item = assumptions.get("emission_factor_pm25_g_per_kg", {})
    status = ef_item.get("status", "")
    return status.startswith("sourced")
