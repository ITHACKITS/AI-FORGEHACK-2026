"""
app/llm/brief.py - Household safety brief written by an LLM from backend facts.

The backend sends grounded facts about one location (nearby volcanoes, fire detections,
fire-prone forests and the 72 h smoke/ash forecast) plus a scenario it chose from those
facts. This module asks Featherless (OpenAI-compatible) to turn them into a short brief
in English or Indonesian, choosing actions from a fixed playbook.

Rules (see AGENTS.md): numbers only come from the facts, at most 2 LLM calls per request,
a timeout on every call, and a deterministic template when the LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Scenario = Literal["ash_threat", "smoke_threat", "prevention", "clear"]
Lang = Literal["en", "id"]

PRIMARY_MODEL = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen3.6-35B-A3B")
FALLBACK_MODEL = os.getenv("FEATHERLESS_FALLBACK_MODEL", "Qwen/Qwen3.5-9B")
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "12"))


class BriefRequest(BaseModel):
    scenario: Scenario
    lang: Lang = "en"
    facts: dict[str, Any] = Field(default_factory=dict)


# Actions the model may choose from, per scenario and language. The template uses the first ones.
PLAYBOOK: dict[str, dict[str, list[str]]] = {
    "ash_threat": {
        "en": [
            "Wear an N95/KN95 mask outdoors; cloth and surgical masks do not stop fine ash.",
            "Remove contact lenses and wear sealed goggles when ash falls.",
            "Cover water tanks, wells and food, and keep at least 3 days of drinking water at home.",
            "Keep children and older relatives indoors with doors and windows shut.",
            "Prepare an evacuation bag (documents, medicine, torch) and follow PVMBG and BPBD instructions.",
            "Clear ash from the roof early while wearing a mask; wet ash is heavy."
        ],
        "id": [
            "Pakai masker N95/KN95 saat di luar rumah; masker kain dan masker bedah tidak menahan abu halus.",
            "Lepas lensa kontak dan pakai kacamata pelindung tertutup saat hujan abu.",
            "Tutup tandon air, sumur, dan makanan, serta siapkan air minum untuk minimal 3 hari.",
            "Jaga anak-anak dan lansia tetap di dalam rumah dengan pintu dan jendela tertutup.",
            "Siapkan tas siaga (dokumen, obat, senter) dan ikuti arahan PVMBG serta BPBD.",
            "Bersihkan abu di atap sejak dini sambil memakai masker; abu basah sangat berat."
        ]
    },
    "smoke_threat": {
        "en": [
            "Buy or prepare an N95/KN95 mask for every household member before the smoke arrives.",
            "Stock drinking water and regular medicines, especially for asthma and heart conditions.",
            "Keep windows and doors closed and make one clean room with an air purifier or a fan with a filter.",
            "Limit outdoor activity for children, older people and pregnant women.",
            "Check on neighbours who have breathing problems.",
            "Follow official air-quality (ISPU) readings and BPBD instructions."
        ],
        "id": [
            "Beli atau siapkan masker N95/KN95 untuk setiap anggota keluarga sebelum asap datang.",
            "Siapkan air minum dan obat rutin, terutama untuk penderita asma dan penyakit jantung.",
            "Tutup jendela dan pintu, lalu siapkan satu ruangan bersih dengan pembersih udara atau kipas berfilter.",
            "Batasi aktivitas luar ruangan bagi anak-anak, lansia, dan ibu hamil.",
            "Periksa kondisi tetangga yang punya gangguan pernapasan.",
            "Pantau angka kualitas udara (ISPU) resmi dan ikuti arahan BPBD."
        ]
    },
    "prevention": {
        "en": [
            "Water the ground and plants around your home and fields, especially in the dry afternoon hours.",
            "Clear dry leaves, grass and rubbish within about 10 m of buildings.",
            "Never burn rubbish or clear land with fire; peat can smoulder underground for weeks.",
            "Help keep firebreaks (cleared strips) between gardens, peatland and houses.",
            "Keep buckets, water drums or a hose ready near the house.",
            "Report smoke or fire early to BPBD or the Manggala Agni fire brigade."
        ],
        "id": [
            "Siram tanah dan tanaman di sekitar rumah dan kebun, terutama saat siang hingga sore yang kering.",
            "Bersihkan daun kering, rumput, dan sampah dalam radius sekitar 10 m dari bangunan.",
            "Jangan membakar sampah atau membuka lahan dengan api; gambut bisa membara di bawah tanah berminggu-minggu.",
            "Bantu jaga sekat bakar (jalur bersih tanpa vegetasi) antara kebun, lahan gambut, dan rumah.",
            "Siapkan ember, drum air, atau selang di dekat rumah.",
            "Laporkan asap atau api sedini mungkin ke BPBD atau Manggala Agni."
        ]
    },
    "clear": {
        "en": [
            "Keep a few N95/KN95 masks and drinking water at home for the haze season.",
            "Never burn rubbish or clear land with fire.",
            "Check StandbAI again before travelling to fire-prone or volcanic areas.",
            "Follow BMKG, PVMBG and BPBD updates."
        ],
        "id": [
            "Simpan beberapa masker N95/KN95 dan air minum di rumah untuk musim kabut asap.",
            "Jangan membakar sampah atau membuka lahan dengan api.",
            "Cek StandbAI lagi sebelum bepergian ke daerah rawan kebakaran atau gunung api.",
            "Ikuti informasi terbaru dari BMKG, PVMBG, dan BPBD."
        ]
    }
}


def _fmt(n: float | None, lang: Lang) -> str:
    if n is None:
        return "-"
    s = f"{n:,.0f}" if abs(n) >= 10 else f"{n:,.1f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "id" else s


def _pass_text(what: str, path: dict[str, Any], lang: Lang) -> str:
    km, hours = _fmt(path.get("closest_km"), lang), path.get("hours_from_now")
    if lang == "id":
        return (f"Sumber {what} hanya sekitar {km} km dari sini." if hours is None
                else f"Sebaran {what} diperkirakan melintas sekitar {km} km dari sini dalam ±{hours} jam.")
    return (f"The {what.lower()} source is only about {km} km from here." if hours is None
            else f"{what} may pass about {km} km from here in about {hours} hours.")


def _template(req: BriefRequest) -> dict[str, Any]:
    """Deterministic brief from the facts, used when the LLM is unavailable."""
    f, lang, sc = req.facts, req.lang, req.scenario
    hs = f.get("hotspots") or {}
    smoke, ash = f.get("smoke_forecast"), f.get("ash_forecast")
    volc = (f.get("volcanoes_nearby") or [None])[0]
    forest = f.get("fire_prone_forest")
    id_ = lang == "id"
    if sc == "ash_threat":
        head = "Abu vulkanik bisa mencapai area ini" if id_ else "Volcanic ash may reach this area"
        parts = []
        if volc:
            parts.append(f"{volc['name']} (Level {volc['level']}) {'berjarak' if id_ else 'is'} {_fmt(volc['distance_km'], lang)} km{'' if id_ else ' away'}.")
        if ash:
            parts.append(_pass_text("abu" if id_ else "Ash", ash, lang))
    elif sc == "smoke_threat":
        head = "Asap kebakaran bisa mencapai area ini" if id_ else "Wildfire smoke may reach this area"
        parts = [(f"Ada {hs.get('within_50km', 0)} titik api dalam radius 50 km (terdekat {_fmt(hs.get('nearest_km'), lang)} km)."
                  if id_ else f"There are {hs.get('within_50km', 0)} fire detections within 50 km (nearest {_fmt(hs.get('nearest_km'), lang)} km).")]
        if smoke:
            parts.append(_pass_text("asap" if id_ else "Smoke", smoke, lang))
    elif sc == "prevention":
        head = "Belum ada api di dekat sini, saatnya mencegah" if id_ else "No fire nearby yet: time to prevent one"
        parts = []
        if forest:
            parts.append(f"Kawasan rawan api {forest['name']} berjarak {_fmt(forest['distance_km'], lang)} km."
                          if id_ else f"The fire-prone area {forest['name']} is {_fmt(forest['distance_km'], lang)} km away.")
        parts.append("Asap dan api belum diperkirakan sampai ke sini dalam 72 jam." if id_ else "No smoke or fire is forecast to reach here within 72 hours.")
    else:
        head = "Tidak ada ancaman asap atau abu di dekat sini" if id_ else "No smoke or ash threat nearby"
        parts = ["Tidak ada titik api, abu, atau asap yang diperkirakan mendekat dalam 72 jam." if id_
                 else "No fires, ash or smoke are forecast to come close within 72 hours."]
    return {"headline": head, "summary": " ".join(parts), "actions": PLAYBOOK[sc][lang][:5]}


def _system_prompt(lang: Lang) -> str:
    language = "Indonesian (Bahasa Indonesia)" if lang == "id" else "English"
    return (
        "You are StandbAI's household safety writer for Indonesia. Write a short brief for families "
        "using ONLY the FACTS you are given.\n"
        "Rules:\n"
        "1. Use only numbers that appear in FACTS. Never invent numbers, dates, places or agencies.\n"
        "2. Choose 4 or 5 actions from PLAYBOOK and adapt them to the facts. Always keep the first two PLAYBOOK "
        "actions. Do not add medical claims that are not in PLAYBOOK.\n"
        f"3. Write everything in {language}. Translate place descriptors into {language}: "
        "'G. Merapi' is 'Mount Merapi' in English, 'TN Sebangau' is 'Sebangau National Park' in English.\n"
        "4. Meaning of the facts: hotspots are satellite fire detections near the location; smoke_forecast and "
        "ash_forecast say how close a forecast plume passes and in how many hours; fire_prone_forest is a nearby "
        "area at risk of burning, not a confirmed fire; ai_detection_check is a machine-learning check of the "
        "nearby detections.\n"
        "5. hours_from_now is null when the location is next to the source itself (near_source true): say the "
        "location is close to the source instead of giving an arrival time.\n"
        "6. Calm, plain language for parents. No markdown. No dates in the headline. Do not use official alert words "
        "(warning, alert, Awas, Siaga, Waspada, Peringatan) in the headline.\n"
        "7. It is not an official warning; do not call it one.\n"
        'Output strict JSON only: {"headline": string (max 12 words), "summary": string (max 2 sentences), '
        '"actions": [4 or 5 strings]}'
    )


def _parse(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    head, summary, actions = obj.get("headline"), obj.get("summary"), obj.get("actions")
    if not isinstance(head, str) or not isinstance(summary, str) or not isinstance(actions, list):
        return None
    actions = [a.strip() for a in actions if isinstance(a, str) and a.strip()][:6]
    if len(actions) < 3:
        return None
    return {"headline": head.strip()[:140], "summary": summary.strip()[:600], "actions": [a[:220] for a in actions]}


def _call_featherless(model: str, req: BriefRequest, key: str) -> dict[str, Any] | None:
    base = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1").rstrip("/")
    user = (
        f"SCENARIO: {req.scenario}\n"
        f"FACTS: {json.dumps(req.facts, ensure_ascii=False)}\n"
        f"PLAYBOOK: {json.dumps(PLAYBOOK[req.scenario][req.lang], ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [{"role": "system", "content": _system_prompt(req.lang)}, {"role": "user", "content": user}],
        "max_tokens": 600,
        "temperature": 0.3,
        # Qwen3 models think by default; the brief needs the answer only.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "standbai-ai/2.0", "Content-Type": "application/json"}
    try:
        r = httpx.post(f"{base}/chat/completions", json=body, headers=headers, timeout=TIMEOUT_S)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content") or ""
        return _parse(content)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:  # fall through to the next engine
        logger.warning("Featherless call failed (%s): %s", model, type(e).__name__)
        return None


def generate_brief(req: BriefRequest) -> dict[str, Any]:
    """Featherless primary model, then the fallback model, then the template (at most 2 LLM calls)."""
    key = os.getenv("FEATHERLESS_API_KEY", "").strip()
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if key:
        for model in (PRIMARY_MODEL, FALLBACK_MODEL):
            out = _call_featherless(model, req, key)
            if out:
                return {**out, "engine": "featherless", "model": model, "generated_at": now}
    return {**_template(req), "engine": "template", "model": None, "generated_at": now}
