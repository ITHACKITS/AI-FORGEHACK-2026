"""
tests/test_advisory.py - Pengujian modul AI advisory & sintesis darurat
"""

import pytest
from app.impact.advisory import generate_emergency_advisory
from app.impact.gemini_advisory import enrich_advisory_with_gemini
from app.impact.validation import evaluate_event_validation

def test_generate_emergency_advisory():
    run_output = {
        "hazard": "volcano",
        "scenario_label": "Skenario Observasi Anak Krakatau",
        "disclaimer": "Hasil pemodelan deterministik."
    }
    impact_data = {
        "population_exposed": {"min": 5000, "median": 12000, "max": 25000},
        "children_exposed": {"min": 1000, "median": 2400, "max": 5000},
        "elderly_exposed": {"min": 500, "median": 1200, "max": 2500}
    }
    facilities_data = {
        "schools": [{"id": 1, "name": "SDN 1 Rajabasa"}],
        "healthcare": [{"id": 2, "name": "Puskesmas Rajabasa"}],
        "airports": [],
        "evacuation_referral_hospitals": []
    }
    adv = generate_emergency_advisory(run_output, impact_data, facilities_data)
    assert adv is not None
    assert "alert_level" in adv
    assert "ppe_guidance" in adv
    assert "school_guidance" in adv
    assert "health_guidance" in adv
    assert "population_summary" in adv

def test_evaluate_event_validation():
    val = evaluate_event_validation("anak_krakatau_2026_09")
    assert val is not None
    assert "event_id" in val
    assert val["event_id"] == "anak_krakatau_2026_09"