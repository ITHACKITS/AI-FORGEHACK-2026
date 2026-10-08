"""
app/schemas/simulation.py

Skema Pydantic v2 untuk simulasi, request, respons, dan aturan integritas data.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RangeMetric(BaseModel):
    """
    Prinsip 1 AGENTS.md:
    Angka dampak selalu rentang {min, median, max}. Tidak ada angka tunggal.
    """
    min: float = Field(..., description="Batas bawah (persentil 5)")
    median: float = Field(..., description="Nilai tengah (persentil 50)")
    max: float = Field(..., description="Batas atas (persentil 95)")


class SimulationSource(BaseModel):
    event_id: Optional[str] = Field(None, description="ID event terdaftar di data/events/ atau ID kustom")
    latitude: Optional[float] = Field(None, ge=-12.0, le=7.0, description="Lintang kustom (-12 s.d. 7)")
    longitude: Optional[float] = Field(None, ge=94.0, le=142.0, description="Bujur kustom (94 s.d. 142)")
    name: Optional[str] = Field(None, description="Nama lokasi kustom")


class ThresholdsConfig(BaseModel):
    min: float = Field(0.90, ge=0.5, le=1.0, description="Ambang area minimum (hampir pasti)")
    mid: float = Field(0.50, ge=0.2, le=0.9, description="Ambang area tengah (kemungkinan besar)")
    max: float = Field(0.05, ge=0.01, le=0.3, description="Ambang area maksimum (mungkin)")


class SimulationOverrides(BaseModel):
    column_height_km: Optional[float] = None
    duration_h: Optional[float] = None
    wind: Optional[Dict[str, Any]] = None
    rain: Optional[Dict[str, Any]] = None
    rain_threshold_mm_h: Optional[float] = None
    hotspots: Optional[Dict[str, Any]] = None


class SimulationRequest(BaseModel):
    hazard: Literal["volcano", "fire"] = Field(..., description="Jenis bahaya")
    mode: Literal["auto", "experiment"] = Field("auto", description="Mode operasi")
    source: SimulationSource = Field(..., description="Sumber data kejadian")
    horizon_hours: int = Field(48, ge=1, le=72, description="Cakupan waktu simulasi (jam)")
    n_members: int = Field(51, ge=1, le=100, description="Jumlah anggota ensemble")
    seed: int = Field(42, description="Seed acak untuk determinisme")
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    overrides: Optional[Dict[str, Any]] = Field(None, description="Hanya diizinkan pada mode experiment")

    @model_validator(mode="after")
    def validate_overrides_and_mode(self):
        if self.mode == "auto" and self.overrides is not None and len(self.overrides) > 0:
            raise ValueError(
                "Mode 'auto' memakai data asli tanpa overrides (HTTP 422). "
                "Gunakan mode 'experiment' jika ingin mengubah parameter."
            )
        return self


class DecisionCardLocation(BaseModel):
    facility_id: str
    name: str
    facility_type: str
    latitude: float
    longitude: float
    exposure_probability: float
    arrival_time_earliest_h: Optional[float] = None
    arrival_time_most_likely_h: Optional[float] = None
    severity_p50: Optional[float] = None
    status: str = "evaluated"


class ImpactResult(BaseModel):
    population_exposed: RangeMetric
    children_exposed: RangeMetric
    elderly_exposed: RangeMetric
    schools_exposed: RangeMetric
    healthcare_exposed: RangeMetric
    airports_exposed: RangeMetric
    area_exposed_km2: RangeMetric
    ispu: Dict[str, Any]
    scenario_label: str
    disclaimer: str
    assumptions_used: List[Dict[str, Any]]


class SimulationRunMeta(BaseModel):
    run_id: str
    hazard: str
    mode: str
    event_id: str
    status: Literal["queued", "running", "completed", "failed"]
    scenario_label: str
    members_source: str
    n_members: int
    seed: int
    created_at_utc: str
    completed_at_utc: Optional[str] = None
    disclaimer: str
    assumptions_used: List[Dict[str, Any]]
