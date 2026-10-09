"""
app/schemas/fire_prediction.py

Skema Pydantic v2 untuk input dan output prediksi kebakaran hutan (Fire Prediction ML).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FirePointRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Koordinat lintang")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Koordinat bujur")
    brightness: Optional[float] = Field(None, description="Suhu kecerahan I-4 VIIRS dalam Kelvin (opsional)")
    frp: Optional[float] = Field(None, ge=0.0, description="Fire Radiative Power dalam Megawatt (opsional)")
    confidence: Optional[float] = Field(None, ge=0.0, le=100.0, description="Tingkat kepercayaan deteksi 0-100 (opsional)")
    is_day: bool = Field(True, description="Apakah observasi dilakukan di siang hari")

class FireBatchRequest(BaseModel):
    points: List[FirePointRequest] = Field(..., description="Daftar titik koordinat untuk diprediksi secara bersamaan")

class FirePointResponse(BaseModel):
    latitude: float
    longitude: float
    is_fire: bool
    fire_probability: float
    risk_level: str
    model_engine: str
    features_used: Dict[str, Any]

class FireBatchResponse(BaseModel):
    status: str
    total_points: int
    fire_detected_count: int
    predictions: List[FirePointResponse]
    model_metadata: Dict[str, Any]
