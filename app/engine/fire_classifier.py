"""
app/engine/fire_classifier.py

Inference Engine untuk Fire Predictor Model Global (NASA FIRMS VIIRS 2020-2025).
Memuat bundle model ML (Random Forest / Gradient Boosting / Voting Ensemble)
dan memprediksi probabilitas kebakaran hutan serta tingkat risiko secara real-time.
"""

import os
import pickle
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("fire_classifier")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fire_predictor_global.pkl")

FEATURES = [
    "latitude", "longitude", "brightness", "frp", "frp_log", "confidence",
    "is_day", "dist_equator", "is_northern", "is_east_hemisphere",
    "brightness_delta", "fire_intensity"
]

class FireClassifier:
    _instance = None
    _bundle = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FireClassifier, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._bundle = pickle.load(f)
                self._is_loaded = True
                logger.info(
                    f"FireClassifier loaded: {self._bundle.get('model_type')} "
                    f"(Accuracy={self._bundle.get('accuracy', 0)*100:.2f}%, AUC={self._bundle.get('roc_auc', 0):.4f})"
                )
            except Exception as e:
                logger.warning(f"Gagal memuat model fire predictor: {e}. Menggunakan fallback heuristik.")
                self._is_loaded = False
        else:
            logger.warning(f"File model tidak ditemukan di {MODEL_PATH}. Menggunakan fallback heuristik.")
            self._is_loaded = False

    @property
    def metadata(self) -> Dict[str, Any]:
        if not self._is_loaded or not self._bundle:
            return {"status": "fallback_heuristic", "model_loaded": False}
        return {
            "model_type": self._bundle.get("model_type", "Unknown"),
            "accuracy": self._bundle.get("accuracy"),
            "roc_auc": self._bundle.get("roc_auc"),
            "cv_f1": self._bundle.get("cv_f1"),
            "training_samples": self._bundle.get("training_samples"),
            "data_source": self._bundle.get("data_source"),
            "features": self._bundle.get("features", FEATURES),
            "model_loaded": True
        }

    def _extract_feature_vector(
        self,
        lat: float,
        lon: float,
        brightness: float,
        frp: float,
        confidence: float,
        is_day: int
    ) -> np.ndarray:
        dist_eq = abs(lat)
        is_north = 1 if lat >= 0 else 0
        is_east = 1 if lon >= 0 else 0
        bright_delta = max(0.0, brightness - 330.0)
        fire_int = frp * (confidence / 100.0)
        frp_log = np.log1p(frp)

        return np.array([
            lat, lon, brightness, frp, frp_log, confidence,
            is_day, dist_eq, is_north, is_east,
            bright_delta, fire_int
        ], dtype=float).reshape(1, -1)

    def predict_point(
        self,
        latitude: float,
        longitude: float,
        brightness: Optional[float] = None,
        frp: Optional[float] = None,
        confidence: Optional[float] = None,
        is_day: bool = True
    ) -> Dict[str, Any]:
        """
        Prediksi probabilitas kebakaran dan level risiko untuk 1 titik koordinat.
        Jika nilai sensor (brightness/frp) kosong, estimasi default diberikan.
        """
        # Nilai default jika observasi mentah
        b_val = brightness if brightness is not None else (335.0 if (frp and frp > 5.0) else 300.0)
        f_val = frp if frp is not None else 0.0
        c_val = confidence if confidence is not None else (80.0 if f_val > 5.0 else 20.0)
        day_int = 1 if is_day else 0

        # Heuristik fallback jika model .pkl belum ada/gagal di-unpickle
        if not self._is_loaded or not self._bundle:
            prob = 0.95 if (f_val > 5.0 or b_val > 330.0) else (0.1 if f_val == 0 else 0.5)
            is_fire = prob >= 0.5
            return {
                "latitude": latitude,
                "longitude": longitude,
                "is_fire": is_fire,
                "fire_probability": round(prob, 4),
                "risk_level": self._classify_risk(prob, f_val),
                "model_engine": "Fallback Heuristic",
                "features_used": {
                    "brightness": b_val, "frp": f_val, "confidence": c_val, "is_day": day_int
                }
            }

        # Menggunakan Machine Learning Model Resmi NASA FIRMS
        vec = self._extract_feature_vector(latitude, longitude, b_val, f_val, c_val, day_int)
        scaler = self._bundle["scaler"]
        model = self._bundle["model"]

        vec_scaled = scaler.transform(vec)
        proba = model.predict_proba(vec_scaled)[0, 1]
        pred_label = int(model.predict(vec_scaled)[0])

        return {
            "latitude": latitude,
            "longitude": longitude,
            "is_fire": bool(pred_label == 1),
            "fire_probability": round(float(proba), 4),
            "risk_level": self._classify_risk(proba, f_val),
            "model_engine": f"{self._bundle.get('model_type')} (NASA FIRMS 2020-2025)",
            "features_used": {
                "brightness": b_val, "frp": f_val, "confidence": c_val, "is_day": day_int
            }
        }

    def predict_batch(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch prediction untuk kumpulan titik api.
        """
        results = []
        for h in hotspots:
            lat = float(h.get("latitude", 0.0))
            lon = float(h.get("longitude", 0.0))
            bright = float(h["brightness"]) if "brightness" in h and h["brightness"] is not None else None
            frp = float(h["frp"]) if "frp" in h and h["frp"] is not None else None
            
            # Konversi confidence jika string ("high", "nominal", "low")
            conf_raw = h.get("confidence", 70.0)
            if isinstance(conf_raw, str):
                c_map = {"h": 90.0, "high": 90.0, "n": 60.0, "nominal": 60.0, "l": 30.0, "low": 30.0}
                conf = c_map.get(conf_raw.lower(), 50.0)
            else:
                conf = float(conf_raw)

            is_day = h.get("is_day", True)
            res = self.predict_point(lat, lon, bright, frp, conf, is_day)
            if "id" in h:
                res["id"] = h["id"]
            results.append(res)
        return results

    def _classify_risk(self, prob: float, frp: float) -> str:
        if prob < 0.3:
            return "AMAN / LOW"
        elif prob < 0.6:
            return "WASPADA / MODERATE"
        elif prob < 0.85 or frp < 30.0:
            return "SIAGA / HIGH"
        else:
            return "AWAS / CRITICAL"
