"""
app/engine/landmask.py

Modul penentuan daratan vs lautan (Land/Sea Mask) berbasis data garis pantai
resmi Indonesia (data/layers/indonesia_coastline.geojson).
Mencegah estimasi populasi, lahan sawah, dan fasilitas fiktif di tengah lautan/selat.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from app.core.config import BACKEND_DIR


class LandMask:
    _unified_land: Optional[shapely.Geometry] = None

    @classmethod
    def _init_land_geometry(cls):
        if cls._unified_land is not None:
            return

        coastline_file = BACKEND_DIR / "data" / "layers" / "indonesia_coastline.geojson"
        if not coastline_file.exists():
            return

        with open(coastline_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        polygons = []
        for feat in data.get("features", []):
            coords = feat.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue
            # Pastikan poligon tertutup
            if coords[0] != coords[-1]:
                coords = list(coords) + [coords[0]]
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append(poly)

        if polygons:
            cls._unified_land = unary_union(polygons)

    @classmethod
    def is_point_on_land(cls, lat: float, lon: float) -> bool:
        """Mengecek apakah suatu koordinat (lat, lon) berada di daratan kepulauan Indonesia."""
        cls._init_land_geometry()
        if cls._unified_land is None:
            return True  # Fallback toleran jika file layer tidak ditemukan
        return bool(shapely.contains_xy(cls._unified_land, lon, lat))

    @classmethod
    def get_mask(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Menghasilkan mask 2D boolean berukuran (len(lats), len(lons))
        True jika sel berada di daratan, False jika berada di laut/selat lepas.
        """
        cls._init_land_geometry()
        n_lat = len(lats)
        n_lon = len(lons)

        if cls._unified_land is None:
            return np.ones((n_lat, n_lon), dtype=bool)

        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        mask = shapely.contains_xy(cls._unified_land, lon_mesh, lat_mesh)
        return np.asarray(mask, dtype=bool)
