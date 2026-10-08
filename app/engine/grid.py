"""
app/engine/grid.py

Definisi grid reguler lat/lon untuk domain simulasi.
Menghitung konversi jarak, area sel, dan pemetaan koordinat ke indeks grid.
"""

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6371000.0


@dataclass
class SimulationGrid:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float = 0.02

    def __post_init__(self):
        self.n_lat = int(np.ceil((self.lat_max - self.lat_min) / self.resolution_deg)) + 1
        self.n_lon = int(np.ceil((self.lon_max - self.lon_min) / self.resolution_deg)) + 1
        self.lats = np.linspace(self.lat_min, self.lat_max, self.n_lat)
        self.lons = np.linspace(self.lon_min, self.lon_max, self.n_lon)

        # Hitung area per sel dalam m² (berdasarkan latitude sel)
        d_lat_rad = np.radians(self.resolution_deg)
        d_lon_rad = np.radians(self.resolution_deg)
        dy = EARTH_RADIUS_M * d_lat_rad
        # dx bervariasi mengikuti cos(lat)
        dx_per_lat = EARTH_RADIUS_M * np.cos(np.radians(self.lats)) * d_lon_rad
        self.cell_area_m2 = np.outer(dx_per_lat * dy, np.ones(self.n_lon))

    def coord_to_index(self, lat: float | np.ndarray, lon: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Konversi latitude dan longitude ke indeks baris (lat) dan kolom (lon)."""
        row = np.clip(np.round((lat - self.lat_min) / self.resolution_deg).astype(int), 0, self.n_lat - 1)
        col = np.clip(np.round((lon - self.lon_min) / self.resolution_deg).astype(int), 0, self.n_lon - 1)
        return row, col

    def index_to_coord(self, row: int, col: int) -> tuple[float, float]:
        """Konversi indeks baris dan kolom ke lat/lon tengah sel."""
        return float(self.lats[row]), float(self.lons[col])

    @classmethod
    def create_for_volcano(cls, center_lat: float, center_lon: float, domain_km: float = 300.0, res_deg: float = 0.02):
        # 1 derajat latitude ≈ 111 km
        half_deg_lat = (domain_km / 2.0) / 111.0
        half_deg_lon = (domain_km / 2.0) / (111.0 * np.cos(np.radians(center_lat)))
        return cls(
            lat_min=center_lat - half_deg_lat,
            lat_max=center_lat + half_deg_lat,
            lon_min=center_lon - half_deg_lon,
            lon_max=center_lon + half_deg_lon,
            resolution_deg=res_deg,
        )

    @classmethod
    def create_for_fire(cls, bbox: list[float], buffer_km: float = 150.0, res_deg: float = 0.02):
        west, south, east, north = bbox
        center_lat = (south + north) / 2.0
        buf_deg_lat = buffer_km / 111.0
        buf_deg_lon = buffer_km / (111.0 * np.cos(np.radians(center_lat)))
        return cls(
            lat_min=south - buf_deg_lat,
            lat_max=north + buf_deg_lat,
            lon_min=west - buf_deg_lon,
            lon_max=east + buf_deg_lon,
            resolution_deg=res_deg,
        )
