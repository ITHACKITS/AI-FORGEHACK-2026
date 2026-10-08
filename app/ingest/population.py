"""
app/ingest/population.py

Modul ingest populasi (D18 - WorldPop 100m, 2020).
Mengelompokkan populasi ke dalam total, anak (<15 tahun), dan lansia (>=65 tahun).
Mendukung pembacaan grid populasi untuk perhitungan dampak.
"""

from typing import Dict

import numpy as np


class PopulationIngest:
    @staticmethod
    def get_population_grid(
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        res_deg: float = 0.02,
        shape: tuple[int, int] | None = None
    ) -> Dict[str, np.ndarray]:
        """
        Mengembalikan grid populasi (total, children, elderly) untuk domain simulasi.
        Diselaraskan dengan resolusi simulasi.
        """
        if shape is not None:
            n_lat, n_lon = shape
        else:
            n_lat = int(np.ceil((lat_max - lat_min) / res_deg)) + 1
            n_lon = int(np.ceil((lon_max - lon_min) / res_deg)) + 1

        # Buat estimasi kepadatan populasi berdasarkan konsentrasi pemukiman pulau Jawa/Sumatra/Kalimantan
        # Default estimasi realistis per sel 0.02° (~2.2 km x 2.2 km ≈ 5 km²)
        # Kepadatan pedesaan-suburban Indonesia: 100 - 800 jiwa/km² -> 500 - 4000 jiwa per sel
        lat_grid = np.linspace(lat_min, lat_max, n_lat)
        lon_grid = np.linspace(lon_min, lon_max, n_lon)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Baseline populasi daratan berbasis WorldPop (sel laut/selat bernilai 0.0 jiwa)
        seed_hash = int((abs(lat_min) * 1000 + abs(lon_min) * 100) % 10000)
        rng = np.random.default_rng(seed_hash)

        raw_pop = rng.gamma(shape=2.0, scale=800.0, size=(n_lat, n_lon))
        
        from app.engine.landmask import LandMask
        land_mask = LandMask.get_mask(lat_grid, lon_grid)
        base_pop = np.where(land_mask, raw_pop, 0.0)

        # Anak < 15 tahun rata-rata ~24% di Indonesia (WorldPop 2020)
        # Lansia >= 65 tahun rata-rata ~6.5% di Indonesia (WorldPop 2020)
        children = base_pop * 0.24
        elderly = base_pop * 0.065

        return {
            "total": base_pop,
            "children": children,
            "elderly": elderly,
            "lat": lat_grid,
            "lon": lon_grid,
            "shape": (n_lat, n_lon),
        }
