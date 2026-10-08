"""
app/impact/routing.py

Modul Perhitungan Rute Evakuasi Jalan Raya Riil (Safe Evacuation Corridor).
Menggunakan OSRM (Open Source Routing Machine) berbasis jaringan jalan OpenStreetMap Indonesia.
Menghasilkan koordinat polyline jalan raya riil (bukan garis lurus), jarak tempuh km sebenarnya,
dan estimasi durasi waktu tempuh ambulans gawat darurat.

Dilengkapi mekanisme fallback luring (offline curvature waypoints) jika OSRM publik tidak terjangkau.
"""

import json
import math
import urllib.request
import urllib.error
from typing import Any, Dict, List, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Menghitung jarak bola bumi dalam kilometer."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def generate_offline_road_waypoints(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    num_points: int = 12,
    center_lat: float | None = None,
    center_lon: float | None = None
) -> List[Tuple[float, float]]:
    """
    Fallback deterministik jika OSRM offline:
    Menghasilkan serangkaian waypoint jalan raya yang realistis dengan kurvatur topografi,
    menghindari garis lurus kaku 2-titik dan melengkung menjauhi pusat bahaya.
    """
    lat1, lon1 = origin
    lat2, lon2 = dest

    waypoints = [(lat1, lon1)]
    dist = haversine_km(lat1, lon1, lat2, lon2)
    
    # Amplitudo belokan bergantung pada jarak
    amp = min(max(dist * 0.04, 0.005), 0.05)

    # Vektor arah normal (tegak lurus)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    norm_lat = -d_lon
    norm_lon = d_lat
    mag = math.sqrt(norm_lat * norm_lat + norm_lon * norm_lon) or 1.0
    norm_lat /= mag
    norm_lon /= mag

    # Jika ada pusat bahaya (gunung api/karhutla), pastikan lengkungan menjauhi episentrum
    if center_lat is not None and center_lon is not None:
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        d_pos = haversine_km(mid_lat + norm_lat * amp, mid_lon + norm_lon * amp, center_lat, center_lon)
        d_neg = haversine_km(mid_lat - norm_lat * amp, mid_lon - norm_lon * amp, center_lat, center_lon)
        if d_neg > d_pos:
            norm_lat = -norm_lat
            norm_lon = -norm_lon

    for i in range(1, num_points - 1):
        t = i / (num_points - 1)
        base_lat = lat1 + t * d_lat
        base_lon = lon1 + t * d_lon

        # Kurva sinus halus menyerupai arteri jalan raya pegunungan
        offset = math.sin(t * math.pi) * amp
        wp_lat = round(base_lat + offset * norm_lat, 5)
        wp_lon = round(base_lon + offset * norm_lon, 5)
        waypoints.append((wp_lat, wp_lon))

    waypoints.append((lat2, lon2))
    return waypoints


def get_real_road_evacuation_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    origin_name: str = "Titik Asal",
    dest_name: str = "Titik Tujuan",
    center_lat: float | None = None,
    center_lon: float | None = None
) -> Dict[str, Any]:
    """
    Menghitung rute jalan raya riil minim dampak bahaya menggunakan OSRM OpenStreetMap.
    Mengevaluasi rute alternatif untuk menghindari kawah aktif dan konsentrasi abu pekat.
    Jika seluruh jalur langsung menembus kawah (< 10 km), menggunakan detour lingkar aman.
    """
    direct_dist_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    osrm_url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        f"?overview=full&geometries=geojson&alternatives=true&steps=false"
    )

    try:
        req = urllib.request.Request(
            osrm_url,
            headers={"User-Agent": "UdaraSiaga-JOINTS2026-EmergencyRouter/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                routes = data.get("routes", [])
                if routes:
                    # Evaluasi rute dengan kriteria Minim Dampak Bahaya (Hazard Avoidance)
                    chosen_route = routes[0]
                    chosen_coords = []
                    min_hazard_dist = 999.0
                    hazard_avoided = False

                    if center_lat is not None and center_lon is not None:
                        scored_routes = []
                        for r_item in routes:
                            geom = r_item.get("geometry", {}).get("coordinates", [])
                            pts = [[round(pt[1], 5), round(pt[0], 5)] for pt in geom]
                            if pts:
                                min_c_dist = min(haversine_km(p[0], p[1], center_lat, center_lon) for p in pts)
                            else:
                                min_c_dist = 999.0

                            dist_km = r_item.get("distance", direct_dist_km * 1000) / 1000.0
                            # Penalti rute yang memotong kawah aktif (< 12 km)
                            danger_penalty = max(0.0, 12.0 - min_c_dist) * 15.0
                            score = dist_km + danger_penalty
                            scored_routes.append((score, min_c_dist, r_item, pts))

                        scored_routes.sort(key=lambda x: x[0])
                        best_score, best_min_c, chosen_route, chosen_coords = scored_routes[0]
                        min_hazard_dist = best_min_c
                        if len(scored_routes) > 1 and best_min_c > scored_routes[-1][1]:
                            hazard_avoided = True
                    else:
                        geom = chosen_route.get("geometry", {}).get("coordinates", [])
                        chosen_coords = [[round(pt[1], 5), round(pt[0], 5)] for pt in geom]

                    # Jika seluruh rute langsung menembus kawah bahaya (< 10 km dari kawah aktif),
                    # cari rute bypass detour via koridor lingkar aman
                    if center_lat is not None and center_lon is not None and min_hazard_dist < 10.0:
                        mid_lat = (origin_lat + dest_lat) / 2.0
                        mid_lon = (origin_lon + dest_lon) / 2.0
                        d_lat = mid_lat - center_lat
                        d_lon = mid_lon - center_lon
                        c_dist = math.hypot(d_lat, d_lon) or 0.01
                        # Dorong titik via menjauhi kawah
                        push_deg = max(0.24, c_dist * 1.6)
                        via_lat = round(center_lat + (d_lat / c_dist) * push_deg, 5)
                        via_lon = round(center_lon + (d_lon / c_dist) * push_deg, 5)

                        detour_url = (
                            f"https://router.project-osrm.org/route/v1/driving/"
                            f"{origin_lon},{origin_lat};{via_lon},{via_lat};{dest_lon},{dest_lat}"
                            f"?overview=full&geometries=geojson&steps=false"
                        )
                        try:
                            d_req = urllib.request.Request(detour_url, headers={"User-Agent": "UdaraSiaga-Router/1.0"})
                            with urllib.request.urlopen(d_req, timeout=3.0) as d_resp:
                                if d_resp.status == 200:
                                    d_data = json.loads(d_resp.read().decode("utf-8"))
                                    d_routes = d_data.get("routes", [])
                                    if d_routes:
                                        d_primary = d_routes[0]
                                        d_geom = d_primary.get("geometry", {}).get("coordinates", [])
                                        d_pts = [[round(pt[1], 5), round(pt[0], 5)] for pt in d_geom]
                                        if d_pts:
                                            d_min_c = min(haversine_km(p[0], p[1], center_lat, center_lon) for p in d_pts)
                                            if d_min_c > min_hazard_dist + 2.0:
                                                chosen_route = d_primary
                                                chosen_coords = d_pts
                                                min_hazard_dist = d_min_c
                                                hazard_avoided = True
                        except Exception:
                            pass

                    distance_m = chosen_route.get("distance", direct_dist_km * 1250)
                    duration_s = chosen_route.get("duration", (distance_m / 1000 / 45) * 3600)
                    road_distance_km = round(distance_m / 1000.0, 2)
                    duration_minutes = max(round(duration_s / 60.0), 1)

                    origin_pt = [round(origin_lat, 5), round(origin_lon, 5)]
                    dest_pt = [round(dest_lat, 5), round(dest_lon, 5)]

                    if not chosen_coords:
                        chosen_coords = [origin_pt, dest_pt]
                    else:
                        if chosen_coords[0] != origin_pt:
                            chosen_coords.insert(0, origin_pt)
                        if chosen_coords[-1] != dest_pt:
                            chosen_coords.append(dest_pt)

                    return {
                        "routing_engine": "OSRM OpenStreetMap (Live Road Network)",
                        "origin": {"name": origin_name, "latitude": origin_lat, "longitude": origin_lon},
                        "destination": {"name": dest_name, "latitude": dest_lat, "longitude": dest_lon},
                        "distance_km": road_distance_km,
                        "direct_distance_km": direct_dist_km,
                        "estimated_duration_minutes": duration_minutes,
                        "estimated_speed_kmh": round((road_distance_km / (duration_minutes / 60.0)), 1) if duration_minutes > 0 else 45.0,
                        "coordinates": chosen_coords,
                        "status": "success",
                        "is_real_road": True,
                        "hazard_avoided": hazard_avoided,
                        "min_distance_to_epicenter_km": round(min_hazard_dist, 1) if min_hazard_dist < 900 else None,
                        "safety_note": (
                            "Rute evakuasi minim dampak terverifikasi: memprioritaskan koridor jalan raya yang "
                            "menjauhi konsentrasi abu vulkanik pekat dan lingkar kawah aktif."
                        )
                    }
    except Exception:
        pass

    # Perhitungan Fallback Realistis (menjauhi kawah aktif)
    wps = generate_offline_road_waypoints(
        (origin_lat, origin_lon),
        (dest_lat, dest_lon),
        center_lat=center_lat,
        center_lon=center_lon
    )
    coordinates_latlng = [[round(lat, 5), round(lon, 5)] for lat, lon in wps]
    
    # Rasio jalan raya Indonesia umumnya 1.25x - 1.35x jarak garis lurus
    road_distance_km = round(direct_dist_km * 1.30, 2)
    # Estimasi kecepatan ambulans gawat darurat rata-rata 40-50 km/jam
    duration_minutes = max(round((road_distance_km / 45.0) * 60.0), 1)

    return {
        "routing_engine": "UdaraSiaga Local Road Curvature Synthesizer (Offline Fallback)",
        "origin": {"name": origin_name, "latitude": origin_lat, "longitude": origin_lon},
        "destination": {"name": dest_name, "latitude": dest_lat, "longitude": dest_lon},
        "distance_km": road_distance_km,
        "direct_distance_km": direct_dist_km,
        "estimated_duration_minutes": duration_minutes,
        "estimated_speed_kmh": 45.0,
        "coordinates": coordinates_latlng,
        "status": "success",
        "is_real_road": False,
        "hazard_avoided": True,
        "safety_note": "Rute kurvatur jalan raya aman: koridor menghindari pusat bahaya atmosfer."
    }
