"""
Repeat-pass drift diagnostics and trajectory alignment.

Given a completed project's frame results (each carrying a lat/lon and a
filename that has a matching process_meta_{filename}.json trace on disk),
this automatically finds pairs of frames that are spatially close but far
apart in the capture sequence -- i.e. the vehicle passed the same physical
spot on a different loop/pass of the same road -- and reports how far apart
the reported world coordinates are, split into components relative to
heading so the lateral-vs-longitudinal-vs-unstructured nature of the drift
is visible directly in the UI rather than requiring a manual offline script.

This module also includes an Iterative Closest Point (ICP) global trajectory
alignment function to automatically pull multi-pass frames towards each other 
and mathematically cancel out static GPS session bias over the whole project.
"""
import os
import json
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from geo_math import haversine_distance, decompose_offset


def _load_telemetry(upload_folder, filename):
    meta_path = os.path.join(upload_folder, f"process_meta_{filename}.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f).get('telemetry', {})
    except Exception:
        return None


def find_repeat_pass_pairs(results, min_index_gap=15, max_dist_m=4.0):
    """
    Greedy nearest-neighbour loop-closure matching.

    A minimum SEQUENCE index gap is required so that frames within the same
    continuous pass (naturally close together, since they're extracted
    every interval_m along the same trajectory) are never matched to each
    other -- only genuinely separate passes over the same spot, which is
    the specific failure mode being diagnosed, get paired.
    """
    pts = [(idx, r) for idx, r in enumerate(results) if r.get('lat') is not None and r.get('lon') is not None]

    candidates = []
    n = len(pts)
    for a in range(n):
        idx_a, ra = pts[a]
        for b in range(a + 1, n):
            idx_b, rb = pts[b]
            if abs(idx_b - idx_a) < min_index_gap:
                continue
            dist = haversine_distance(ra['lat'], ra['lon'], rb['lat'], rb['lon'])
            if dist <= max_dist_m:
                candidates.append((dist, idx_a, idx_b))

    # Closest matches win first, so one frame doesn't get claimed by a
    # farther candidate before a nearer, more likely-correct match is seen.
    candidates.sort(key=lambda c: c[0])
    used = set()
    pairs = []
    for dist, idx_a, idx_b in candidates:
        if idx_a in used or idx_b in used:
            continue
        used.add(idx_a)
        used.add(idx_b)
        pairs.append((idx_a, idx_b, dist))

    pairs.sort(key=lambda p: p[0])
    return pairs


def build_pass_diagnostic_report(results, upload_folder, min_index_gap=15, max_dist_m=4.0):
    pairs = find_repeat_pass_pairs(results, min_index_gap, max_dist_m)

    pair_reports = []
    for idx_a, idx_b, dist in pairs:
        ra, rb = results[idx_a], results[idx_b]
        ta = _load_telemetry(upload_folder, ra['filename'])
        tb = _load_telemetry(upload_folder, rb['filename'])
        if ta is None or tb is None:
            continue

        raw_a_lat, raw_a_lon = ta.get('raw_lat', ta.get('lat')), ta.get('raw_lon', ta.get('lon'))
        raw_b_lat, raw_b_lon = tb.get('raw_lat', tb.get('lat')), tb.get('raw_lon', tb.get('lon'))
        heading_a = ta.get('heading', 0.0) or 0.0
        heading_b = tb.get('heading', 0.0) or 0.0

        if None in (raw_a_lat, raw_a_lon, raw_b_lat, raw_b_lon):
            continue

        raw_long, raw_lat_off = decompose_offset(raw_a_lat, raw_a_lon, raw_b_lat, raw_b_lon, heading_a)
        corr_long, corr_lat_off = decompose_offset(ra['lat'], ra['lon'], rb['lat'], rb['lon'], heading_a)

        speed_a = ta.get('speed_ms')
        speed_b = tb.get('speed_ms')

        pair_reports.append({
            "frame_a": ra['original_name'], "frame_b": rb['original_name'],
            "filename_a": ra['filename'], "filename_b": rb['filename'],
            "lat_a": ra['lat'], "lon_a": ra['lon'],
            "lat_b": rb['lat'], "lon_b": rb['lon'],
            "distance_m": round(dist, 3),
            "raw_longitudinal_m": round(raw_long, 3),
            "raw_lateral_m": round(raw_lat_off, 3),
            "corrected_longitudinal_m": round(corr_long, 3),
            "corrected_lateral_m": round(corr_lat_off, 3),
            "heading_a": round(heading_a % 360, 1),
            "heading_b": round(heading_b % 360, 1),
            "delta_heading": round((heading_b - heading_a + 180) % 360 - 180, 1),
            "speed_a_ms": round(speed_a, 2) if speed_a is not None else None,
            "speed_b_ms": round(speed_b, 2) if speed_b is not None else None
        })

    if pair_reports:
        lat_vals = [abs(p['corrected_lateral_m']) for p in pair_reports]
        long_vals = [abs(p['corrected_longitudinal_m']) for p in pair_reports]
        head_deltas = [abs(p['delta_heading']) for p in pair_reports]
        summary = {
            "pair_count": len(pair_reports),
            "mean_lateral_m": round(sum(lat_vals) / len(lat_vals), 3),
            "max_lateral_m": round(max(lat_vals), 3),
            "mean_longitudinal_m": round(sum(long_vals) / len(long_vals), 3),
            "max_longitudinal_m": round(max(long_vals), 3),
            "mean_delta_heading": round(sum(head_deltas) / len(head_deltas), 2),
            "max_delta_heading": round(max(head_deltas), 2)
        }
    else:
        summary = {"pair_count": 0}

    return {"pairs": pair_reports, "summary": summary}


def align_project(results, min_index_gap=15, max_dist_m=4.0):
    """
    Iterative Closest Point (ICP) global trajectory alignment.
    Smooths out run-to-run static GPS session biases by mathematically pulling 
    repeat-pass frames towards earlier frames that hit the same physical road 
    stretch. A continuous offset is interpolated across the entire project length 
    so even stretches without overlapping passes get safely translated.
    """
    valid_indices = [i for i, r in enumerate(results) if r.get('lat') is not None and r.get('lon') is not None]
    if len(valid_indices) < 20: 
        return {"error": "Not enough GPS data points to perform trajectory alignment."}
    
    origin_lat = results[valid_indices[0]]['lat']
    origin_lon = results[valid_indices[0]]['lon']
    R_earth = 6378137.0
    cos_lat = math.cos(math.radians(origin_lat))
    
    def latlon_to_xy(lat, lon):
        x = math.radians(lon - origin_lon) * cos_lat * R_earth
        y = math.radians(lat - origin_lat) * R_earth
        return np.array([x, y])
        
    def xy_to_latlon(x, y):
        lat = origin_lat + math.degrees(y / R_earth)
        lon = origin_lon + math.degrees(x / (R_earth * cos_lat))
        return lat, lon

    coords = np.zeros((len(results), 2))
    for i in valid_indices:
        coords[i] = latlon_to_xy(results[i]['lat'], results[i]['lon'])
        
    current_coords = coords.copy()
    
    # 5 iterations of ICP is usually enough to fully align standard repeat-pass drifts
    for iteration in range(5):
        tree = cKDTree(current_coords[valid_indices])
        
        pull_vectors = np.zeros((len(results), 2))
        pull_weights = np.zeros(len(results))
        
        for idx in valid_indices:
            indices_in_subset = tree.query_ball_point(current_coords[idx], r=max_dist_m)
            valid_targets = [valid_indices[j] for j in indices_in_subset if valid_indices[j] < idx - min_index_gap]
            
            if not valid_targets: 
                continue
            
            target_idx = min(valid_targets, key=lambda j: np.linalg.norm(current_coords[j] - current_coords[idx]))
            dist = np.linalg.norm(current_coords[target_idx] - current_coords[idx])
            
            weight = max(0.1, 1.0 - (dist / max_dist_m))
            pull = current_coords[target_idx] - current_coords[idx]
            
            pull_vectors[idx] += pull * weight
            pull_weights[idx] += weight
            
        mask = pull_weights > 0
        if not np.any(mask):
            break
            
        pull_vectors[mask] /= pull_weights[mask][:, None]
        
        indices_with_pull = np.where(mask)[0]
        if len(indices_with_pull) > 1:
            interp_x = interp1d(indices_with_pull, pull_vectors[indices_with_pull, 0], bounds_error=False, fill_value=(0.0, 0.0))
            interp_y = interp1d(indices_with_pull, pull_vectors[indices_with_pull, 1], bounds_error=False, fill_value=(0.0, 0.0))
            
            drift_correction = np.zeros((len(results), 2))
            drift_correction[:, 0] = interp_x(np.arange(len(results)))
            drift_correction[:, 1] = interp_y(np.arange(len(results)))
            
            # Heavy low-pass filtering to ensure the alignment vector varies continuously 
            # and gracefully across the entire sequence.
            drift_correction[:, 0] = gaussian_filter1d(drift_correction[:, 0], sigma=15.0)
            drift_correction[:, 1] = gaussian_filter1d(drift_correction[:, 1], sigma=15.0)
            
            current_coords += drift_correction
        else:
            break

    for i in valid_indices:
        new_lat, new_lon = xy_to_latlon(current_coords[i, 0], current_coords[i, 1])
        results[i]['lat'] = new_lat
        results[i]['lon'] = new_lon
        
    return {"success": True, "results": results}
