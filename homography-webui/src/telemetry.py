import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from geo_math import haversine_distance, calculate_bearing

def evaluate_telemetry_health(streams):
    report = {
        "gps_score": 100.0,
        "imu_score": 100.0,
        "warnings": [],
        "metrics": {
            "avg_gps_speed_error_ms": 0.0,
            "bad_fix_ratio": 0.0,
            "avg_grav_mag_error": 0.0,
            "max_speed_error": 0.0
        }
    }
    
    gps_key = "GPS9" if "GPS9" in streams else ("GPS5" if "GPS5" in streams else None)
    
    if gps_key and len(streams[gps_key]) > 1:
        gps_data = streams[gps_key]
        speed_errors = []
        bad_dop_count = 0
        bad_fix_count = 0
        
        for i in range(1, len(gps_data)):
            prev = gps_data[i-1]
            curr = gps_data[i]
            
            dt = curr["time_sec"] - prev["time_sec"]
            if dt <= 0: continue
            
            lat1, lon1 = prev["data"][0], prev["data"][1]
            lat2, lon2 = curr["data"][0], curr["data"][1]
            doppler_speed_curr = curr["data"][3]
            
            dop, fix = 1.0, 3.0 # Safest defaults for older GPS5
            if gps_key == "GPS9" and len(curr["data"]) >= 9:
                dop = curr["data"][7]
                fix = curr["data"][8]
            
            if fix < 3: bad_fix_count += 1
            if dop > 3.0: bad_dop_count += 1
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            derived_speed = dist / dt
            
            if doppler_speed_curr > 1.0 or derived_speed > 1.0:
                speed_errors.append(abs(derived_speed - doppler_speed_curr))
        
        avg_speed_error = sum(speed_errors) / len(speed_errors) if speed_errors else 0
        max_speed_error = max(speed_errors) if speed_errors else 0.0
        
        report["metrics"]["avg_gps_speed_error_ms"] = round(avg_speed_error, 3)
        report["metrics"]["bad_fix_ratio"] = round(bad_fix_count / len(gps_data), 3)
        report["metrics"]["max_speed_error"] = round(max_speed_error, 2)
        
        gps_penalty = (avg_speed_error * 5) + ((bad_fix_count / len(gps_data)) * 40) + ((bad_dop_count / len(gps_data)) * 15)
        if max_speed_error > 15.0: gps_penalty += 15  
            
        report["gps_score"] = max(0.0, min(100.0, 100.0 - gps_penalty))
        
        if max_speed_error > 15.0:
            report["warnings"].append(f"Kinematic Anomaly: GPS coordinate jumped violently (Speed delta: {max_speed_error:.1f} m/s).")
        if report["gps_score"] < 80:
            report["warnings"].append(f"GPS Quality Degraded (Score: {report['gps_score']:.1f}%). High spatial drift.")
            
    if "GRAV" in streams and len(streams["GRAV"]) > 0:
        grav_data = streams["GRAV"]
        mag_errors = []
        
        for item in grav_data:
            x, y, z = item["data"]
            mag = math.sqrt(x*x + y*y + z*z)
            mag_errors.append(abs(1.0 - mag))
            
        avg_mag_error = sum(mag_errors) / len(mag_errors) if mag_errors else 0
        report["metrics"]["avg_grav_mag_error"] = round(avg_mag_error, 4)
        
        imu_penalty = avg_mag_error * 200 
        report["imu_score"] = max(0.0, min(100.0, 100.0 - imu_penalty))
        
        if report["imu_score"] < 90:
            report["warnings"].append(f"IMU Calibration/Vibration Issue (Score: {report['imu_score']:.1f}%). Gravity vector unstable.")
            
    return report

def _filter_gps_speed_outliers(valid_gps, max_speed_error_ms=8.0):
    if len(valid_gps) < 3:
        return valid_gps
    filtered = [valid_gps[0]]
    for curr in valid_gps[1:]:
        prev = filtered[-1]
        dt = curr["time_sec"] - prev["time_sec"]
        if dt <= 0:
            continue
        dist = haversine_distance(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        derived_speed = dist / dt
        if derived_speed > 1.0 and abs(derived_speed - curr["speed"]) > max_speed_error_ms:
            continue
        filtered.append(curr)
    return filtered if len(filtered) >= 2 else valid_gps

def _distance_smoothing_window(n_samples, duration_sec, avg_speed_mps, target_meters=1.5, min_w=5, polyorder=3):
    if duration_sec <= 0 or n_samples < min_w + 2:
        return None
    hz = n_samples / duration_sec
    safe_speed = max(float(avg_speed_mps), 0.5)
    duration_for_target = target_meters / safe_speed
    w = int(round(hz * duration_for_target))
    if w % 2 == 0:
        w += 1
    max_w = n_samples if n_samples % 2 != 0 else n_samples - 1
    w = max(min_w, min(w, max_w))
    if w <= polyorder:
        return None
    return w

def get_telemetry_interpolators(streams):
    interpolators = {}
    valid_gps = []
    shared_avg_speed = None

    if "GPS9" in streams:
        for s in streams["GPS9"]:
            if len(s["data"]) >= 9:
                lat, lon, alt, s2d, s3d, days, secs, dop, fix = s["data"]
                if fix >= 2 and dop <= 5.0:
                    valid_gps.append({"time_sec": s["time_sec"], "lat": lat, "lon": lon, "speed": s2d})
    elif "GPS5" in streams:
        for s in streams["GPS5"]:
            if len(s["data"]) >= 5:
                lat, lon, alt, s2d, s3d = s["data"]
                if lat != 0.0 and lon != 0.0:
                    valid_gps.append({"time_sec": s["time_sec"], "lat": lat, "lon": lon, "speed": s2d})

    valid_gps = _filter_gps_speed_outliers(valid_gps)

    if valid_gps:
        times = np.array([s["time_sec"] for s in valid_gps])
        data = np.array([[s["lat"], s["lon"], s["speed"]] for s in valid_gps])
        
        duration = times[-1] - times[0]
        shared_avg_speed = float(np.mean(data[:, 2])) if len(data) > 0 else None
        
        w_speed = _distance_smoothing_window(len(data), duration, shared_avg_speed or 5.0, target_meters=2.0)
        w_heading = _distance_smoothing_window(len(data), duration, shared_avg_speed or 5.0, target_meters=10.0, min_w=11)
        w_pos = _distance_smoothing_window(len(data), duration, shared_avg_speed or 5.0, target_meters=5.0, min_w=5)
        
        if w_pos is not None:
            data[:, 0] = savgol_filter(data[:, 0], w_pos, 3)
            data[:, 1] = savgol_filter(data[:, 1], w_pos, 3)
            
        if w_speed is not None:
            data[:, 2] = savgol_filter(data[:, 2], w_speed, 3)
            
        headings = np.zeros(len(data))
        seg_times = times.astype(np.float64).copy()
        for i in range(len(data) - 1):
            headings[i] = calculate_bearing(data[i, 0], data[i, 1], data[i + 1, 0], data[i + 1, 1])
            seg_times[i] = (times[i] + times[i + 1]) / 2.0
        headings[-1] = headings[-2] if len(headings) > 1 else 0.0
        seg_times[-1] = times[-1]
        
        headings_rad = np.radians(headings)
        unwrapped_rad = np.unwrap(headings_rad)
        
        if w_heading is not None:
            unwrapped_rad = savgol_filter(unwrapped_rad, w_heading, 3)
            
        continuous_headings_deg = np.degrees(unwrapped_rad)
        
        # Use cubic spline interpolation rather than linear to eliminate piecewise 
        # angular trajectories (stair-stepping) between discrete GPS points.
        kind_gps = 'cubic' if len(times) >= 4 else 'linear'
        interpolators["gps"] = interp1d(times, data[:, :2], axis=0, kind=kind_gps, bounds_error=False, fill_value="extrapolate")
        interpolators["speed"] = interp1d(times, data[:, 2], kind=kind_gps, bounds_error=False, fill_value="extrapolate")
        
        kind_h = 'cubic' if len(seg_times) >= 4 else 'linear'
        interpolators["heading"] = interp1d(seg_times, continuous_headings_deg, kind=kind_h, bounds_error=False, fill_value="extrapolate")

    if "GRAV" in streams:
        times = np.array([s["time_sec"] for s in streams["GRAV"]])
        gravs = np.array([s["data"] for s in streams["GRAV"]])
        
        # Kinematic Compensation: Remove centrifugal and longitudinal acceleration forces 
        # that masquerade as false camera roll/pitch during cornering and braking.
        if valid_gps and "heading" in interpolators and "speed" in interpolators:
            dt = 0.25
            for i in range(len(times)):
                t = times[i]
                v = float(interpolators["speed"](t))
                if v > 0.5:
                    h1 = float(interpolators["heading"](t - dt))
                    h2 = float(interpolators["heading"](t + dt))
                    dh = (h2 - h1 + 180) % 360 - 180
                    yaw_rate_rad = math.radians(dh) / (2 * dt)
                    
                    # Centrifugal acceleration (lateral)
                    ac_x = (v * yaw_rate_rad) / 9.81
                    gravs[i, 0] += ac_x
                    
                    # Longitudinal acceleration (forward/backward squat)
                    v1 = float(interpolators["speed"](t - dt))
                    v2 = float(interpolators["speed"](t + dt))
                    a_lon = (v2 - v1) / (2 * dt)
                    ac_z = a_lon / 9.81
                    gravs[i, 2] += ac_z
                    
        duration = times[-1] - times[0] if len(times) > 0 else 0
        avg_speed_for_grav = shared_avg_speed if shared_avg_speed is not None else 5.0
        w = _distance_smoothing_window(len(gravs), duration, avg_speed_for_grav, target_meters=2.0, min_w=11)
        
        if w is not None:
            gravs[:,0] = savgol_filter(gravs[:,0], w, 3)
            gravs[:,1] = savgol_filter(gravs[:,1], w, 3)
            gravs[:,2] = savgol_filter(gravs[:,2], w, 3)
            
        kind_g = 'cubic' if len(times) >= 4 else 'linear'
        interpolators["grav_x"] = interp1d(times, gravs[:,0], kind=kind_g, bounds_error=False, fill_value="extrapolate")
        interpolators["grav_y"] = interp1d(times, gravs[:,1], kind=kind_g, bounds_error=False, fill_value="extrapolate")
        interpolators["grav_z"] = interp1d(times, gravs[:,2], kind=kind_g, bounds_error=False, fill_value="extrapolate")

    return interpolators
