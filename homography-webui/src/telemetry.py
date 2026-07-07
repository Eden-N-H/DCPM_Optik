import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from geo_math import haversine_distance

def evaluate_telemetry_health(streams):
    report = {
        "gps_score": 100.0,
        "imu_score": 100.0,
        "warnings": [],
        "metrics": {
            "avg_gps_speed_error_ms": 0.0,
            "bad_fix_ratio": 0.0,
            "avg_grav_mag_error": 0.0,
            "max_jerk_detected": 0.0
        }
    }
    
    gps_key = "GPS9" if "GPS9" in streams else ("GPS5" if "GPS5" in streams else None)
    
    if gps_key and len(streams[gps_key]) > 1:
        gps_data = streams[gps_key]
        speed_errors = []
        jerks = []
        bad_dop_count = 0
        bad_fix_count = 0
        
        for i in range(1, len(gps_data)):
            prev = gps_data[i-1]
            curr = gps_data[i]
            
            dt = curr["time_sec"] - prev["time_sec"]
            if dt <= 0: continue
            
            lat1, lon1 = prev["data"][0], prev["data"][1]
            lat2, lon2 = curr["data"][0], curr["data"][1]
            doppler_speed_prev = prev["data"][3]
            doppler_speed_curr = curr["data"][3]
            
            dop, fix = 1.0, 3.0 # Safest defaults for older GPS5
            if gps_key == "GPS9" and len(curr["data"]) >= 9:
                dop = curr["data"][7]
                fix = curr["data"][8]
            
            if fix < 3: bad_fix_count += 1
            if dop > 3.0: bad_dop_count += 1
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            derived_speed = dist / dt
            
            accel = abs(doppler_speed_curr - doppler_speed_prev) / dt
            jerks.append(accel / dt)
            
            if doppler_speed_curr > 1.0 or derived_speed > 1.0:
                speed_errors.append(abs(derived_speed - doppler_speed_curr))
        
        avg_speed_error = sum(speed_errors) / len(speed_errors) if speed_errors else 0
        max_jerk = max(jerks) if jerks else 0
        
        report["metrics"]["avg_gps_speed_error_ms"] = round(avg_speed_error, 3)
        report["metrics"]["bad_fix_ratio"] = round(bad_fix_count / len(gps_data), 3)
        report["metrics"]["max_jerk_detected"] = round(max_jerk, 2)
        
        gps_penalty = (avg_speed_error * 5) + ((bad_fix_count / len(gps_data)) * 40) + ((bad_dop_count / len(gps_data)) * 15)
        if max_jerk > 20.0: gps_penalty += 15  
            
        report["gps_score"] = max(0.0, min(100.0, 100.0 - gps_penalty))
        
        if max_jerk > 20.0:
            report["warnings"].append(f"Kinematic Anomaly: GPS coordinate jumped violently (Jerk: {max_jerk:.1f} m/s³).")
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

def get_telemetry_interpolators(streams):
    interpolators = {}
    valid_gps = []
    
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
                    
    if valid_gps:
        times = np.array([s["time_sec"] for s in valid_gps])
        data = np.array([[s["lat"], s["lon"], s["speed"]] for s in valid_gps])
        
        if len(data) > 11:
            w = min(31, len(data) if len(data)%2!=0 else len(data)-1)
            data[:,0] = savgol_filter(data[:,0], w, 3)
            data[:,1] = savgol_filter(data[:,1], w, 3)
        
        interpolators["gps"] = interp1d(times, data[:, :2], axis=0, bounds_error=False, fill_value="extrapolate")
        interpolators["speed"] = interp1d(times, data[:, 2], bounds_error=False, fill_value="extrapolate")

    # --- NEW VECTOR-BASED HOMOGRAPHY (From Tester) ---
    # We now interpolate X, Y, Z directly instead of Pitch and Roll
    if "GRAV" in streams:
        times = np.array([s["time_sec"] for s in streams["GRAV"]])
        gravs = np.array([s["data"] for s in streams["GRAV"]]) # Shape: Nx3
        
        if len(gravs) > 11:
            w = min(31, len(gravs) if len(gravs)%2!=0 else len(gravs)-1)
            gravs[:,0] = savgol_filter(gravs[:,0], w, 3)
            gravs[:,1] = savgol_filter(gravs[:,1], w, 3)
            gravs[:,2] = savgol_filter(gravs[:,2], w, 3)
            
        interpolators["grav_x"] = interp1d(times, gravs[:,0], bounds_error=False, fill_value="extrapolate")
        interpolators["grav_y"] = interp1d(times, gravs[:,1], bounds_error=False, fill_value="extrapolate")
        interpolators["grav_z"] = interp1d(times, gravs[:,2], bounds_error=False, fill_value="extrapolate")

    return interpolators