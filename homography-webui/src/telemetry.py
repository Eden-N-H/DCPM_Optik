import math
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
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
            
            dop, fix = 1.0, 3.0
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

    # CRITICAL: GPMF packet streams can sometimes append slightly out-of-order chunks. 
    # Sorting ensures absolute monotonic time, preventing math errors during differentiation/smoothing.
    valid_gps = sorted(valid_gps, key=lambda x: x["time_sec"])
    valid_gps = _filter_gps_speed_outliers(valid_gps)

    if valid_gps:
        times = np.array([s["time_sec"] for s in valid_gps])
        data = np.array([[s["lat"], s["lon"], s["speed"]] for s in valid_gps])
        
        duration = times[-1] - times[0]
        hz = len(data) / duration if duration > 0.1 else 18.0
        shared_avg_speed = float(np.mean(data[:, 2])) if len(data) > 0 else 5.0
        safe_speed = max(shared_avg_speed, 1.0)
        
        # 1. Project global lat/lon to a localized Cartesian flat-plane (Metric X/Y).
        # This completely avoids floating point errors when computing tangents on micro-movements.
        R_earth = 6378137.0
        lat0_rad = math.radians(data[0, 0])
        
        lats_rad = np.radians(data[:, 0])
        lons_rad = np.radians(data[:, 1])
        
        # Y is North (meters), X is East (meters) relative to the first GPS point.
        y_m = (lats_rad - lats_rad[0]) * R_earth
        x_m = (lons_rad - lons_rad[0]) * R_earth * math.cos(lat0_rad)
        
        # 2. Aggressive Spatial Smoothing.
        # Gaussian smoothing completely eliminates the GPS "zigzag" scatter without introducing 
        # the mathematical overshoot/ringing that polynomial filters cause around sharp corners.
        sigma_sec = 2.0 / safe_speed
        sigma_samples = max(1.0, sigma_sec * hz)
        
        x_smooth = gaussian_filter1d(x_m, sigma=sigma_samples, mode='nearest')
        y_smooth = gaussian_filter1d(y_m, sigma=sigma_samples, mode='nearest')
        speed_smooth = gaussian_filter1d(data[:, 2], sigma=sigma_samples, mode='nearest')
        
        # Convert the smoothed metric coordinates cleanly back to lat/lon for the UI/Map
        smoothed_lat = np.degrees(y_smooth / R_earth) + data[0, 0]
        smoothed_lon = np.degrees(x_smooth / (R_earth * math.cos(lat0_rad))) + data[0, 1]
        
        # 3. Analytic Heading via Calculus Derivatives (dx/dt, dy/dt)
        # Taking the absolute mathematical tangent of the smoothed path perfectly isolates
        # the car's heading, entirely preventing the zigzag/wobble seen in the map footprint.
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        
        # atan2(dx, dy) where X is East and Y is North yields exact map bearing (0=North, 90=East)
        headings_rad = np.arctan2(dx, dy)
        continuous_headings_deg = (np.degrees(headings_rad) + 360) % 360
        
        # 4. Construct Interpolators
        if len(times) >= 4:
            interpolators["gps"] = PchipInterpolator(times, np.column_stack((smoothed_lat, smoothed_lon)), extrapolate=True)
            interpolators["speed"] = PchipInterpolator(times, speed_smooth, extrapolate=True)
            
            # Unwrap before PCHIP so the spline doesn't violently swing 360 degrees if a corner crosses North
            headings_unwrapped = np.unwrap(np.radians(continuous_headings_deg))
            raw_heading_spline = PchipInterpolator(times, headings_unwrapped, extrapolate=True)
            
            # Wrap the spline evaluator so standard calls still return 0-360 degrees
            def heading_evaluator(t):
                return (math.degrees(float(raw_heading_spline(t))) + 360) % 360
            interpolators["heading"] = heading_evaluator
            
        else:
            interpolators["gps"] = interp1d(times, np.column_stack((smoothed_lat, smoothed_lon)), axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolators["speed"] = interp1d(times, speed_smooth, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolators["heading"] = interp1d(times, continuous_headings_deg, kind='linear', bounds_error=False, fill_value="extrapolate")

    # IMU Gravity handling
    if "GRAV" in streams:
        times = np.array([s["time_sec"] for s in streams["GRAV"]])
        gravs = np.array([s["data"] for s in streams["GRAV"]])
        
        duration = times[-1] - times[0] if len(times) > 0 else 0
        hz = len(gravs) / duration if duration > 0.1 else 100.0
        
        # Smooth gravity to kill engine vibration, but keep it responsive
        sigma_sec = 0.5
        sigma_samples = max(1.0, sigma_sec * hz)
        
        gravs[:,0] = gaussian_filter1d(gravs[:,0], sigma=sigma_samples, mode='nearest')
        gravs[:,1] = gaussian_filter1d(gravs[:,1], sigma=sigma_samples, mode='nearest')
        gravs[:,2] = gaussian_filter1d(gravs[:,2], sigma=sigma_samples, mode='nearest')
        
        if len(times) >= 4:
            interpolators["grav_x"] = PchipInterpolator(times, gravs[:,0], extrapolate=True)
            interpolators["grav_y"] = PchipInterpolator(times, gravs[:,1], extrapolate=True)
            interpolators["grav_z"] = PchipInterpolator(times, gravs[:,2], extrapolate=True)
        else:
            interpolators["grav_x"] = interp1d(times, gravs[:,0], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolators["grav_y"] = interp1d(times, gravs[:,1], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolators["grav_z"] = interp1d(times, gravs[:,2], kind='linear', bounds_error=False, fill_value="extrapolate")

    return interpolators
