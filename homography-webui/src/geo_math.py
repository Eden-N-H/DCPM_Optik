import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378137.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def local_to_global(lat, lon, heading_deg, local_x, local_z):
    R = 6378137.0
    d = math.hypot(local_x, local_z)
    true_heading_rad = math.radians(heading_deg) + math.atan2(local_x, local_z)
    
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    
    out_lat = math.asin(math.sin(lat_rad)*math.cos(d/R) + math.cos(lat_rad)*math.sin(d/R)*math.cos(true_heading_rad))
    out_lon = lon_rad + math.atan2(math.sin(true_heading_rad)*math.sin(d/R)*math.cos(lat_rad), math.cos(d/R) - math.sin(lat_rad)*math.sin(out_lat))
    return math.degrees(out_lat), math.degrees(out_lon)

def global_to_local(base_lat, base_lon, base_heading, target_lat, target_lon):
    """
    Inverse of local_to_global: given a target lat/lon, find its (x, z) 
    coordinates relative to base_lat/base_lon aligned with base_heading.
    x is lateral (right), z is longitudinal (forward).
    """
    dist = haversine_distance(base_lat, base_lon, target_lat, target_lon)
    bearing = calculate_bearing(base_lat, base_lon, target_lat, target_lon)
    angle = math.radians(bearing - base_heading)
    x = dist * math.sin(angle)
    z = dist * math.cos(angle)
    return x, z

def apply_camera_offset(lat, lon, heading_deg, offset_right_m=0.0, offset_forward_m=0.0):
    if offset_right_m == 0.0 and offset_forward_m == 0.0:
        return lat, lon
    return local_to_global(lat, lon, heading_deg, offset_right_m, offset_forward_m)

def decompose_offset(lat1, lon1, lat2, lon2, heading_deg):
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    if dist < 1e-9:
        return 0.0, 0.0
    bearing_to_2 = calculate_bearing(lat1, lon1, lat2, lon2)
    angle_diff = math.radians(bearing_to_2 - heading_deg)
    longitudinal = dist * math.cos(angle_diff)
    lateral = dist * math.sin(angle_diff)
    return longitudinal, lateral
