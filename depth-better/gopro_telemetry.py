import os
import re
import struct
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def extract_binary_stream(mp4_path, bin_path):
    """Uses FFmpeg to extract the GPMD telemetry track from the MP4."""
    cmd = ["ffmpeg", "-y", "-i", mp4_path, "-map", "0:m:handler_name:GoPro MET", "-f", "data", "-c", "copy", bin_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to checking streams if metadata tag fails
        cmd = ["ffmpeg", "-y", "-i", mp4_path, "-map", "0:3", "-f", "data", "-c", "copy", bin_path]
        subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(bin_path) and os.path.getsize(bin_path) > 0

def unpack_gpmf_data(type_val, size, repeat, raw_data):
    type_char = type_val.decode('ascii', errors='ignore')
    fmt_map = {'s': 'h', 'S': 'H', 'l': 'i', 'L': 'I', 'f': 'f', 'd': 'd', 'B': 'B', 'b': 'b', 'J': 'Q', 'j': 'q'}
    if type_char in fmt_map:
        py_type = fmt_map[type_char]
        vals_per_repeat = size // struct.calcsize(py_type)
        fmt = f">{vals_per_repeat * repeat}{py_type}"
        try:
            unpacked = struct.unpack(fmt, raw_data[:struct.calcsize(fmt)])
            if vals_per_repeat > 1:
                return [unpacked[i:i+vals_per_repeat] for i in range(0, len(unpacked), vals_per_repeat)]
            return list(unpacked)
        except:
            return raw_data
    elif type_char == 'c':
        return raw_data.decode('ascii', errors='ignore').strip('\x00')
    return raw_data

def parse_gpmf(data, offset=0, end=None):
    if end is None: end = len(data)
    elements = []
    while offset < end:
        if offset + 8 > end: break
        key = data[offset:offset+4].decode('ascii', errors='ignore')
        type_val = data[offset+4:offset+5]
        size, repeat = struct.unpack('>BH', data[offset+5:offset+8])
        length = size * repeat
        padded_length = (length + 3) & ~3
        value_offset = offset + 8
        if value_offset + length > end: break
        
        node = {'key': key, 'type': type_val.decode('ascii', errors='ignore'), 'size': size, 'repeat': repeat, 'raw': data[value_offset:value_offset + length]}
        if type_val == b'\x00':
            node['children'] = parse_gpmf(data, value_offset, value_offset + length)
            node['value'] = None
        else:
            node['value'] = unpack_gpmf_data(type_val, size, repeat, node['raw'])
            
        elements.append(node)
        offset += 8 + padded_length
    return elements

def extract_all_telemetry(ast):
    streams = {}
    METADATA_KEYS = {'STNM', 'SCAL', 'UNIT', 'SIUN', 'TYPE', 'MTRY', 'OUTR', 'ORIN', 'TICK', 'TSMP', 'TIMO', 'EMP'}
    for devc in ast:
        if devc['key'] == 'DEVC':
            for item in devc.get('children', []):
                if item['key'] == 'STRM':
                    strm_dict = {t['key']: t for t in item.get('children', [])}
                    scal_node = strm_dict.get('SCAL')
                    scal = scal_node['value'] if scal_node and scal_node['value'] is not None else [1]
                    if not isinstance(scal, (list, tuple)): scal = [scal]
                    
                    for d_key in [k for k in strm_dict.keys() if k not in METADATA_KEYS]:
                        d_node = strm_dict[d_key]
                        raw_samples = d_node.get('value')
                        if isinstance(raw_samples, bytes): continue
                        if not isinstance(raw_samples, list): raw_samples = [raw_samples]
                            
                        if d_key not in streams: streams[d_key] = []
                        
                        for row in raw_samples:
                            if isinstance(row, (tuple, list)):
                                scaled_row = [val / (scal[i] if i < len(scal) and scal[i] != 0 else (scal[-1] if len(scal)>0 and scal[-1]!=0 else 1)) if isinstance(val, (int, float)) else val for i, val in enumerate(row)]
                                streams[d_key].append(tuple(scaled_row))
                            elif isinstance(row, (int, float)):
                                streams[d_key].append(row / (scal[0] if len(scal)>0 and scal[0]!=0 else 1))
    return streams

def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Converts GPS WGS84 coordinates to local Cartesian (East, North, Up) in meters."""
    R = 6378137.0 # Earth radius
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = dlon * R * np.cos(np.radians(lat0))
    y = dlat * R
    z = alt - alt0
    return np.array([x, y, z])

def process_gopro_telemetry(mp4_path, num_frames):
    """
    Extracts telemetry, synchronizes it to the video frames, and calculates 
    the mathematically precise Extrinsics and Gravity vectors.
    """
    bin_path = mp4_path + ".bin"
    if not extract_binary_stream(mp4_path, bin_path):
        return None
        
    with open(bin_path, "rb") as f:
        ast = parse_gpmf(f.read())
    
    if os.path.exists(bin_path):
        os.remove(bin_path)
        
    streams = extract_all_telemetry(ast)
    
    # We need Camera Orientation (CORI), GPS (GPS5 or GPS9), and Gravity (GRAV)
    cori_data = streams.get('CORI')
    gps_data = streams.get('GPS5') or streams.get('GPS9')
    grav_data = streams.get('GRAV')
    
    if not cori_data or not gps_data or not grav_data:
        return None # Missing required telemetry for absolute poses

    # Interpolation setup (assuming uniform distribution across the extracted frames)
    # Action cameras capture at varying rates (GPS @ 18Hz, CORI @ 200Hz). 
    # We map them cleanly to the 'num_frames' using simple linear mapping.
    frame_times = np.linspace(0, 1, num_frames)
    
    # --- 1. Process GPS (Translation) ---
    gps_arr = np.array(gps_data) # [Lat, Lon, Alt, ...]
    lat0, lon0, alt0 = gps_arr[0, 0], gps_arr[0, 1], gps_arr[0, 2]
    enu_coords = np.array([wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0) for lat, lon, alt, *_ in gps_arr])
    
    gps_times = np.linspace(0, 1, len(enu_coords))
    translations = np.zeros((num_frames, 3))
    for i in range(3):
        translations[:, i] = np.interp(frame_times, gps_times, enu_coords[:, i])
        
    # --- 2. Process CORI (Rotation) ---
    # GoPro quaternions are [w, x, y, z]. Scipy needs [x, y, z, w].
    cori_arr = np.array(cori_data)
    quats_xyzw = np.column_stack([cori_arr[:, 1], cori_arr[:, 2], cori_arr[:, 3], cori_arr[:, 0]])
    
    cori_times = np.linspace(0, 1, len(quats_xyzw))
    slerp = Slerp(cori_times, R.from_quat(quats_xyzw))
    interpolated_rotations = slerp(frame_times)
    
    # --- 3. Process GRAV (Gravity Vector) ---
    grav_arr = np.array(grav_data)
    grav_times = np.linspace(0, 1, len(grav_arr))
    gravity_vectors = np.zeros((num_frames, 3))
    for i in range(3):
        gravity_vectors[:, i] = np.interp(frame_times, grav_times, grav_arr[:, i])
        
    # Normalize gravity vectors
    norms = np.linalg.norm(gravity_vectors, axis=1, keepdims=True)
    gravity_vectors = gravity_vectors / np.where(norms == 0, 1, norms)

    # --- 4. Construct Extrinsics (World to Camera) ---
    extrinsics = np.zeros((num_frames, 4, 4))
    for i in range(num_frames):
        c2w = np.eye(4)
        
        # GoPro coordinate system vs OpenCV coordinate system alignment
        R_gopro = interpolated_rotations[i].as_matrix()
        # CV convention: X right, Y down, Z forward
        # GoPro convention usually X right, Y up, Z backward (OpenGL)
        R_align = R.from_euler('x', 180, degrees=True).as_matrix()
        R_cv = R_gopro @ R_align
        
        c2w[:3, :3] = R_cv
        c2w[:3, 3] = translations[i]
        
        # Extrinsics are world-to-camera (inverse of c2w)
        w2c = np.linalg.inv(c2w)
        extrinsics[i] = w2c
        
    return {
        'extrinsics': extrinsics.astype(np.float32),
        'gravity_vectors': gravity_vectors.astype(np.float32)
    }

def get_rotation_between_vectors(v_from, v_to):
    """Calculates the rotation matrix required to align v_from to v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    s = np.linalg.norm(cross)
    
    if s == 0:
        if dot > 0: return np.eye(3)
        else: return -np.eye(3)
        
    skew_sym = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    
    R = np.eye(3) + skew_sym + np.dot(skew_sym, skew_sym) * ((1 - dot) / (s ** 2))
    return R
