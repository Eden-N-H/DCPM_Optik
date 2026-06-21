import os
import re
import struct
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def extract_jpeg_metadata_blocks(file_path):
    if not os.path.exists(file_path): return None, None
    xmp_data, gpmf_data = None, None
    with open(file_path, 'rb') as f:
        if f.read(2) != b'\xff\xd8': return None, None
        while True:
            marker_header = f.read(2)
            if len(marker_header) < 2 or marker_header[0] != 0xff: break
            marker_type = marker_header[1]
            if marker_type in (0xd9, 0xda): break
            length_bytes = f.read(2)
            if len(length_bytes) < 2: break
            length = struct.unpack('>H', length_bytes)[0]
            payload = f.read(length - 2)
            
            if marker_type == 0xe1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
                xmp_data = payload[29:].decode('utf-8', errors='ignore')
            elif marker_type == 0xe6 and payload.startswith(b'GoPro\x00'):
                gpmf_data = payload[6:]
                
    return xmp_data, gpmf_data

def parse_xmp_gpano(xmp_string):
    gpano_data = {}
    if not xmp_string: return gpano_data
    attr_matches = re.findall(r'GPano:(\w+)=["\']([^"\']+)["\']', xmp_string)
    for key, val in attr_matches: gpano_data[key] = val
    elem_matches = re.findall(r'<GPano:(\w+)>([^<]+)</GPano:\1>', xmp_string)
    for key, val in elem_matches: gpano_data[key] = val
    return gpano_data

def unpack_gpmf_data(type_val, size, repeat, raw_data):
    type_char = type_val.decode('ascii', errors='ignore')
    fmt_map = {'s': 'h', 'S': 'H', 'l': 'i', 'L': 'I', 'f': 'f', 'd': 'd', 'B': 'B', 'b': 'b', 'J': 'Q', 'j': 'q'}
    if type_char in fmt_map:
        py_type = fmt_map[type_char]
        vals_per_repeat = size // struct.calcsize(py_type)
        fmt = f">{vals_per_repeat * repeat}{py_type}"
        try:
            unpacked = struct.unpack(fmt, raw_data[:struct.calcsize(fmt)])
            if vals_per_repeat > 1: return [list(unpacked[i:i+vals_per_repeat]) for i in range(0, len(unpacked), vals_per_repeat)]
            return list(unpacked)
        except Exception: return raw_data
    elif type_char == 'c': return raw_data.decode('ascii', errors='ignore').strip('\x00')
    elif type_char in ['F', 'U']:
        results = [(raw_data[i*size : (i+1)*size]).decode('ascii', errors='ignore').strip('\x00') for i in range(repeat)]
        return results if repeat > 1 else results[0]
    return raw_data 

def unpack_complex_gpmf(type_str, size, repeat, raw_data):
    matches = re.findall(r'(\[\d+\])?([a-zA-Z])', type_str)
    parsed = [(int(b[1:-1]) if b else 1, c) for b, c in matches]
    fmt_map = {'b': ('b',1), 'B': ('B',1), 'c': ('s',1), 'd': ('d',8), 'f': ('f',4), 'l': ('i',4), 'L': ('I',4), 's': ('h',2), 'S': ('H',2)}
    fmt, expected_size, flat_chars = '>', 0, []
    
    for count, char in parsed:
        if char not in fmt_map: return None
        py_fmt, bytes_per_item = fmt_map[char]
        if py_fmt.endswith('s'):
            fmt += f"{count * (int(py_fmt[:-1]) if len(py_fmt)>1 else 1)}s"
            expected_size += count * (int(py_fmt[:-1]) if len(py_fmt)>1 else 1)
            flat_chars.append(char)
        else:
            fmt += f"{count}{py_fmt}"
            expected_size += count * bytes_per_item
            flat_chars.extend([char] * count)
            
    if expected_size == 0 or expected_size > size: return None
    results = []
    try:
        for i in range(repeat):
            chunk = raw_data[i*size : (i+1)*size]
            if len(chunk) < expected_size: break
            unpacked = struct.unpack(fmt, chunk[:expected_size])
            cleaned = [val.decode('ascii', errors='ignore').strip('\x00') if char in ('F','U','c') else val for val, char in zip(unpacked, flat_chars)]
            results.append(cleaned[0] if len(cleaned) == 1 else list(cleaned))
        return results
    except Exception: return None

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
        else: node['value'] = unpack_gpmf_data(type_val, size, repeat, node['raw'])
            
        elements.append(node)
        offset += 8 + padded_length
    return elements

def flatten_global_ast(ast):
    out = {}
    for node in ast:
        if node.get('children'): out.update(flatten_global_ast(node['children']))
        else:
            val = node.get('value')
            if val is None: val = node.get('raw')
            if isinstance(val, list) and len(val) == 1: val = val[0]
            out[node['key']] = val
    return out

def extract_all_telemetry(ast):
    constants, streams = {}, {}
    METADATA_KEYS = {'STNM', 'SCAL', 'UNIT', 'SIUN', 'TYPE', 'MTRY', 'OUTR', 'ORIN', 'TICK', 'TSMP', 'TIMO', 'EMP'}

    for devc in ast:
        if devc['key'] == 'DEVC':
            for item in devc.get('children', []):
                if item['key'] != 'STRM':
                    val = item.get('value') or item.get('raw')
                    constants[item['key']] = val[0] if isinstance(val, list) and len(val) == 1 else val
                else:
                    strm_dict = {t['key']: t for t in item.get('children', [])}
                    scal = strm_dict.get('SCAL', {}).get('value', [1])
                    if not isinstance(scal, (list, tuple)): scal = [scal]
                    
                    type_str = "".join(strm_dict.get('TYPE', {}).get('value', [])) if isinstance(strm_dict.get('TYPE', {}).get('value', []), list) else strm_dict.get('TYPE', {}).get('value', '')
                    for d_key in [k for k in strm_dict.keys() if k not in METADATA_KEYS]:
                        d_node = strm_dict[d_key]
                        raw_samples = d_node.get('value', [])
                        
                        if isinstance(raw_samples, bytes) and type_str and d_node['repeat'] > 0:
                            complex_res = unpack_complex_gpmf(type_str, d_node['size'], d_node['repeat'], d_node['raw'])
                            if complex_res is not None: raw_samples = complex_res
                            
                        if isinstance(raw_samples, bytes): raw_samples = []
                        if not isinstance(raw_samples, list): raw_samples = [raw_samples]
                            
                        if d_key not in streams: streams[d_key] = []
                        
                        for row in raw_samples:
                            if isinstance(row, (tuple, list)):
                                scaled = [val / (scal[i] if i < len(scal) and scal[i] != 0 else (scal[-1] if len(scal) > 0 and scal[-1] != 0 else 1)) if isinstance(val, (int, float)) else val for i, val in enumerate(row)]
                                streams[d_key].append(scaled)
                            elif isinstance(row, (int, float)):
                                streams[d_key].append(row / (scal[0] if len(scal) > 0 and scal[0] != 0 else 1))
                            else: streams[d_key].append(row)
    return constants, streams

def _iter_boxes(data, offset=0, end=None):
    if end is None: end = len(data)
    while offset + 8 <= end:
        size = struct.unpack_from(">I", data, offset)[0]
        box_type = data[offset + 4: offset + 8].decode("latin-1", errors="ignore")
        if size == 1: size = struct.unpack_from(">Q", data, offset + 8)[0]; header = 16
        elif size == 0: size = end - offset; header = 8
        else: header = 8
        yield box_type, data[offset + header: offset + size]
        offset += size

def _find_box(data, *path):
    current = data
    for step in path:
        found = None
        for btype, bdata in _iter_boxes(current):
            if btype == step: found = bdata; break
        if found is None: return None
        current = found
    return current

def _find_all_boxes(data, target):
    for btype, bdata in _iter_boxes(data):
        if btype == target: yield bdata

def _find_gpmf_samples_with_timing(mp4_bytes):
    moov = _find_box(mp4_bytes, "moov")
    if not moov: return []
    mvhd = _find_box(moov, "mvhd")
    mv_timescale = struct.unpack_from(">I", mvhd, 12)[0] if mvhd else 600

    for trak in _find_all_boxes(moov, "trak"):
        mdia = _find_box(trak, "mdia")
        if not mdia: continue
        mdhd = _find_box(mdia, "mdhd")
        trak_timescale = struct.unpack_from(">I", mdhd, 12)[0] if mdhd else mv_timescale
        stbl = _find_box(mdia, "minf", "stbl")
        if not stbl: continue
        stsd = _find_box(stbl, "stsd")
        if not stsd or struct.unpack_from(">I", stsd, 4)[0] == 0 or stsd[12:16] != b"gpmd": continue

        stco, co64 = _find_box(stbl, "stco"), _find_box(stbl, "co64")
        if stco: offsets = [struct.unpack_from(">I", stco, 8 + i * 4)[0] for i in range(struct.unpack_from(">I", stco, 4)[0])]
        elif co64: offsets = [struct.unpack_from(">Q", co64, 8 + i * 8)[0] for i in range(struct.unpack_from(">I", co64, 4)[0])]
        else: offsets = []

        stsz = _find_box(stbl, "stsz")
        default_sz, n_samples = struct.unpack_from(">I", stsz, 4)[0], struct.unpack_from(">I", stsz, 8)[0]
        sizes = [default_sz] * n_samples if default_sz else [struct.unpack_from(">I", stsz, 12 + i * 4)[0] for i in range(n_samples)]

        stts = _find_box(stbl, "stts")
        sample_durations = []
        for i in range(struct.unpack_from(">I", stts, 4)[0]):
            count, delta = struct.unpack_from(">II", stts, 8 + i * 8)
            sample_durations.extend([delta] * count)

        stsc = _find_box(stbl, "stsc")
        n_sc = struct.unpack_from(">I", stsc, 4)[0]
        sc_rows = [struct.unpack_from(">III", stsc, 8 + i * 12) for i in range(n_sc)]

        samples, idx, current_time_ticks = [], 0, 0
        for ei, (first_chunk, spc, _) in enumerate(sc_rows):
            next_first = sc_rows[ei + 1][0] if ei + 1 < n_sc else len(offsets) + 1
            for ci in range(first_chunk - 1, next_first - 1):
                if ci >= len(offsets) or idx >= len(sizes): break
                chunk_off = offsets[ci]
                for _ in range(spc):
                    if idx >= len(sizes): break
                    sz = sizes[idx]
                    dur_ticks = sample_durations[idx] if idx < len(sample_durations) else 0
                    samples.append((chunk_off, sz, current_time_ticks / trak_timescale, dur_ticks / trak_timescale))
                    chunk_off += sz
                    current_time_ticks += dur_ticks
                    idx += 1
        return samples
    return []

def extract_streams_with_time(mp4_path):
    data = Path(mp4_path).read_bytes()
    
    global_constants = {}
    gpmf_global_blob = _find_box(data, "moov", "udta", "GPMF")
    if gpmf_global_blob:
        global_ast = parse_gpmf(gpmf_global_blob)
        global_constants = flatten_global_ast(global_ast)
        
    samples = _find_gpmf_samples_with_timing(data)
    timed_streams = {}
    
    for off, sz, t_start, t_dur in samples:
        if sz == 0: continue
        ast = parse_gpmf(data[off: off + sz])
        constants, chunk_streams = extract_all_telemetry(ast)
        global_constants.update(constants) 
        
        for key, val_list in chunk_streams.items():
            if not val_list: continue
            n_items = len(val_list)
            if key not in timed_streams: timed_streams[key] = []
            for i, val in enumerate(val_list):
                timed_streams[key].append({
                    "time_sec": t_start + (i / n_items) * t_dur,
                    "data": val
                })
    return timed_streams, global_constants

# =======================================================================
# INTERNAL ACCURACY & SENSOR HEALTH EVALUATION
# =======================================================================
def _local_haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def evaluate_telemetry_health(streams):
    """
    Evaluates GoPro telemetry against physical constraints and internal consistency
    without requiring ground truth hardware.
    """
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
    
    # 1. GPS Internal Consistency & Physical Constraints (Jerk)
    if "GPS9" in streams and len(streams["GPS9"]) > 1:
        gps_data = streams["GPS9"]
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
            dop = curr["data"][7]
            fix = curr["data"][8]
            
            if fix < 3: bad_fix_count += 1
            if dop > 3.0: bad_dop_count += 1
            
            dist = _local_haversine(lat1, lon1, lat2, lon2)
            derived_speed = dist / dt
            
            # Acceleration / Jerk Physics check
            accel = abs(doppler_speed_curr - doppler_speed_prev) / dt
            jerks.append(accel / dt)
            
            # Compare derived speed to doppler speed (filter noise at standstill)
            if doppler_speed_curr > 1.0 or derived_speed > 1.0:
                speed_errors.append(abs(derived_speed - doppler_speed_curr))
        
        avg_speed_error = sum(speed_errors) / len(speed_errors) if speed_errors else 0
        max_jerk = max(jerks) if jerks else 0
        
        report["metrics"]["avg_gps_speed_error_ms"] = round(avg_speed_error, 3)
        report["metrics"]["bad_fix_ratio"] = round(bad_fix_count / len(gps_data), 3)
        report["metrics"]["max_jerk_detected"] = round(max_jerk, 2)
        
        # Penalties for GPS
        gps_penalty = (avg_speed_error * 5) + ((bad_fix_count / len(gps_data)) * 40) + ((bad_dop_count / len(gps_data)) * 15)
        if max_jerk > 20.0: gps_penalty += 15  # Impossible kinematic movement penalty
            
        report["gps_score"] = max(0.0, min(100.0, 100.0 - gps_penalty))
        
        if max_jerk > 20.0:
            report["warnings"].append(f"Kinematic Anomaly: GPS coordinate jumped violently (Jerk: {max_jerk:.1f} m/s³).")
        if report["gps_score"] < 80:
            report["warnings"].append(f"GPS Quality Degraded (Score: {report['gps_score']:.1f}%). High spatial drift.")
            
    # 2. IMU Gravity Magnitude (1G Check)
    if "GRAV" in streams and len(streams["GRAV"]) > 0:
        grav_data = streams["GRAV"]
        mag_errors = []
        
        for item in grav_data:
            x, y, z = item["data"]
            mag = math.sqrt(x*x + y*y + z*z)
            mag_errors.append(abs(1.0 - mag))
            
        avg_mag_error = sum(mag_errors) / len(mag_errors) if mag_errors else 0
        report["metrics"]["avg_grav_mag_error"] = round(avg_mag_error, 4)
        
        # Penalty: 0.05 average deviation is a 10% penalty
        imu_penalty = avg_mag_error * 200 
        report["imu_score"] = max(0.0, min(100.0, 100.0 - imu_penalty))
        
        if report["imu_score"] < 90:
            report["warnings"].append(f"IMU Calibration/Vibration Issue (Score: {report['imu_score']:.1f}%). Gravity vector unstable.")
            
    return report

def get_telemetry_interpolators(streams):
    interpolators = {}
    
    if "GPS9" in streams:
        valid_gps = []
        for s in streams["GPS9"]:
            lat, lon, alt, s2d, s3d, days, secs, dop, fix = s["data"]
            if fix >= 2 and dop <= 5.0:
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

    if "GRAV" in streams:
        times = np.array([s["time_sec"] for s in streams["GRAV"]])
        pitches, rolls = [], []
        for s in streams["GRAV"]:
            x, y, z = s["data"]
            pitches.append(-np.degrees(np.arctan2(z, y)))
            rolls.append(np.degrees(np.arctan2(x, y)))
            
        p_arr, r_arr = np.array(pitches), np.array(rolls)
        if len(p_arr) > 11:
            w = min(31, len(p_arr) if len(p_arr)%2!=0 else len(p_arr)-1)
            p_arr = savgol_filter(p_arr, w, 3)
            r_arr = savgol_filter(r_arr, w, 3)
            
        interpolators["pitch"] = interp1d(times, p_arr, bounds_error=False, fill_value="extrapolate")
        interpolators["roll"] = interp1d(times, r_arr, bounds_error=False, fill_value="extrapolate")

    return interpolators