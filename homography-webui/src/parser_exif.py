import os
import re
import struct
import math
import exifread
from utils import sanitize_meta
from parser_gpmf import parse_gpmf, extract_all_telemetry, flatten_global_ast

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

def extract_full_photo_metadata(filepath):
    lat, lon = None, None
    exif_dict = {}
    try:
        with open(filepath, 'rb') as f: tags = exifread.process_file(f, details=False)
        for tag, val in tags.items():
            if tag.startswith('JPEG') or tag.startswith('Thumbnail') or tag.startswith('EXIF MakerNote'): continue
            exif_dict[tag] = str(val.printable) if hasattr(val, 'printable') else str(val)
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            def convert_to_degrees(value):
                d, m, s = value.values
                return float(d.num)/d.den + (float(m.num)/m.den)/60.0 + (float(s.num)/s.den)/3600.0
            lat = convert_to_degrees(tags['GPS GPSLatitude'])
            if tags.get('GPS GPSLatitudeRef') and tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
            lon = convert_to_degrees(tags['GPS GPSLongitude'])
            if tags.get('GPS GPSLongitudeRef') and tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
            exif_dict['Parsed_Latitude'] = lat
            exif_dict['Parsed_Longitude'] = lon
    except Exception: pass

    grav_vec, xfov, yfov, klns = None, None, None, None
    pitch_ui, roll_ui = None, None
    xmp_dict, gpmf_dict = {}, {}
    try:
        xmp_raw, gpmf_raw = extract_jpeg_metadata_blocks(filepath)
        if xmp_raw:
            xmp_dict = parse_xmp_gpano(xmp_raw)
        if gpmf_raw:
            ast = parse_gpmf(gpmf_raw)
            constants, _ = extract_all_telemetry(ast)
            global_constants = flatten_global_ast(ast)
            constants.update(global_constants)
            gpmf_dict = constants
            
            # --- NEW VECTOR-BASED HOMOGRAPHY & YFOV TELEMETRY (From Tester) ---
            if 'GRAV' in constants:
                grav_vec = list(constants['GRAV'])
                # --- RESTORED FOR UI DISPLAY ONLY ---
                # Derive Euler angles solely for UI display (does not affect vector homography)
                gx, gy, gz = grav_vec
                pitch_ui = -math.degrees(math.atan2(gz, gy))
                roll_ui = math.degrees(math.atan2(gx, gy))
            
            xfov = constants.get('XFOV', None)
            yfov = constants.get('YFOV', None)
            
            if xfov is None:
                zfov, aruw = constants.get('ZFOV'), constants.get('ARUW')
                if zfov is not None and aruw is not None:
                    try: xfov = math.degrees(2.0 * math.atan(math.tan(math.radians(float(zfov)) / 2.0) * (float(aruw) / math.sqrt(float(aruw)**2 + 1))))
                    except Exception: pass
            
            klns = constants.get('KLNS', None)
    except Exception: pass

    full_meta = {
        "EXIF": sanitize_meta(exif_dict), 
        "XMP_GPano": sanitize_meta(xmp_dict), 
        "GPMF": sanitize_meta(gpmf_dict), 
        "Computed_Variables": {
            "Latitude": lat, "Longitude": lon, 
            "grav_vec": grav_vec, "XFOV": xfov, "YFOV": yfov, 
            "KLNS": klns, "Pitch_UI": pitch_ui, "Roll_UI": roll_ui
        }
    }
    return lat, lon, grav_vec, klns, xfov, yfov, pitch_ui, roll_ui, full_meta