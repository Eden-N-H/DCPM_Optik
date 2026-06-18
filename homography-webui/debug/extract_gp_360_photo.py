import os
import re
import struct
import sys
import exifread

# Target file path
FILE_PATH = "/home/mezza/Downloads/OneDrive_1_10-06-2026/GSAI6003.JPG"

# ==============================================================================
# 1. NATIVE JPEG MARKER EXTRACTION (EXIF, XMP, GPMF)
# ==============================================================================

def extract_jpeg_metadata_blocks(file_path):
    """Scans the JPEG structure to extract EXIF (APP1), XMP (APP1), and GPMF (APP6) payloads."""
    print(f"--- 1. SCANNING JPEG MARKERS ---")
    if not os.path.exists(file_path):
        print(f"[-] File not found: {file_path}")
        return None, None

    xmp_data = None
    gpmf_data = None

    with open(file_path, 'rb') as f:
        if f.read(2) != b'\xff\xd8':
            print("[-] Error: Not a valid JPEG file.")
            return None, None

        while True:
            marker_header = f.read(2)
            if len(marker_header) < 2 or marker_header[0] != 0xff:
                break
                
            marker_type = marker_header[1]
            if marker_type in (0xd9, 0xda): # EOI or SOS
                break
                
            length_bytes = f.read(2)
            if len(length_bytes) < 2: break
            length = struct.unpack('>H', length_bytes)[0]
            
            payload = f.read(length - 2)
            
            # APP1: XMP Metadata
            if marker_type == 0xe1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
                xmp_data = payload[29:].decode('utf-8', errors='ignore')
                print("[+] Found XMP/GPano Marker (APP1)")

            # APP6: GoPro GPMF Metadata
            elif marker_type == 0xe6 and payload.startswith(b'GoPro\x00'):
                gpmf_data = payload[6:]
                print(f"[+] Found GoPro GPMF Marker (APP6) - {len(gpmf_data)} bytes")

    return xmp_data, gpmf_data

# ==============================================================================
# 2. XMP / GPANO PARSER
# ==============================================================================

def parse_xmp_gpano(xmp_string):
    """Extracts Google Panorama XML tags from the XMP payload."""
    gpano_data = {}
    if not xmp_string: return gpano_data
    
    # Match Attributes: GPano:ProjectionType="equirectangular"
    attr_matches = re.findall(r'GPano:(\w+)=["\']([^"\']+)["\']', xmp_string)
    for key, val in attr_matches:
        gpano_data[key] = val
        
    # Match Elements: <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
    elem_matches = re.findall(r'<GPano:(\w+)>([^<]+)</GPano:\1>', xmp_string)
    for key, val in elem_matches:
        gpano_data[key] = val
        
    return gpano_data

# ==============================================================================
# 3. EXIF & GPS PARSER
# ==============================================================================

def extract_exif_data(file_path):
    """Uses exifread to extract standard EXIF and GPS coordinates."""
    exif_dict = {}
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
        for tag, val in tags.items():
            if tag.startswith('JPEG') or tag.startswith('Thumbnail') or tag.startswith('EXIF MakerNote'):
                continue
            exif_dict[tag] = val
            
        # Parse readable GPS
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
            
    except Exception as e:
        print(f"[-] EXIF Error: {e}")
        
    return exif_dict

# ==============================================================================
# 4. GPMF PARSER LOGIC
# ==============================================================================

def unpack_gpmf_data(type_val, size, repeat, raw_data):
    type_char = type_val.decode('ascii', errors='ignore')
    fmt_map = {'s': 'h', 'S': 'H', 'l': 'i', 'L': 'I', 'f': 'f', 'd': 'd', 'B': 'B', 'b': 'b', 'J': 'Q', 'j': 'q'}
    
    if type_char in fmt_map:
        py_type = fmt_map[type_char]
        vals_per_repeat = size // struct.calcsize(py_type)
        fmt = f">{vals_per_repeat * repeat}{py_type}"
        try:
            unpacked = struct.unpack(fmt, raw_data[:struct.calcsize(fmt)])
            if vals_per_repeat > 1: return [unpacked[i:i+vals_per_repeat] for i in range(0, len(unpacked), vals_per_repeat)]
            return list(unpacked)
        except: return raw_data
    elif type_char == 'c': return raw_data.decode('ascii', errors='ignore').strip('\x00')
    elif type_char in ['F', 'U']:
        results = [(raw_data[i*size : (i+1)*size]).decode('ascii', errors='ignore').strip('\x00') for i in range(repeat)]
        return results if repeat > 1 else results[0]
    return raw_data 

def parse_gpmf(data, offset=0, end=None):
    if end is None: end = len(data)
    elements = []
    while offset < end:
        if offset + 8 > end: break
        key = data[offset:offset+4].decode('ascii', errors='ignore')
        type_val = data[offset+4:offset+5]
        size, repeat = struct.unpack('>BH', data[offset+5:offset+8])
        length, padded_length = size * repeat, (size * repeat + 3) & ~3
        
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
    constants = {}
    for devc in ast:
        if devc['key'] == 'DEVC':
            for item in devc.get('children', []):
                if item['key'] != 'STRM':
                    val = item.get('value')
                    if val is None: val = item.get('raw')
                    if isinstance(val, list) and len(val) == 1: val = val[0]
                    constants[item['key']] = val
    return constants

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    target_file = FILE_PATH
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    print("\n==============================================")
    print(f" ANALYZING: {os.path.basename(target_file)}")
    print("==============================================\n")

    # 1. Extract Payloads
    xmp_raw, gpmf_raw = extract_jpeg_metadata_blocks(target_file)
    print()

    # 2. Parse EXIF & GPS
    print("==============================================")
    print("            EXIF / GPS METADATA (APP1)        ")
    print("==============================================")
    exif_data = extract_exif_data(target_file)
    for k, v in sorted(exif_data.items()):
        if "Parsed" in k:
            print(f"  {k:<20} : {v:.6f}°")
        else:
            # Truncate overly long values
            val_str = str(v)
            if len(val_str) > 60: val_str = val_str[:60] + "..."
            print(f"  {k:<20} : {val_str}")
            
    if not exif_data:
        print("  [-] No EXIF data found.")

    # 3. Parse XMP / GPano
    print("\n==============================================")
    print("            XMP / GPANO METADATA (APP1)       ")
    print("==============================================")
    if xmp_raw:
        gpano_data = parse_xmp_gpano(xmp_raw)
        for k, v in sorted(gpano_data.items()):
            print(f"  {k:<20} : {v}")
    else:
        print("  [-] No XMP GPano data found.")

    # 4. Parse GPMF
    print("\n==============================================")
    print("            GOPRO GPMF METADATA (APP6)        ")
    print("==============================================")
    if gpmf_raw:
        ast = parse_gpmf(gpmf_raw)
        gpmf_constants = extract_all_telemetry(ast)
        for key, val in sorted(gpmf_constants.items()):
            print(f"  {key:<20} : {val}")
    else:
        print("  [-] No GPMF payload found.")

    print("\n[+] Extraction complete!\n")

if __name__ == "__main__":
    main()