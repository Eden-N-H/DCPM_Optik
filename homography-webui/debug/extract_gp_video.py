import os
import re
import struct
import subprocess

FILE_PATH = "/home/mezza/Downloads/Airstrip Road.mp4"
BIN_PATH = "telemetry.bin"

# ==============================================================================
# 1. MP4 NATIVE HEADER PARSER (For Global Metadata / ExifTool replacement)
# ==============================================================================

def extract_global_gpmf(mp4_path):
    """
    Rapidly seeks through the MP4 box structure to find the global settings payload
    located at: moov -> udta -> GPMF. Reads efficiently without loading the whole file.
    """
    print("--- 1. EXTRACTING GLOBAL METADATA FROM MP4 HEADER ---")
    if not os.path.exists(mp4_path):
        print(f"[-] File not found: {mp4_path}")
        return None

    with open(mp4_path, "rb") as f:
        f.seek(0, 2)
        eof = f.tell()
        f.seek(0)
        
        def find_atom(target, end_offset):
            while f.tell() + 8 <= end_offset:
                offset = f.tell()
                header = f.read(8)
                if len(header) < 8: break
                
                size = struct.unpack(">I", header[:4])[0]
                btype = header[4:8].decode("latin-1", errors="ignore")
                
                header_len = 8
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                    header_len = 16
                elif size == 0:
                    size = end_offset - offset
                    
                if btype == target:
                    return offset + header_len, offset + size
                    
                f.seek(offset + size)
            return None, None

        # Traverse hierarchy
        moov_start, moov_end = find_atom("moov", eof)
        if not moov_start: return None
        
        f.seek(moov_start)
        udta_start, udta_end = find_atom("udta", moov_end)
        if not udta_start: return None
        
        f.seek(udta_start)
        gpmf_start, gpmf_end = find_atom("GPMF", udta_end)
        if not gpmf_start: return None
        
        f.seek(gpmf_start)
        blob = f.read(gpmf_end - gpmf_start)
        print(f"[+] Successfully extracted moov/udta/GPMF payload ({len(blob)} bytes)\n")
        return blob

# ==============================================================================
# 2. TELEMETRY TRACK EXTRACTION (For Time-Series Sensor Data)
# ==============================================================================

def extract_binary_stream(mp4_path, bin_path):
    print("--- 2. EXTRACTING TELEMETRY TRACK VIA FFMPEG ---")
    cmd = ["ffmpeg", "-y", "-i", mp4_path, "-map", "0:3", "-f", "data", "-c", "copy", bin_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[-] FFmpeg failed to extract stream. Ensure ffmpeg is installed.")
        return False
    print(f"[+] Extracted time-series binary track to {bin_path} ({os.path.getsize(bin_path) / 1024:.1f} KB)\n")
    return True

# ==============================================================================
# 3. GPMF UNPACKING AND PARSING
# ==============================================================================

def unpack_gpmf_data(type_val, size, repeat, raw_data):
    """Unpacks basic homogeneous binary data arrays."""
    type_char = type_val.decode('ascii', errors='ignore')
    
    fmt_map = {
        's': 'h', 'S': 'H', 'l': 'i', 'L': 'I', 
        'f': 'f', 'd': 'd', 'B': 'B', 'b': 'b', 
        'J': 'Q', 'j': 'q'
    }
    
    if type_char in fmt_map:
        py_type = fmt_map[type_char]
        vals_per_repeat = size // struct.calcsize(py_type)
        fmt = f">{vals_per_repeat * repeat}{py_type}"
        
        try:
            unpacked = struct.unpack(fmt, raw_data[:struct.calcsize(fmt)])
            if vals_per_repeat > 1:
                return [unpacked[i:i+vals_per_repeat] for i in range(0, len(unpacked), vals_per_repeat)]
            return list(unpacked)
        except Exception:
            return raw_data
            
    elif type_char == 'c':
        return raw_data.decode('ascii', errors='ignore').strip('\x00')
        
    elif type_char in ['F', 'U']:
        results = []
        for i in range(repeat):
            chunk = raw_data[i*size : (i+1)*size]
            results.append(chunk.decode('ascii', errors='ignore').strip('\x00'))
        return results if repeat > 1 else results[0]
        
    return raw_data 

def unpack_complex_gpmf(type_str, size, repeat, raw_data):
    """Dynamically compiles a C-Struct schema to unpack heterogeneous payloads."""
    matches = re.findall(r'(\[\d+\])?([a-zA-Z])', type_str)
    parsed = []
    for bracket, char in matches:
        count = int(bracket[1:-1]) if bracket else 1
        parsed.append((count, char))
        
    fmt_map = {
        'b': ('b', 1), 'B': ('B', 1), 'c': ('s', 1), 'd': ('d', 8),
        'f': ('f', 4), 'l': ('i', 4), 'L': ('I', 4), 's': ('h', 2),
        'S': ('H', 2), 'j': ('q', 8), 'J': ('Q', 8), 'F': ('4s', 4),
        'U': ('16s', 16)
    }
    
    fmt = '>'
    expected_size = 0
    flat_chars = []
    
    for count, char in parsed:
        if char not in fmt_map: return None
        py_fmt, bytes_per_item = fmt_map[char]
        
        if py_fmt.endswith('s'):
            base_len = int(py_fmt[:-1]) if len(py_fmt) > 1 else 1
            total_len = count * base_len
            fmt += f"{total_len}s"
            expected_size += total_len
            flat_chars.append(char)
        else:
            fmt += f"{count}{py_fmt}"
            expected_size += count * bytes_per_item
            flat_chars.extend([char] * count)
            
    if expected_size == 0 or expected_size > size:
        return None
        
    results = []
    try:
        for i in range(repeat):
            chunk = raw_data[i*size : (i+1)*size]
            if len(chunk) < expected_size: break
            
            unpacked = struct.unpack(fmt, chunk[:expected_size])
            
            cleaned = []
            for j, char in enumerate(flat_chars):
                val = unpacked[j]
                if char in ('F', 'U', 'c'):
                    try: val = val.decode('ascii', errors='ignore').strip('\x00')
                    except: pass
                cleaned.append(val)
            
            results.append(cleaned[0] if len(cleaned) == 1 else tuple(cleaned))
        return results
    except Exception:
        return None

def parse_gpmf(data, offset=0, end=None):
    """Recursively decodes the TLV GPMF structure into an AST."""
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
        
        node = {
            'key': key,
            'type': type_val.decode('ascii', errors='ignore'),
            'size': size,
            'repeat': repeat,
            'raw': data[value_offset:value_offset + length]
        }
        
        if type_val == b'\x00':
            node['children'] = parse_gpmf(data, value_offset, value_offset + length)
            node['value'] = None
        else:
            node['value'] = unpack_gpmf_data(type_val, size, repeat, node['raw'])
            
        elements.append(node)
        offset += 8 + padded_length
        
    return elements

def flatten_global_ast(ast):
    """Flattens nested global setting AST nodes into a single key-value dictionary."""
    out = {}
    for node in ast:
        if node.get('children'):
            out.update(flatten_global_ast(node['children']))
        else:
            val = node.get('value')
            if val is None: val = node.get('raw')
            if isinstance(val, list) and len(val) == 1: val = val[0]
            out[node['key']] = val
    return out

def extract_all_telemetry(ast):
    constants = {}
    streams = {}
    
    METADATA_KEYS = {'STNM', 'SCAL', 'UNIT', 'SIUN', 'TYPE', 'MTRY', 'OUTR', 'ORIN', 'TICK', 'TSMP', 'TIMO', 'EMP'}

    for devc in ast:
        if devc['key'] == 'DEVC':
            for item in devc.get('children', []):
                
                # Map Device Constants
                if item['key'] != 'STRM':
                    val = item.get('value')
                    if val is None: val = item.get('raw')
                    if isinstance(val, list) and len(val) == 1: val = val[0]
                    constants[item['key']] = val
                    
                # Map Time-Series Streams
                else:
                    strm_dict = {t['key']: t for t in item.get('children', [])}
                    
                    scal_node = strm_dict.get('SCAL')
                    scal = scal_node['value'] if scal_node and scal_node['value'] is not None else [1]
                    if not isinstance(scal, (list, tuple)): scal = [scal]
                    
                    type_node = strm_dict.get('TYPE')
                    type_str = ''
                    if type_node:
                        tv = type_node['value']
                        if isinstance(tv, list) and all(isinstance(x, str) for x in tv): type_str = "".join(tv)
                        elif isinstance(tv, str): type_str = tv
                        
                    stnm = strm_dict.get('STNM', {}).get('value', 'Unknown Data')
                    unit = strm_dict.get('UNIT', {}).get('value', '')
                    if isinstance(stnm, list): stnm = "".join(str(x) for x in stnm)
                    if isinstance(unit, list): unit = "".join(str(x) for x in unit)
                    
                    data_keys = [k for k in strm_dict.keys() if k not in METADATA_KEYS]
                    
                    for d_key in data_keys:
                        d_node = strm_dict[d_key]
                        raw_samples = d_node.get('value')
                        
                        if isinstance(raw_samples, bytes) and type_str:
                            if d_node['repeat'] == 0:
                                raw_samples = []
                            else:
                                complex_res = unpack_complex_gpmf(type_str, d_node['size'], d_node['repeat'], d_node['raw'])
                                if complex_res is not None:
                                    raw_samples = complex_res
                        
                        if isinstance(raw_samples, bytes) and d_node['repeat'] == 0: raw_samples = []
                        if not isinstance(raw_samples, list):
                            raw_samples = [raw_samples] if not isinstance(raw_samples, bytes) else [raw_samples]
                            
                        if d_key not in streams:
                            streams[d_key] = {'name': stnm, 'units': unit, 'samples': []}
                            
                        for row in raw_samples:
                            if isinstance(row, (tuple, list)):
                                scaled_row = []
                                for i, val in enumerate(row):
                                    if isinstance(val, (int, float)):
                                        s_factor = scal[i] if i < len(scal) and scal[i] != 0 else (scal[-1] if len(scal) > 0 and scal[-1] != 0 else 1)
                                        scaled_row.append(val / s_factor)
                                    else:
                                        scaled_row.append(val)
                                streams[d_key]['samples'].append(tuple(scaled_row))
                            elif isinstance(row, (int, float)):
                                s_factor = scal[0] if len(scal) > 0 and scal[0] != 0 else 1
                                streams[d_key]['samples'].append(row / s_factor)
                            else:
                                streams[d_key]['samples'].append(row)

    return constants, streams

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return
        
    # --- PHASE 1: GLOBAL METADATA (ExifTool Alternative) ---
    global_blob = extract_global_gpmf(FILE_PATH)
    global_dict = {}
    if global_blob:
        global_ast = parse_gpmf(global_blob)
        global_dict = flatten_global_ast(global_ast)
        
    # --- PHASE 2: TIME-SERIES TELEMETRY TRACK ---
    if not extract_binary_stream(FILE_PATH, BIN_PATH): return

    print("--- 3. PARSING TELEMETRY AST ---")
    with open(BIN_PATH, "rb") as f:
        ast = parse_gpmf(f.read())
    
    print("--- 4. EXTRACTING TELEMETRY DATA STREAMS ---")
    constants, streams = extract_all_telemetry(ast)

    # --- PRINTING UNIFIED RESULTS ---
    print("\n==============================================")
    print("      GLOBAL CAMERA/OPTICAL SETTINGS (moov)   ")
    print("==============================================")
    if global_dict:
        for key, val in sorted(global_dict.items()):
            # Filter out massive binary blobs for cleaner output
            if isinstance(val, bytes) and len(val) > 100:
                print(f"  {key:<6} : [Binary Data: {len(val)} bytes]")
            else:
                print(f"  {key:<6} : {val}")
    else:
        print("  [-] No global settings found in moov/udta.")

    print("\n==============================================")
    print("      TELEMETRY DEVICE CONSTANTS (track)      ")
    print("==============================================")
    for key, val in constants.items():
        print(f"  {key:<6} : {val}")

    print("\n==============================================")
    print("            TELEMETRY DATA STREAMS            ")
    print("==============================================")
    
    for key, data in sorted(streams.items()):
        samples = data['samples']
        count = len(samples)
        if count == 0:
            print(f"[{key}] {data['name']} -> NO DATA DETECTED (0 Samples)")
            print("-" * 50)
            continue
        
        name = data['name']
        units = f"({data['units']})" if data['units'] else ""
        print(f"[{key}] {name} {units}")
        print(f"       -> Total Samples: {count:,}")
        
        sample_val = samples[0]
        sample_str = f"Raw Bytes ({len(sample_val)} bytes)" if isinstance(sample_val, bytes) else str(sample_val)
        print(f"       -> Sample [0]   : {sample_str}")
        print("-" * 50)
        
    print("\nExtraction complete! Both Global Metadata and Telemetry Tracks successfully parsed.")

if __name__ == "__main__":
    main()