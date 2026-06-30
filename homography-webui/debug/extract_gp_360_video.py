import os
import re
import struct
import subprocess
import sys
import json

# Target file path (can be overridden by CLI argument)
FILE_PATH = "/home/mezza/Downloads/360-videos/Rossi Road Part 1.360"
BIN_PATH = "telemetry_360.bin"

# ==============================================================================
# 1. GPMF UNPACKING AND PARSING LOGIC
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
            if vals_per_repeat > 1: return [list(unpacked[i:i+vals_per_repeat]) for i in range(0, len(unpacked), vals_per_repeat)]
            return list(unpacked)
        except: return raw_data
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
            results.append(cleaned[0] if len(cleaned) == 1 else tuple(cleaned))
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
                    stnm = "".join(str(x) for x in strm_dict.get('STNM', {}).get('value', 'Unknown')) if isinstance(strm_dict.get('STNM', {}).get('value'), list) else strm_dict.get('STNM', {}).get('value', 'Unknown')
                    
                    for d_key in [k for k in strm_dict.keys() if k not in METADATA_KEYS]:
                        d_node = strm_dict[d_key]
                        raw_samples = d_node.get('value', [])
                        
                        if isinstance(raw_samples, bytes) and type_str and d_node['repeat'] > 0:
                            complex_res = unpack_complex_gpmf(type_str, d_node['size'], d_node['repeat'], d_node['raw'])
                            if complex_res is not None: raw_samples = complex_res
                            
                        if isinstance(raw_samples, bytes): raw_samples = []
                        if not isinstance(raw_samples, list): raw_samples = [raw_samples]
                            
                        if d_key not in streams: streams[d_key] = {'name': stnm, 'samples': []}
                        
                        for row in raw_samples:
                            if isinstance(row, (tuple, list)):
                                scaled = [val / (scal[i] if i < len(scal) and scal[i] != 0 else (scal[-1] if len(scal) > 0 and scal[-1] != 0 else 1)) if isinstance(val, (int, float)) else val for i, val in enumerate(row)]
                                streams[d_key]['samples'].append(tuple(scaled))
                            elif isinstance(row, (int, float)):
                                streams[d_key]['samples'].append(row / (scal[0] if len(scal) > 0 and scal[0] != 0 else 1))
                            else: streams[d_key]['samples'].append(row)
    return constants, streams

# ==============================================================================
# 2. FFPROBE & FFMPEG EXTRACTION
# ==============================================================================

def extract_binary_stream(mp4_path, bin_path):
    print("--- 1. PROBING STREAMS VIA FFPROBE ---")
    cmd_probe = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", mp4_path]
    res = subprocess.run(cmd_probe, capture_output=True, text=True)
    if res.returncode != 0:
        print("[-] FFprobe failed.")
        return False
        
    streams = json.loads(res.stdout).get("streams", [])
    gpmd_idx = -1
    for s in streams:
        if s.get("codec_tag_string") == "gpmd" or s.get("codec_name") == "bin_data":
            gpmd_idx = s.get("index")
            break
            
    if gpmd_idx == -1:
        print("[-] No GPMD telemetry stream found in file!")
        return False
        
    print(f"[+] Found GPMD telemetry track at stream index {gpmd_idx}")
    print("\n--- 2. EXTRACTING TELEMETRY TRACK VIA FFMPEG ---")
    
    cmd_extract = ["ffmpeg", "-y", "-i", mp4_path, "-map", f"0:{gpmd_idx}", "-f", "data", "-c", "copy", bin_path]
    subprocess.run(cmd_extract, capture_output=True, text=True)
    
    if os.path.exists(bin_path):
        print(f"[+] Extracted time-series binary track to {bin_path} ({os.path.getsize(bin_path) / 1024:.1f} KB)\n")
        return True
    return False

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    target_file = FILE_PATH
    if len(sys.argv) > 1: target_file = sys.argv[1]
        
    print(f"\n==============================================")
    print(f" ANALYZING 360 VIDEO: {os.path.basename(target_file)}")
    print(f"==============================================\n")

    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return

    if not extract_binary_stream(target_file, BIN_PATH):
        return

    print("--- 3. PARSING TELEMETRY AST ---")
    with open(BIN_PATH, "rb") as f:
        ast = parse_gpmf(f.read())
    
    print("--- 4. EXTRACTING TELEMETRY DATA STREAMS ---")
    constants, streams = extract_all_telemetry(ast)

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
        if count == 0: continue
        print(f"[{key}] {data['name']}")
        print(f"       -> Total Samples: {count:,}")
        
        sample_val = samples[0]
        sample_str = f"Raw Bytes ({len(sample_val)} bytes)" if isinstance(sample_val, bytes) else str(sample_val)
        print(f"       -> Sample [0]   : {sample_str}")
        print("-" * 50)
        
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()