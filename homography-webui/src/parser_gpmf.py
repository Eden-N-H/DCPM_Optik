import struct
import re
from pathlib import Path

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
