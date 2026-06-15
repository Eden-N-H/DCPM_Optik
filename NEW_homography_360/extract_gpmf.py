import struct
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
# MP4 BOX WALKER
# ─────────────────────────────────────────────

def _iter_boxes(data: bytes, offset: int = 0, end: int = None):
    if end is None:
        end = len(data)
    while offset + 8 <= end:
        size     = struct.unpack_from(">I", data, offset)[0]
        box_type = data[offset + 4: offset + 8].decode("latin-1")
        if size == 1:
            size   = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size   = end - offset
            header = 8
        else:
            header = 8
        yield box_type, data[offset + header: offset + size]
        offset += size


def _find_box(data: bytes, *path: str) -> Optional[bytes]:
    current = data
    for step in path:
        found = None
        for btype, bdata in _iter_boxes(current):
            if btype == step:
                found = bdata
                break
        if found is None:
            return None
        current = found
    return current


def _find_all_boxes(data: bytes, target: str):
    for btype, bdata in _iter_boxes(data):
        if btype == target:
            yield bdata


# ─────────────────────────────────────────────
# GPMF DECODER (with timing metadata)
# ─────────────────────────────────────────────

_TYPE_MAP = {
    'b': ('b', 1), 'B': ('B', 1),
    's': ('h', 2), 'S': ('H', 2),
    'l': ('i', 4), 'L': ('I', 4),
    'q': ('q', 8), 'Q': ('Q', 8),
    'f': ('f', 4), 'd': ('d', 8),
    'J': ('Q', 8),
}
_STRING_TYPES = frozenset(('c', 'U', 'F'))
_NESTED       = '\x00'


def _decode_gpmf(data: bytes) -> list:
    results = []
    offset  = 0
    while offset + 8 <= len(data):
        fourcc    = data[offset: offset + 4].decode("latin-1")
        type_char = chr(data[offset + 4])
        size      = data[offset + 5]
        repeat    = struct.unpack_from(">H", data, offset + 6)[0]
        offset   += 8

        total   = size * repeat
        payload = data[offset: offset + total]
        offset += (total + 3) & ~3

        if type_char == _NESTED:
            results.append({
                "fourcc": fourcc, "type": "nested",
                "size": size, "repeat": repeat,
                "values": _decode_gpmf(payload),
            })
        elif type_char in _STRING_TYPES:
            results.append({
                "fourcc": fourcc, "type": type_char,
                "size": size, "repeat": repeat,
                "values": payload.decode("latin-1").rstrip('\x00'),
            })
        else:
            info = _TYPE_MAP.get(type_char)
            if info is None:
                values = payload.hex()
            else:
                fmt_char, item_size = info
                n_per   = size // item_size if item_size else 1
                fmt     = f">{n_per}{fmt_char}"
                try:
                    unpacked = [struct.unpack_from(fmt, payload, i * size) for i in range(repeat)]
                    values   = [list(v) if n_per > 1 else v[0] for v in unpacked]
                except struct.error:
                    values = payload.hex()
            results.append({
                "fourcc": fourcc, "type": type_char,
                "size": size, "repeat": repeat,
                "values": values,
            })
    return results


def _flatten_streams_with_timing(entries: list, streams: dict, target: Optional[str],
                                 scale: Optional[list], base_time: float, duration: float):
    """Walks tree, applying scales and computing approximate sample timestamps."""
    current_scale = scale

    total_samples_in_block = 0
    for entry in entries:
        if entry["type"] != "nested" and (not target or entry["fourcc"] == target):
            if isinstance(entry["values"], list):
                total_samples_in_block = max(total_samples_in_block, len(entry["values"]))

    for entry in entries:
        fourcc = entry["fourcc"]
        if fourcc == "SCAL":
            current_scale = entry["values"]
        elif entry["type"] == "nested":
            _flatten_streams_with_timing(entry["values"], streams, target, scale=None,
                                         base_time=base_time, duration=duration)
        else:
            if target and fourcc != target:
                continue
            values = entry["values"]
            if not isinstance(values, list):
                continue

            if current_scale is not None:
                try:
                    s = current_scale
                    if isinstance(s, list) and len(s) == 1:
                        s = s[0]
                        values = [[x / s for x in r] if isinstance(r, list) else r / s for r in values]
                    elif isinstance(s, list):
                        values = [[x / s[i] for i, x in enumerate(r)] if isinstance(r, list) else r for r in values]
                except (TypeError, ZeroDivisionError, IndexError):
                    pass
                current_scale = None

            n_items = len(values)
            if n_items > 0:
                for idx, val in enumerate(values):
                    sample_time = base_time + (idx / n_items) * duration
                    streams.setdefault(fourcc, []).append({
                        "time_sec": sample_time,
                        "data": val
                    })


def _find_gpmf_samples_with_timing(mp4_bytes: bytes) -> list:
    """Finds GPMF samples and parses stts/mvhd tables to align timestamps accurately."""
    moov = _find_box(mp4_bytes, "moov")
    if moov is None:
        raise ValueError("No 'moov' box found.")

    mvhd = _find_box(moov, "mvhd")
    mv_timescale = struct.unpack_from(">I", mvhd, 12)[0] if mvhd else 600

    for trak in _find_all_boxes(moov, "trak"):
        mdia = _find_box(trak, "mdia")
        if mdia is None:
            continue
        mdhd = _find_box(mdia, "mdhd")
        trak_timescale = struct.unpack_from(">I", mdhd, 12)[0] if mdhd else mv_timescale

        stbl = _find_box(mdia, "minf", "stbl")
        if stbl is None:
            continue

        stsd = _find_box(stbl, "stsd")
        if stsd is None or struct.unpack_from(">I", stsd, 4)[0] == 0:
            continue
        if stsd[12:16] != b"gpmd":
            continue

        stco, co64 = _find_box(stbl, "stco"), _find_box(stbl, "co64")
        offsets = []
        if stco:
            n = struct.unpack_from(">I", stco, 4)[0]
            offsets = [struct.unpack_from(">I", stco, 8 + i * 4)[0] for i in range(n)]
        elif co64:
            n = struct.unpack_from(">I", co64, 4)[0]
            offsets = [struct.unpack_from(">Q", co64, 8 + i * 8)[0] for i in range(n)]

        stsz = _find_box(stbl, "stsz")
        default_sz = struct.unpack_from(">I", stsz, 4)[0]
        n_samples  = struct.unpack_from(">I", stsz, 8)[0]
        sizes = [default_sz] * n_samples if default_sz else [struct.unpack_from(">I", stsz, 12 + i * 4)[0] for i in range(n_samples)]

        stts = _find_box(stbl, "stts")
        n_excl = struct.unpack_from(">I", stts, 4)[0]
        sample_durations = []
        for i in range(n_excl):
            count, delta = struct.unpack_from(">II", stts, 8 + i * 8)
            sample_durations.extend([delta] * count)

        stsc = _find_box(stbl, "stsc")
        n_sc = struct.unpack_from(">I", stsc, 4)[0]
        sc_rows = [struct.unpack_from(">III", stsc, 8 + i * 12) for i in range(n_sc)]

        samples = []
        idx = 0
        current_time_ticks = 0

        for ei, (first_chunk, spc, _) in enumerate(sc_rows):
            next_first = sc_rows[ei + 1][0] if ei + 1 < n_sc else len(offsets) + 1
            for ci in range(first_chunk - 1, next_first - 1):
                if ci >= len(offsets) or idx >= len(sizes):
                    break
                chunk_off = offsets[ci]
                for _ in range(spc):
                    if idx >= len(sizes):
                        break
                    sz = sizes[idx]
                    dur_ticks = sample_durations[idx] if idx < len(sample_durations) else 0

                    t_start = current_time_ticks / trak_timescale
                    t_dur   = dur_ticks / trak_timescale

                    samples.append((chunk_off, sz, t_start, t_dur))
                    chunk_off += sz
                    current_time_ticks += dur_ticks
                    idx += 1
        return samples
    raise ValueError("No valid GPMF track metadata matched layout metrics.")


def extract_streams_with_time(mp4_path: str, target: str = None) -> dict:
    """Returns telemetry structures containing exact mapped timeline positions."""
    data = Path(mp4_path).read_bytes()
    samples = _find_gpmf_samples_with_timing(data)
    streams = {}
    for off, sz, t_start, t_dur in samples:
        if sz == 0:
            continue
        blob = data[off: off + sz]
        _flatten_streams_with_timing(_decode_gpmf(blob), streams, target=target, scale=None,
                                     base_time=t_start, duration=t_dur)
    return streams


# ─────────────────────────────────────────────
# TIME SYNCHRONIZATION & INTERPOLATION
# ─────────────────────────────────────────────

def get_telemetry_interpolators(streams: dict):
    """Creates numerical interpolators for fast timeline lookups (e.g. GPS5, GYRO, ACCL)."""
    interpolators = {}
    for stream_name, samples in streams.items():
        times = np.array([s["time_sec"] for s in samples])

        first_val = samples[0]["data"]
        if isinstance(first_val, list):
            data_arr = np.array([s["data"] for s in samples])
            interp_func = interp1d(times, data_arr, axis=0, bounds_error=False, fill_value="extrapolate")
        else:
            data_arr = np.array([s["data"] for s in samples])
            interp_func = interp1d(times, data_arr, bounds_error=False, fill_value="extrapolate")

        interpolators[stream_name] = interp_func
    return interpolators