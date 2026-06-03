"""
Extract GPMF (GoPro Metadata Format) telemetry from a GoPro MP4 file.

Dependencies:
    pip install pandas   (optional, for CSV output)

─── List available streams ───────────────────────────────────────────────────
python extract_gpmf.py input.mp4 --list

─── Dump all streams to CSV ──────────────────────────────────────────────────
python extract_gpmf.py input.mp4 --output data.csv

─── Filter to a single stream ────────────────────────────────────────────────
python extract_gpmf.py input.mp4 --stream GPS5 --output gps.csv

─── JSON output ──────────────────────────────────────────────────────────────
python extract_gpmf.py input.mp4 --output data.json

─── Dump raw decoded GPMF tree ───────────────────────────────────────────────
python extract_gpmf.py input.mp4 --raw
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path
from typing import Optional

TEST_FOLDER      = "Test_Videos"
OUTPUT_FOLDER    = "Extracted_Data"

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
# GPMF DECODER
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
    """Recursively parse a GPMF byte stream into a list of entry dicts."""
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
        offset += (total + 3) & ~3   # 4-byte align

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


def _flatten_streams(entries: list, streams: dict,
                     target: Optional[str], scale: Optional[list]):
    """Walk decoded GPMF entries, apply SCAL scaling, collect named streams."""
    current_scale = scale
    for entry in entries:
        fourcc = entry["fourcc"]

        if fourcc == "SCAL":
            current_scale = entry["values"]

        elif entry["type"] == "nested":
            _flatten_streams(entry["values"], streams, target, scale=None)

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
                        values = [[x / s for x in r] if isinstance(r, list) else r / s
                                  for r in values]
                    elif isinstance(s, list):
                        values = [[x / s[i] for i, x in enumerate(r)] if isinstance(r, list) else r
                                  for r in values]
                except (TypeError, ZeroDivisionError, IndexError):
                    pass
                current_scale = None

            streams.setdefault(fourcc, []).extend(values)


# ─────────────────────────────────────────────
# GPMF TRACK LOCATOR
# ─────────────────────────────────────────────

def _find_gpmf_samples(mp4_bytes: bytes) -> list[tuple[int, int]]:
    """Return list of (file_offset, size) for every GPMF sample in the file."""
    moov = _find_box(mp4_bytes, "moov")
    if moov is None:
        raise ValueError("No 'moov' box found — not a valid MP4 file.")

    for trak in _find_all_boxes(moov, "trak"):
        mdia = _find_box(trak, "mdia")
        if mdia is None:
            continue
        stbl = _find_box(mdia, "minf", "stbl")
        if stbl is None:
            continue

        # Only process the track whose sample description contains 'gpmd'
        stsd = _find_box(stbl, "stsd")
        if stsd is None:
            continue
        n_entries = struct.unpack_from(">I", stsd, 4)[0]
        off = 8
        is_gpmf = False
        for _ in range(n_entries):
            if off + 8 > len(stsd):
                break
            esz = struct.unpack_from(">I", stsd, off)[0]
            if stsd[off + 4: off + 8] == b"gpmd":
                is_gpmf = True
                break
            off += max(esz, 8)
        if not is_gpmf:
            continue

        # Chunk offsets (stco = 32-bit, co64 = 64-bit)
        stco = _find_box(stbl, "stco")
        co64 = _find_box(stbl, "co64")
        if stco:
            n = struct.unpack_from(">I", stco, 4)[0]
            offsets = [struct.unpack_from(">I", stco, 8 + i * 4)[0] for i in range(n)]
        elif co64:
            n = struct.unpack_from(">I", co64, 4)[0]
            offsets = [struct.unpack_from(">Q", co64, 8 + i * 8)[0] for i in range(n)]
        else:
            continue

        # Sample sizes
        stsz = _find_box(stbl, "stsz")
        if stsz is None:
            continue
        default_sz = struct.unpack_from(">I", stsz, 4)[0]
        n_samples  = struct.unpack_from(">I", stsz, 8)[0]
        sizes = ([default_sz] * n_samples if default_sz
                 else [struct.unpack_from(">I", stsz, 12 + i * 4)[0] for i in range(n_samples)])

        # Build samples-per-chunk map
        stsc    = _find_box(stbl, "stsc")
        n_sc    = struct.unpack_from(">I", stsc, 4)[0]
        sc_rows = [struct.unpack_from(">III", stsc, 8 + i * 12) for i in range(n_sc)]

        samples = []
        idx = 0
        for ei, (first_chunk, spc, _) in enumerate(sc_rows):
            next_first = sc_rows[ei + 1][0] if ei + 1 < n_sc else len(offsets) + 1
            for ci in range(first_chunk - 1, next_first - 1):
                if ci >= len(offsets):
                    break
                chunk_off = offsets[ci]
                for _ in range(spc):
                    if idx >= len(sizes):
                        break
                    samples.append((chunk_off, sizes[idx]))
                    chunk_off += sizes[idx]
                    idx += 1
        return samples

    raise ValueError(
        "No GPMF telemetry track found. "
        "Ensure this is a GoPro video recorded with telemetry enabled."
    )


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def extract_raw_blobs(mp4_path: str) -> list[bytes]:
    """Return raw GPMF byte blobs, one per sample."""
    data    = Path(mp4_path).read_bytes()
    samples = _find_gpmf_samples(data)
    return [data[off: off + sz] for off, sz in samples if sz > 0]


def extract_streams(mp4_path: str, target: str = None) -> dict:
    """Return dict of {stream_name: [samples]} with SCAL scaling applied."""
    streams = {}
    for blob in extract_raw_blobs(mp4_path):
        _flatten_streams(_decode_gpmf(blob), streams, target=target, scale=None)
    return streams


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def _save_csv(streams: dict, output_path: str):
    try:
        import pandas as pd
        for name, rows in streams.items():
            path = _stream_path(output_path, name, len(streams))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {name} → {path}  ({len(rows)} rows)")
    except ImportError:
        for name, rows in streams.items():
            path = _stream_path(output_path, name, len(streams))
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row if isinstance(row, (list, tuple)) else [row])
            print(f"  {name} → {path}  ({len(rows)} rows)")


def _save_json(streams: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(streams, f, indent=2)
    print(f"  JSON → {output_path}")


def _stream_path(base: str, name: str, n_streams: int) -> str:
    """Return per-stream output path when multiple streams are present."""
    if n_streams == 1:
        return base
    p = Path(base)
    return str(p.with_stem(f"{p.stem}_{name}"))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Extract GPMF telemetry from a GoPro MP4 file.")
    ap.add_argument("mp4",            help="Filename inside Test_Videos/")
    ap.add_argument("--stream", "-s", help="Filter to one 4CC stream (e.g. GPS5, GYRO, ACCL)")
    ap.add_argument("--output", "-o", help="Output filename (.csv or .json); saved to Extracted_Data/")
    ap.add_argument("--list",   "-l", action="store_true", help="List available streams and exit")
    ap.add_argument("--raw",    "-r", action="store_true", help="Dump raw decoded GPMF tree as JSON")
    args = ap.parse_args()

    # ── Resolve paths
    filename = Path(args.mp4).name
    mp4_path = Path(TEST_FOLDER) / filename
    video_stem = mp4_path.stem

    output_dir = Path(OUTPUT_FOLDER) / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Input]  {mp4_path}")

    try:
        # ── Raw tree dump
        if args.raw:
            blobs = extract_raw_blobs(str(mp4_path))
            tree  = [entry for blob in blobs for entry in _decode_gpmf(blob)]
            out   = json.dumps(tree, indent=2, default=str)
            if args.output:
                dest = output_dir / Path(args.output).name
                dest.write_text(out)
                print(f"  Raw GPMF tree → {dest}")
            else:
                print(out)
            return

        # ── Extract streams
        streams = extract_streams(str(mp4_path), target=args.stream)

        if not streams:
            print("No telemetry streams found.")
            return

        if args.list:
            print("Available streams:")
            for name, rows in streams.items():
                print(f"  {name}  ({len(rows)} samples)")
            return

        print(f"[Streams] {', '.join(streams)}")

        # ── Save or preview
        if args.output:
            dest = str(output_dir / Path(args.output).name)
            if Path(args.output).suffix.lower() == ".json":
                _save_json(streams, dest)
            else:
                _save_csv(streams, dest)
        else:
            for name, rows in streams.items():
                print(f"\n── {name}  ({len(rows)} samples) ──")
                for row in rows[:5]:
                    print(f"  {row}")
                if len(rows) > 5:
                    print(f"  ... {len(rows) - 5} more rows")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()