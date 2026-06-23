"""
drain_calib_tool.py
-------------------
Manual segmentation tool for evaluating orthographic rectification accuracy.

Given a GoPro image of a rectangular drain (with GPMF pitch data), this tool:
  1. Extracts pitch from GPMF metadata (or lets you set it manually)
  2. Optionally de-equirectangularises a 360° image (if applicable)
  3. Applies BEV (bird's-eye-view) homography to orthographically rectify
  4. Lets you click to manually outline the drain polygon on the rectified image
  5. Computes the pixel area → real-world area using the GSD
  6. Reports accuracy vs. your known ground-truth area

Controls (in the BEV window):
  Left-click   – add a polygon vertex
  Right-click  – undo the last vertex
  Enter / D    – finish polygon & compute area
  R            – reset polygon
  Q / Escape   – quit without computing

Usage:
  python drain_calib_tool.py --image path/to/gopro.jpg --known_area 0.09
  python drain_calib_tool.py --image path/to/gopro.jpg --known_area 0.09 \\
      --cam_height 1.8 --pitch -20 --is_360 --fov 100

Arguments:
  --image        Path to the input image
  --known_area   Known real-world area of the drain in m² (e.g. 0.09 for 300×300 mm)
  --cam_height   Camera height above ground in metres (default: 1.6)
  --pitch        Camera pitch in degrees, negative = downward (default: auto from GPMF)
  --is_360       Flag: treat the image as equirectangular 360° (default: False)
  --fov          Horizontal FOV in degrees for 360→rectilinear projection (default: 100)
  --gsd          Ground sample distance in m/px for BEV grid (default: 0.005)
  --z_near       Near clipping plane in metres (default: 0.5)
  --z_far        Far clipping plane in metres (default: 6.0)
  --x_range      Half-width of BEV view in metres (default: 3.0)
  --save         Save annotated BEV image to disk
"""

import cv2
import numpy as np
import math
import struct
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# GPMF pitch extraction  (mirrors core_math.extract_gpmf_pitch)
# ─────────────────────────────────────────────────────────────────────────────

def extract_gpmf_pitch(filepath: str, fallback_pitch: float = -15.0) -> float:
    """Pull pitch from GPMF atom embedded in a GoPro JPEG/MP4."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        # Try the GRAV atom first (gravity vector)
        idx = data.find(b"GRAV")
        if idx != -1:
            payload = data[idx + 8: idx + 8 + 12]
            if len(payload) == 12:
                x, y, z = struct.unpack(">fff", payload)
                pitch = math.degrees(math.atan2(z, y))
                print(f"[GPMF] GRAV atom found → pitch = {-abs(pitch):.1f}°")
                return -abs(pitch)

        # Fallback: look for CORI (camera orientation) quaternion
        idx = data.find(b"CORI")
        if idx != -1:
            payload = data[idx + 8: idx + 8 + 16]
            if len(payload) == 16:
                w, px, py, pz = struct.unpack(">ffff", payload)
                # Convert quaternion to pitch (rotation around X axis)
                sinp = 2.0 * (w * py - pz * px)
                pitch_rad = math.asin(max(-1.0, min(1.0, sinp)))
                pitch = math.degrees(pitch_rad)
                print(f"[GPMF] CORI atom found → pitch = {-abs(pitch):.1f}°")
                return -abs(pitch)

    except Exception as e:
        print(f"[GPMF] Read error: {e}")

    print(f"[GPMF] No usable atom found – using fallback pitch = {fallback_pitch}°")
    return fallback_pitch


# ─────────────────────────────────────────────────────────────────────────────
# 360° → rectilinear  (mirrors core_math.equirectangular_to_rectilinear)
# ─────────────────────────────────────────────────────────────────────────────

def equirectangular_to_rectilinear(
    equi_img: np.ndarray,
    fov_deg: float = 100.0,
    pitch_deg: float = -15.0,
    yaw_deg: float = 0.0,
    output_width: int = 1280,
    output_height: int = 720,
):
    h, w = equi_img.shape[:2]
    f = (output_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array(
        [[f, 0, output_width / 2.0], [0, f, output_height / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    K_inv = np.linalg.inv(K)

    pitch = math.radians(pitch_deg)
    yaw   = math.radians(yaw_deg)

    R_pitch = np.array([
        [1, 0,              0            ],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)],
    ])
    R_yaw = np.array([
        [ math.cos(yaw), 0, math.sin(yaw)],
        [0,              1, 0            ],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    R_inv = np.linalg.inv(R_yaw @ R_pitch)

    xs, ys = np.meshgrid(np.arange(output_width), np.arange(output_height))
    pixels = np.stack((xs, ys, np.ones_like(xs)), axis=-1).reshape(-1, 3).T

    rays = R_inv @ (K_inv @ pixels)
    norms = np.linalg.norm(rays, axis=0, keepdims=True)
    rays = rays / np.where(norms == 0, 1, norms)

    theta = np.arctan2(rays[0], rays[2])
    phi   = np.arcsin(np.clip(rays[1], -1, 1))

    map_x = ((theta / (2 * math.pi) + 0.5) * w).reshape(output_height, output_width).astype(np.float32)
    map_y = ((phi / math.pi + 0.5) * h).reshape(output_height, output_width).astype(np.float32)

    rect = cv2.remap(equi_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return rect, K


# ─────────────────────────────────────────────────────────────────────────────
# BEV homography  (mirrors core_math.get_bev_homography)
# ─────────────────────────────────────────────────────────────────────────────

def get_bev_homography(
    K: np.ndarray,
    cam_height_m: float,
    pitch_deg: float,
    gsd: float  = 0.005,
    z_near: float = 0.5,
    z_far: float  = 6.0,
    x_range: float = 3.0,
):
    pitch_rad = math.radians(-pitch_deg)   # flip sign: downward pitch is negative input
    road_pts = np.array(
        [[-x_range, z_near], [x_range, z_near], [x_range, z_far], [-x_range, z_far]],
        dtype=np.float32,
    )

    bev_w = int((2 * x_range) / gsd)
    bev_h = int((z_far - z_near) / gsd)
    bev_pts = np.array([[0, bev_h], [bev_w, bev_h], [bev_w, 0], [0, 0]], dtype=np.float32)

    rect_pts = []
    for pt in road_pts:
        X, Z = pt
        Y = cam_height_m
        Y_rot = Y * math.cos(pitch_rad) - Z * math.sin(pitch_rad)
        Z_rot = Y * math.sin(pitch_rad) + Z * math.cos(pitch_rad)
        u = (K[0, 0] * X / Z_rot) + K[0, 2]
        v = (K[1, 1] * Y_rot / Z_rot) + K[1, 2]
        rect_pts.append([u, v])

    H = cv2.getPerspectiveTransform(np.array(rect_pts, dtype=np.float32), bev_pts)
    return H, bev_w, bev_h, gsd, x_range, z_far


# ─────────────────────────────────────────────────────────────────────────────
# Interactive polygon collector
# ─────────────────────────────────────────────────────────────────────────────

class PolygonCollector:
    """Collect polygon vertices from mouse clicks on a displayed image."""

    INSTRUCTIONS = [
        "LEFT CLICK  : add vertex",
        "RIGHT CLICK : undo last",
        "ENTER / D   : compute area",
        "R           : reset",
        "Q / ESC     : quit",
    ]

    def __init__(self, bev_img: np.ndarray, gsd: float, known_area_m2: float):
        self.base      = bev_img.copy()
        self.gsd       = gsd
        self.known_m2  = known_area_m2
        self.pts: list[tuple[int, int]] = []
        self.done      = False
        self.result    = None          # filled after polygon closed

        self._render()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _render(self):
        canvas = self.base.copy()
        h, w = canvas.shape[:2]

        # Instruction overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 22 + 16 * len(self.INSTRUCTIONS)), (20, 20, 20), -1)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
        for i, line in enumerate(self.INSTRUCTIONS):
            cv2.putText(canvas, line, (8, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # Known area reference
        cv2.putText(canvas, f"Known area: {self.known_m2} m²", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 80), 1, cv2.LINE_AA)

        # Polygon
        if len(self.pts) >= 2:
            cv2.polylines(canvas, [np.array(self.pts, dtype=np.int32)], False,
                          (0, 255, 255), 2, cv2.LINE_AA)
        for p in self.pts:
            cv2.circle(canvas, p, 4, (0, 120, 255), -1, cv2.LINE_AA)

        # Scale bar: 0.5 m
        scale_px = int(0.5 / self.gsd)
        bar_y, bar_x = h - 30, w - scale_px - 20
        cv2.line(canvas, (bar_x, bar_y), (bar_x + scale_px, bar_y), (255, 255, 255), 2)
        cv2.putText(canvas, "0.5 m", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        self.canvas = canvas
        cv2.imshow("BEV Drain Segmentation", canvas)

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mouse_cb(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
            self._render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pts:
                self.pts.pop()
                self._render()

    # ── area computation ──────────────────────────────────────────────────────

    def _compute(self) -> dict:
        if len(self.pts) < 3:
            return None

        poly = np.array(self.pts, dtype=np.float32)
        area_px2    = cv2.contourArea(poly)
        area_m2     = area_px2 * (self.gsd ** 2)
        error_pct   = abs(area_m2 - self.known_m2) / self.known_m2 * 100.0

        # Minimum bounding rectangle (useful for a rectangular drain)
        rect       = cv2.minAreaRect(poly.astype(np.int32))
        box        = cv2.boxPoints(rect)
        (cx, cy), (bw_px, bh_px), angle = rect
        bw_m, bh_m = bw_px * self.gsd, bh_px * self.gsd

        return {
            "vertices":          len(self.pts),
            "area_px2":          round(area_px2, 1),
            "area_m2":           round(area_m2,  6),
            "known_m2":          self.known_m2,
            "error_m2":          round(abs(area_m2 - self.known_m2), 6),
            "error_pct":         round(error_pct, 2),
            "min_rect_w_m":      round(bw_m, 4),
            "min_rect_h_m":      round(bh_m, 4),
            "min_rect_area_m2":  round(bw_m * bh_m, 6),
            "gsd_m_per_px":      self.gsd,
            "centroid_px":       (round(float(cx), 1), round(float(cy), 1)),
        }

    def _draw_result(self, result: dict):
        canvas = self.base.copy()
        poly   = np.array(self.pts, dtype=np.int32)

        # Filled polygon (semi-transparent)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (0, 180, 255))
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        cv2.polylines(canvas, [poly], True, (0, 220, 255), 2, cv2.LINE_AA)

        # Vertices
        for p in self.pts:
            cv2.circle(canvas, p, 4, (0, 80, 255), -1)

        # Result overlay
        lines = [
            f"Measured area : {result['area_m2']:.5f} m²",
            f"Known area    : {result['known_m2']} m²",
            f"Error         : {result['error_m2']:.5f} m²  ({result['error_pct']:.1f} %)",
            f"Min-rect size : {result['min_rect_w_m']:.3f} x {result['min_rect_h_m']:.3f} m",
            f"GSD           : {result['gsd_m_per_px']*1000:.1f} mm/px",
        ]

        box_h = 20 + 18 * len(lines)
        overlay2 = canvas.copy()
        cv2.rectangle(overlay2, (0, 0), (340, box_h), (10, 10, 10), -1)
        canvas = cv2.addWeighted(overlay2, 0.65, canvas, 0.35, 0)

        col = (80, 255, 80) if result["error_pct"] < 10 else \
              (0, 200, 255) if result["error_pct"] < 20 else (60, 60, 255)

        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        cv2.imshow("BEV Drain Segmentation", canvas)
        return canvas

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow("BEV Drain Segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BEV Drain Segmentation", 1000, 700)
        cv2.setMouseCallback("BEV Drain Segmentation", self.mouse_cb)

        print("\n[Tool] Window open – follow on-screen instructions.")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key in (13, ord("d")):        # Enter or D → finish
                result = self._compute()
                if result is None:
                    print("[Tool] Need at least 3 points to compute area.")
                    continue
                self.result = result
                final_canvas = self._draw_result(result)
                print("\n[Tool] Polygon closed. Press any key to exit the window.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return final_canvas

            elif key == ord("r"):            # R → reset
                self.pts.clear()
                self._render()
                print("[Tool] Polygon reset.")

            elif key in (27, ord("q")):      # Esc / Q → quit
                cv2.destroyAllWindows()
                print("[Tool] Aborted by user.")
                return None


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, image_path: str, pitch: float):
    bar = "=" * 58
    err = result["error_pct"]
    rating = "✓ EXCELLENT" if err < 5 else \
             "✓ GOOD"      if err < 10 else \
             "⚠ DRIFT"     if err < 20 else \
             "✕ CHECK SETUP"

    print(f"\n{bar}")
    print(f"  DRAIN AREA ACCURACY REPORT")
    print(f"{bar}")
    print(f"  Image         : {Path(image_path).name}")
    print(f"  Camera pitch  : {pitch:.1f}°")
    print(f"  GSD           : {result['gsd_m_per_px']*1000:.1f} mm/px")
    print(f"  Vertices used : {result['vertices']}")
    print(f"{bar}")
    print(f"  Measured area : {result['area_m2']:.5f} m²  ({result['area_m2']*1e6:.0f} mm²)")
    print(f"  Known area    : {result['known_m2']} m²     ({result['known_m2']*1e6:.0f} mm²)")
    print(f"  Absolute err  : {result['error_m2']:.5f} m²  ({result['error_m2']*1e6:.0f} mm²)")
    print(f"  Relative err  : {err:.2f} %")
    print(f"  Rating        : {rating}")
    print(f"{bar}")
    print(f"  Min bounding rect : {result['min_rect_w_m']:.3f} m × {result['min_rect_h_m']:.3f} m")
    print(f"  Min-rect area     : {result['min_rect_area_m2']:.5f} m²")
    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manual drain segmentation tool for ortho-rectification accuracy."
    )
    parser.add_argument("--image",       required=True,        help="Path to GoPro image")
    parser.add_argument("--known_area",  type=float, required=True,
                        help="Known real-world drain area in m² (e.g. 0.09 for 300×300 mm)")
    parser.add_argument("--cam_height",  type=float, default=1.6,   help="Camera height in metres")
    parser.add_argument("--pitch",       type=float, default=None,
                        help="Camera pitch in degrees (negative = downward). Auto-detected from GPMF if omitted.")
    parser.add_argument("--is_360",      action="store_true",
                        help="Treat image as equirectangular 360°")
    parser.add_argument("--fov",         type=float, default=100.0,
                        help="Horizontal FOV for 360→rectilinear (degrees)")
    parser.add_argument("--gsd",         type=float, default=0.005,
                        help="BEV ground sample distance in m/px (default 0.005 = 5 mm/px)")
    parser.add_argument("--z_near",      type=float, default=0.5)
    parser.add_argument("--z_far",       type=float, default=6.0)
    parser.add_argument("--x_range",     type=float, default=3.0)
    parser.add_argument("--save",        action="store_true",
                        help="Save annotated BEV image next to the input file")
    args = parser.parse_args()

    # ── Load image ────────────────────────────────────────────────────────────
    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"[ERROR] cv2 could not read: {args.image}")
    print(f"[Info] Loaded image {img.shape[1]}×{img.shape[0]} px")

    # ── Pitch ─────────────────────────────────────────────────────────────────
    if args.pitch is not None:
        pitch = args.pitch
        print(f"[Info] Using manual pitch = {pitch}°")
    else:
        pitch = extract_gpmf_pitch(args.image)

    # ── Optional: 360° de-projection ─────────────────────────────────────────
    if args.is_360:
        print(f"[Info] De-projecting equirectangular → rectilinear (FOV={args.fov}°, pitch={pitch}°)")
        rect_img, K = equirectangular_to_rectilinear(
            img, fov_deg=args.fov, pitch_deg=pitch, output_width=1280, output_height=720
        )
    else:
        rect_img = img.copy()
        h, w = rect_img.shape[:2]
        # Build K for a standard perspective camera (GoPro ≈ 122° HFOV)
        fov_px = 122.0
        f = (w / 2.0) / math.tan(math.radians(fov_px) / 2.0)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float32)
        print(f"[Info] Using rectilinear mode – K focal ≈ {f:.0f} px")

    # ── BEV homography ────────────────────────────────────────────────────────
    print(f"[Info] Building BEV homography  "
          f"(h={args.cam_height} m, pitch={pitch}°, GSD={args.gsd*1000:.0f} mm/px, "
          f"z=[{args.z_near}, {args.z_far}] m, x=±{args.x_range} m)")

    H, bev_w, bev_h, gsd, x_range, z_far = get_bev_homography(
        K, args.cam_height, pitch,
        gsd=args.gsd, z_near=args.z_near, z_far=args.z_far, x_range=args.x_range
    )
    bev_img = cv2.warpPerspective(rect_img, H, (bev_w, bev_h))
    print(f"[Info] BEV image: {bev_w}×{bev_h} px  "
          f"({bev_w*gsd:.2f} m × {bev_h*gsd:.2f} m scene)")

    # ── Interactive segmentation ───────────────────────────────────────────────
    collector = PolygonCollector(bev_img, gsd=gsd, known_area_m2=args.known_area)
    final_canvas = collector.run()

    if collector.result is None:
        print("[Info] No result – exiting.")
        return

    result = collector.result
    print_report(result, args.image, pitch)

    # ── Optional save ─────────────────────────────────────────────────────────
    if args.save and final_canvas is not None:
        stem    = Path(args.image).stem
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.image).parent
        out_bev = out_dir / f"{stem}_bev_calib_{ts}.jpg"
        out_raw = out_dir / f"{stem}_bev_raw_{ts}.jpg"
        cv2.imwrite(str(out_bev), final_canvas)
        cv2.imwrite(str(out_raw), bev_img)
        print(f"[Saved] Annotated BEV → {out_bev}")
        print(f"[Saved] Raw BEV       → {out_raw}")


if __name__ == "__main__":
    main()