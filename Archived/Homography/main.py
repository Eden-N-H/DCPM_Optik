import cv2
import numpy as np
from PIL import Image, ExifTags
import argparse
import sys
import os

'''
─── Basic usage (auto-detects VP, outputs road_ortho.jpg + road_debug.jpg) ───
python extract_gpmf.py input.jpg

─── Specify output filenames ─────────────────────────────────────────────────
python extract_gpmf.py input.jpg --output my_ortho.jpg --debug my_debug.jpg

─── Set ortho output resolution (default 800×1100) ──────────────────────────
python extract_gpmf.py input.jpg --out-w 1000 --out-h 1400

─── Set camera height above ground in metres (default 1.2) ──────────────────
python extract_gpmf.py input.jpg --camera-height 1.5

─── Skip lens undistortion (if not a GoPro or already undistorted) ───────────
python extract_gpmf.py input.jpg --no-undistort

─── Save edge detection debug image + print edge pixel count ────────────────
python extract_gpmf.py input.jpg --debug-edges

─── Manually click the vanishing point in an interactive window ──────────────
python extract_gpmf.py input.jpg --manual-vp

─── Directly supply vanishing point coordinates (skip RANSAC entirely) ───────
python extract_gpmf.py input.jpg --vp 640 280

─── Tune the road ROI (fraction from top to ignore, default 0.35) ────────────
python extract_gpmf.py input.jpg --roi-top 0.40

─── Tune trapezoid spread (near = bottom width, far = top width) ─────────────
python extract_gpmf.py input.jpg --near-spread 0.45 --far-spread 0.18

─── Tune edge detection parameters ──────────────────────────────────────────
python extract_gpmf.py input.jpg --gaussian-k 9 --morph-k 5 --canny-lo 30 --canny-hi 90

─── If VP detection is unreliable, you supply it manually from a previous debug ──
python extract_gpmf.py input.jpg --vp 512 310 --no-undistort --out-w 800 --out-h 1100
'''

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SENSOR_W = 6.17   # GoPro HERO13 sensor width  (mm)
SENSOR_H = 4.55   # GoPro HERO13 sensor height (mm)

DIST = np.array([-0.3510, 0.1350, 0.0, 0.0, -0.0300], dtype=np.float64)

MAX_PIXELS      = 2_000_000   # auto-downscale threshold
MIN_LEN_FRAC    = 0.07        # Hough min line length as fraction of image width
CAMERA_HEIGHT_M = 1.2         # assumed camera height above ground (metres)

TEST_FOLDER = "Test_Images"

# ─────────────────────────────────────────────
# ANNOTATION CONSTANTS
# ─────────────────────────────────────────────

FONT      = cv2.FONT_HERSHEY_SIMPLEX
LABEL_CLR = (0, 242, 255)   # yellow-cyan

# ─────────────────────────────────────────────
# EXIF
# ─────────────────────────────────────────────

def _dms_to_decimal(dms, ref):
    d, m, s = (x.numerator / x.denominator for x in dms)
    value = d + m / 60 + s / 3600
    return round(-value if ref in ("S", "W") else value, 6)


def extract_exif(path):
    raw = Image.open(path)._getexif() or {}
    exif = {ExifTags.TAGS.get(tag, tag): val for tag, val in raw.items()}
    gps = exif.get("GPSInfo")
    if gps:
        try:
            exif["GPSDecimal"] = {
                "latitude":  _dms_to_decimal(gps[2], gps[1]),
                "longitude": _dms_to_decimal(gps[4], gps[3]),
            }
        except Exception:
            exif["GPSDecimal"] = None
    return exif


def build_K(exif, img_w, img_h):
    fl = exif.get("FocalLength", 2.71)
    if hasattr(fl, "numerator"):
        fl = fl.numerator / fl.denominator
    W = exif.get("ExifImageWidth",  img_w)
    H = exif.get("ExifImageHeight", img_h)
    fx = fl * (W / SENSOR_W)
    fy = fl * (H / SENSOR_H)
    return np.array([[fx, 0, img_w / 2.0],
                     [0, fy, img_h / 2.0],
                     [0,  0,          1.0]], dtype=np.float64)


# ─────────────────────────────────────────────
# DOWNSCALE
# ─────────────────────────────────────────────

def maybe_downscale(img, K, max_px=MAX_PIXELS):
    h, w = img.shape[:2]
    npx = h * w
    if npx <= max_px:
        return img, K
    scale = np.sqrt(max_px / npx)
    nw, nh = int(w * scale), int(h * scale)
    img_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    K_small = K.copy()
    K_small[0] *= scale
    K_small[1] *= scale
    print(f"[Downscale] {w}×{h} → {nw}×{nh}  (scale={scale:.3f})")
    return img_small, K_small


# ─────────────────────────────────────────────
# UNDISTORT
# ─────────────────────────────────────────────

def undistort(img, K):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (w, h), 0.0, (w, h))
    out = cv2.undistort(img, K, DIST, None, new_K)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = out[y:y + rh, x:x + rw]
        new_K[0, 2] -= x
        new_K[1, 2] -= y
    return out, new_K


# ─────────────────────────────────────────────
# EDGE DETECTION  (dual-channel)
# ─────────────────────────────────────────────

def prepare_edges(img, roi_top_frac=0.35, gaussian_k=7, morph_k=3,
                  canny_lo=40, canny_hi=110, debug_edges=False):
    h, w = img.shape[:2]
    roi_y = int(h * roi_top_frac)
    roi   = img[roi_y:]

    gk = gaussian_k | 1  # ensure odd kernel size

    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sat      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]

    edges_lum = cv2.Canny(cv2.GaussianBlur(gray, (gk, gk), 0), canny_lo,     canny_hi)
    edges_sat = cv2.Canny(cv2.GaussianBlur(sat,  (gk, gk), 0), canny_lo // 2, canny_hi // 2)

    edges  = cv2.bitwise_or(edges_lum, edges_sat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if debug_edges:
        cv2.imwrite("debug_edges.jpg", edges)
        print(f"[Edges]  non-zero pixels: {np.count_nonzero(edges)}")

    return edges, roi_y


# ─────────────────────────────────────────────
# HOUGH LINES
# ─────────────────────────────────────────────

def detect_lines(edges, roi_y, img_shape, angle_lo=30, angle_hi=85):
    h, w = img_shape[:2]
    min_len = max(60, int(w * MIN_LEN_FRAC))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                          minLineLength=min_len, maxLineGap=80)
    if raw is None:
        return np.empty((0, 4), dtype=int)

    def _filter(lines_raw, a_lo, a_hi):
        out = []
        half_w = w / 2.0
        for x1, y1, x2, y2 in lines_raw.reshape(-1, 4):
            y1g, y2g = y1 + roi_y, y2 + roi_y
            dx, dy   = float(x2 - x1), float(y2g - y1g)
            angle    = 90.0 if abs(dx) < 1e-3 else abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if not (a_lo <= angle <= a_hi):
                continue
            if abs(dx) > 1e-3:
                intercept_y = y1g + (dy / dx) * (half_w - x1)
                if not (h * 0.15 <= intercept_y <= h * 0.45):
                    continue
            out.append([x1, y1g, x2, y2g])
        return out

    out = _filter(raw, angle_lo, angle_hi)
    if len(out) < 8:
        print(f"[Lines]  sparse ({len(out)}), retrying with wider angle floor")
        out = _filter(raw, 15, angle_hi)
    return np.array(out, dtype=int) if out else np.empty((0, 4), dtype=int)


# ─────────────────────────────────────────────
# VANISHING POINT
# ─────────────────────────────────────────────

def _line_to_h(x1, y1, x2, y2):
    return np.cross([float(x1), float(y1), 1.0], [float(x2), float(y2), 1.0])


def _intersect(l1, l2):
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10:
        return None
    return pt[:2] / pt[2]


def ransac_vp(lines, h, w, n_iter=3000, inlier_thresh=15.0, min_inliers=4):
    if len(lines) < 2:
        print("[VP] fallback — insufficient lines")
        return np.array([w / 2.0, h * 0.35]), np.zeros(len(lines), dtype=bool)

    hl      = [_line_to_h(*l) for l in lines]
    lengths = np.array([np.hypot(float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in lines])
    lengths /= lengths.max()
    sides   = np.array([(lines[k][0] + lines[k][2]) / 2.0 >= w / 2.0
                        for k in range(len(lines))], dtype=int)  # 0=left, 1=right

    best, best_n, best_inliers = None, 0.0, []
    half_w = w / 2.0

    for _ in range(n_iter):
        i, j = np.random.choice(len(hl), 2, replace=False)
        vp = _intersect(hl[i], hl[j])
        if vp is None or vp[1] > h * 0.50 or abs(vp[0] - half_w) > half_w * 2:
            continue

        inlier_idx = []
        for k, l in enumerate(hl):
            if abs(l[0]*vp[0] + l[1]*vp[1] + l[2]) / (np.hypot(l[0], l[1]) + 1e-9) >= inlier_thresh:
                continue
            x1, y1, x2, y2 = lines[k]
            mx, my   = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tx, ty   = vp[0] - mx, vp[1] - my
            dx, dy   = float(x2 - x1), float(y2 - y1)
            cos_a    = abs(tx*dx + ty*dy) / ((np.hypot(tx, ty) + 1e-9) * (np.hypot(dx, dy) + 1e-9))
            if cos_a < 0.7:
                continue
            inlier_idx.append(k)

        if len(inlier_idx) < min_inliers:
            continue
        inlier_sides = sides[inlier_idx]
        if not (np.any(inlier_sides == 0) and np.any(inlier_sides == 1)):
            continue

        weighted_n = sum(lengths[k] for k in inlier_idx)
        weighted_n *= 1.0 + max(0.0, (h * 0.50 - vp[1]) / (h * 0.50)) * 0.3
        if weighted_n > best_n:
            best_n, best, best_inliers = weighted_n, vp.copy(), inlier_idx

    if best is None:
        print("[VP] fallback — no bilateral VP found")
        return np.array([w / 2.0, h * 0.35]), np.zeros(len(lines), dtype=bool)

    mask = np.zeros(len(lines), dtype=bool)
    mask[best_inliers] = True
    print(f"[VP] ({best[0]:.0f}, {best[1]:.0f})  inliers {mask.sum()}/{len(lines)} ({100*mask.mean():.0f}%)")
    return best, mask


def pick_vp_manually(img):
    clone = img.copy()
    vp    = [None]
    win   = "Select vanishing point — press any key to confirm"

    def onclick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vp[0] = (x, y)
            vis = clone.copy()
            cv2.circle(vis, (x, y), 14, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 22, (0, 255, 255), 2)
            cv2.imshow(win, vis)

    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, onclick)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if vp[0] is None:
        print("[VP] no click received, using default")
        h, w = img.shape[:2]
        return np.array([w / 2.0, h * 0.35])
    print(f"[VP] manual selection ({vp[0][0]}, {vp[0][1]})")
    return np.array([float(vp[0][0]), float(vp[0][1])])


# ─────────────────────────────────────────────
# ROAD TRAPEZOID  (symmetric left/right)
# ─────────────────────────────────────────────

def road_trapezoid(vp, h, w, near_spread=0.38, far_spread=0.16):
    """
    Projects a real-world rectangle onto the image plane via the VP.
    Near corners are placed at the bottom of the image; far corners are
    found by casting rays from the VP through the near corners up to far_y.
    The homography will unwarp this back into a rectangle.
    Order: TL, TR, BR, BL  (far-left, far-right, near-right, near-left)
    """
    vx, vy  = float(vp[0]), float(vp[1])
    near_y  = h * 0.92
    far_y   = max(h * 0.45, vy + h * 0.05)
    cx      = w / 2.0
    near_hw = w * near_spread

    near_l = np.array([cx - near_hw, near_y])
    near_r = np.array([cx + near_hw, near_y])

    def _ray_at_y(corner, target_y):
        t = (target_y - vy) / (corner[1] - vy + 1e-9)
        return np.array([vx + t * (corner[0] - vx), target_y])

    far_l = _ray_at_y(near_l, far_y)
    far_r = _ray_at_y(near_r, far_y)

    print(
        f"[Trap]  VP=({vx:.0f},{vy:.0f})\n"
        f"        near y={near_y:.0f}  [{near_l[0]:.0f} – {near_r[0]:.0f}]\n"
        f"        far  y={far_y:.0f}   [{far_l[0]:.0f} – {far_r[0]:.0f}]"
    )
    return np.array([far_l, far_r, near_r, near_l], dtype=np.float32)


# ─────────────────────────────────────────────
# WARP
# ─────────────────────────────────────────────

def warp_road(img, src, out_w=800, out_h=1100):
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    H   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (out_w, out_h)), H


# ─────────────────────────────────────────────
# METRIC PROJECTION
# ─────────────────────────────────────────────

def pixels_to_meters(K, vp, camera_height, pixel_points):
    """
    Project 2-D image pixel coordinates onto the flat ground plane using
    a pinhole model tilted toward the vanishing point.
    Returns Nx2 array of (X, Y) metric coordinates.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    theta  = np.arctan2(vp[1] - cy, fy)

    pts    = np.asarray(pixel_points, dtype=np.float64)
    denom  = (pts[:, 1] - cy) * np.cos(theta) + fy * np.sin(theta)
    valid  = np.abs(denom) > 1e-5
    Y      = np.where(valid, (camera_height * fy) / denom, 0.0)
    X      = np.where(valid, (pts[:, 0] - cx) * Y / fx,   0.0)
    return np.stack([X, Y], axis=1)


# ─────────────────────────────────────────────
# ANNOTATION HELPERS
# ─────────────────────────────────────────────

def put_label(img, text, pos, scale=0.55, color=LABEL_CLR, thickness=2):
    """Draw text with a dark drop-shadow for readability."""
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y),     FONT, scale, color,      thickness,     cv2.LINE_AA)


def draw_dimension_line(img, p1, p2, text, color=LABEL_CLR, thickness=2):
    """Bi-directional arrow between two points with a centred label."""
    cv2.arrowedLine(img, tuple(p1), tuple(p2), color, thickness, tipLength=0.04)
    cv2.arrowedLine(img, tuple(p2), tuple(p1), color, thickness, tipLength=0.04)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    put_label(img, text, (mid[0] + 6, mid[1] - 6))


def annotate_ortho(warped, metrics, out_w, out_h):
    """Draw dimension labels on the ortho image."""
    pad = 30

    draw_dimension_line(warped, (pad, out_h - pad), (out_w - pad, out_h - pad),
                        f"Near W: {metrics['near_w']:.2f} m")
    draw_dimension_line(warped, (pad, pad), (out_w - pad, pad),
                        f"Far W: {metrics['far_w']:.2f} m")
    draw_dimension_line(warped, (pad, pad), (pad, out_h - pad),
                        f"L: {metrics['left_h']:.2f} m")
    draw_dimension_line(warped, (out_w - pad, pad), (out_w - pad, out_h - pad),
                        f"R: {metrics['right_h']:.2f} m")

    patch_area = (metrics['near_w'] + metrics['far_w']) / 2 * (metrics['left_h'] + metrics['right_h']) / 2
    put_label(warped, f"Patch ~{patch_area:.2f} m2", (pad + 5, out_h // 2), scale=0.5)

    return warped


def annotate_debug(vis, lines, vp, src, inlier_mask, metrics, camera_height):
    """
    Draw vanishing point, trapezoid, inlier lines, and metric labels
    on the original (undistorted) image.
    """
    h, w = vis.shape[:2]

    for i, (x1, y1, x2, y2) in enumerate(lines):
        colour = (0, 220, 0) if (inlier_mask is not None and inlier_mask[i]) else (0, 60, 0)
        cv2.line(vis, (x1, y1), (x2, y2), colour, 2)

    src_c = np.clip(src.copy(), [0, 0], [w - 1, h - 1])
    cv2.polylines(vis, [src_c.astype(int)], True, (0, 255, 255), 3)

    vpx = int(np.clip(vp[0], 0, w - 1))
    vpy = int(np.clip(vp[1], 0, h - 1))
    for pt in src_c.astype(int):
        cv2.line(vis, (vpx, vpy), tuple(pt), (120, 120, 255), 1)
    cv2.circle(vis, (vpx, vpy), 14, (0, 0, 255), -1)
    cv2.circle(vis, (vpx, vpy), 22, (0, 255, 255), 2)
    put_label(vis, "VP", (vpx + 18, vpy - 6), scale=0.7, color=(0, 80, 255))

    for pt, lbl in zip(src_c.astype(int), ["TL", "TR", "BR", "BL"]):
        put_label(vis, lbl, (pt[0] + 6, pt[1] + 6), scale=0.8, color=(255, 220, 0))

    # Metric dimension labels
    bl, br = src_c[3].astype(int), src_c[2].astype(int)
    tl, tr = src_c[0].astype(int), src_c[1].astype(int)

    put_label(vis, f"Near W: {metrics['near_w']:.2f} m",
              ((bl[0]+br[0])//2, (bl[1]+br[1])//2 + 22), scale=0.6)
    put_label(vis, f"Far W: {metrics['far_w']:.2f} m",
              ((tl[0]+tr[0])//2, (tl[1]+tr[1])//2 - 10), scale=0.6)
    put_label(vis, f"H: {metrics['left_h']:.2f} m",
              ((tl[0]+bl[0])//2 - 120, (tl[1]+bl[1])//2), scale=0.6)
    put_label(vis, f"Camera height: {camera_height:.1f} m (assumed)",
              (10, h - 12), scale=0.5, color=(180, 180, 180))

    return vis


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Road surface orthorectification + metric measurement")
    ap.add_argument("image")
    ap.add_argument("--output",        default="road_ortho.jpg")
    ap.add_argument("--debug",         default="road_debug.jpg")
    ap.add_argument("--out-w",         type=int,   default=800)
    ap.add_argument("--out-h",         type=int,   default=1100)
    ap.add_argument("--roi-top",       type=float, default=0.35)
    ap.add_argument("--gaussian-k",    type=int,   default=7)
    ap.add_argument("--morph-k",       type=int,   default=3)
    ap.add_argument("--canny-lo",      type=int,   default=40)
    ap.add_argument("--canny-hi",      type=int,   default=110)
    ap.add_argument("--near-spread",   type=float, default=0.38)
    ap.add_argument("--far-spread",    type=float, default=0.16)
    ap.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_M,
                    help="Camera height above ground in metres")
    ap.add_argument("--no-undistort",  action="store_true")
    ap.add_argument("--debug-edges",   action="store_true")
    ap.add_argument("--manual-vp",     action="store_true")
    ap.add_argument("--vp",            type=float, nargs=2, metavar=("X", "Y"))
    args = ap.parse_args()

    # ── Resolve paths
    filename   = os.path.basename(args.image)
    args.image = os.path.join(TEST_FOLDER, filename)
    image_name = os.path.splitext(filename)[0]
    output_dir = os.path.join("Processed_Images", image_name)
    os.makedirs(output_dir, exist_ok=True)
    args.output = os.path.join(output_dir, "road_ortho.jpg")
    args.debug  = os.path.join(output_dir, "road_debug.jpg")

    # ── Load
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot open: {args.image}")
    h0, w0 = img.shape[:2]
    print(f"\n[Input]  {args.image}  {w0}×{h0}")

    # ── EXIF + GPS
    exif = extract_exif(args.image)
    gps  = exif.get("GPSDecimal")
    if gps:
        print(f"[GPS]    {gps['latitude']}, {gps['longitude']}")

    # ── Intrinsics
    K = build_K(exif, w0, h0)
    print(f"[K]      fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # ── Downscale → undistort → get working dimensions
    img, K = maybe_downscale(img, K)
    if not args.no_undistort:
        img, K = undistort(img, K)
        print(f"[Undist] → {img.shape[1]}×{img.shape[0]}")
    h, w = img.shape[:2]

    # ── Edges & lines
    edges, roi_y = prepare_edges(img, roi_top_frac=args.roi_top,
                                 gaussian_k=args.gaussian_k, morph_k=args.morph_k,
                                 canny_lo=args.canny_lo, canny_hi=args.canny_hi,
                                 debug_edges=args.debug_edges)
    lines = detect_lines(edges, roi_y, img.shape)
    print(f"[Lines]  {len(lines)}")

    # ── Vanishing point
    if args.vp:
        vp          = np.array(args.vp, dtype=np.float64)
        inlier_mask = np.zeros(len(lines), dtype=bool)
        print(f"[VP] from args ({vp[0]:.0f}, {vp[1]:.0f})")
    elif args.manual_vp:
        vp          = pick_vp_manually(img)
        inlier_mask = np.zeros(len(lines), dtype=bool)
    else:
        vp, inlier_mask = ransac_vp(lines, h, w)

    # ── Trapezoid + warp
    src           = road_trapezoid(vp, h, w, near_spread=args.near_spread,
                                   far_spread=args.far_spread)
    warped, H_mat = warp_road(img, src, args.out_w, args.out_h)

    # ── Metric measurements for the four trapezoid corners
    # src order: TL=0, TR=1, BR=2, BL=3
    # The trapezoid is a projection of a real-world rectangle, so near and far
    # widths are equal. We measure from the near row (BL/BR) as it's closest
    # to the camera and has the most reliable depth estimate.
    corners_m = pixels_to_meters(K, vp, args.camera_height, src.reshape(-1, 2))

    def _dist(a, b):
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    road_w = _dist(corners_m[3], corners_m[2])   # BL → BR (near row)
    metrics = {
        "far_w":   road_w,
        "near_w":  road_w,
        "left_h":  _dist(corners_m[0], corners_m[3]),  # TL → BL
        "right_h": _dist(corners_m[1], corners_m[2]),  # TR → BR
    }

    print(f"\n── Ground Patch Metrics ────────────────────")
    print(f"  Camera height : {args.camera_height:.2f} m (assumed)")
    print(f"  Far width     : {metrics['far_w']:.3f} m")
    print(f"  Near width    : {metrics['near_w']:.3f} m")
    print(f"  Left length   : {metrics['left_h']:.3f} m")
    print(f"  Right length  : {metrics['right_h']:.3f} m")

    # ── Annotate + save
    warped = annotate_ortho(warped, metrics, args.out_w, args.out_h)
    cv2.imwrite(args.output, warped)
    print(f"\n[Output] {args.output}  ({args.out_w}×{args.out_h})")

    vis = annotate_debug(img.copy(), lines, vp, src, inlier_mask,
                         metrics, args.camera_height)
    cv2.imwrite(args.debug, vis)
    print(f"[Debug]  {args.debug}")

    print(f"\n── Summary ─────────────────────────────────")
    print(f"  VP position   : ({vp[0]:.0f}, {vp[1]:.0f})")
    print(f"  Lines used    : {len(lines)}")


if __name__ == "__main__":
    main()