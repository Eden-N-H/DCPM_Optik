import traceback
import torch

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("[+] SAM2 package imported successfully.")
except ImportError as e:
    print(f"[-] IMPORT ERROR: {e}")
    print("    Did you run: pip install git+https://github.com/facebookresearch/sam2.git ?")
    exit(1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[*] Using device: {device}")

# Test 1: Short config name (Standard SAM2 behavior)
try:
    print("[*] Attempting to load with config='sam2.1_hiera_l.yaml'...")
    model = build_sam2(config_file="sam2.1_hiera_l.yaml", ckpt_path="models/sam2.1_hiera_large.pt", device=device)
    print("[+] SUCCESS with short config!")
    exit(0)
except Exception as e:
    print(f"[-] FAILED with short config: {e}")

# Test 2: Long config name (Fallback)
try:
    print("\n[*] Attempting to load with config='configs/sam2.1/sam2.1_hiera_l.yaml'...")
    model = build_sam2(config_file="configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="models/sam2.1_hiera_large.pt", device=device)
    print("[+] SUCCESS with long config!")
    exit(0)
except Exception as e:
    print(f"[-] FAILED with long config.")
    print("\n--- FULL TRACEBACK ---")
    traceback.print_exc()
