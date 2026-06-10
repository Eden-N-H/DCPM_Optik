import os
import base64
import requests
from io import BytesIO
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "https://13d3-34-127-71-7.ngrok-free.app/" 
IMAGE_PATH = "assets/dog.jpg"
PROMPT = "dog"
OUTPUT_MASK_PREFIX = "dog_mask"

# ==========================================
# MAIN EXECUTION
# ==========================================
def run_segmentation():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at '{IMAGE_PATH}'.")
        return

    headers = {
        "ngrok-skip-browser-warning": "true"
    }
    
    endpoint = f"{API_URL.rstrip('/')}/image/segment"
    print(f"Target Image: {IMAGE_PATH}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Connecting to: {endpoint}")

    try:
        with open(IMAGE_PATH, 'rb') as img_file:
            files = {'image': img_file}
            data = {'prompt': PROMPT}
            
            response = requests.post(endpoint, headers=headers, files=files, data=data)
            
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print("Response payload:", response.text)
            return

        result = response.json()
        
        if not result.get("success"):
            print("API Error:", result.get("error", "Unknown error occurred"))
            return

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        masks_b64 = result.get("masks_base64", [])

        print(f"\nSuccess! Segmented {len(masks_b64)} target(s).")

        for idx, mask_b64 in enumerate(masks_b64):
            score = scores[idx] if idx < len(scores) else "N/A"
            box = boxes[idx] if idx < len(boxes) else "N/A"
            
            # Safe numeric check and formatting
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"Target {idx}: Confidence Score = {score_str} | Box = {box}")

            # Decode the base64 PNG mask
            mask_data = base64.b64decode(mask_b64)
            mask_img = Image.open(BytesIO(mask_data))
            
            out_path = f"{OUTPUT_MASK_PREFIX}_{idx}.png"
            mask_img.save(out_path)
            print(f"  -> Saved binary mask to: {out_path}")

    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Failed to reach the server.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")

if __name__ == "__main__":
    run_segmentation()