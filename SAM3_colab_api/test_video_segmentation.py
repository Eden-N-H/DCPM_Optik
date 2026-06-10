# TODO: doesn't work







import os
import base64
import requests
from io import BytesIO
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "https://7f06-34-127-71-7.ngrok-free.app/" 
VIDEO_PATH = "assets/russia.mp4"
PROMPT = "person"
OUTPUT_DIR = "video_masks_output"

# ==========================================
# MAIN EXECUTION
# ==========================================
def run_video_tracking():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find video at '{VIDEO_PATH}'.")
        return

    # Create local directory to store result masks
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    headers = {
        "ngrok-skip-browser-warning": "true"
    }
    
    session_id = None
    endpoint_base = API_URL.rstrip('/')

    try:
        # 1. Start Tracking Session (Upload Video)
        print(f"1. Uploading {VIDEO_PATH} and initializing session...")
        upload_url = f"{endpoint_base}/video/start_session"
        
        with open(VIDEO_PATH, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(upload_url, headers=headers, files=files)
            
        if response.status_code != 200:
            print(f"Failed to start session. Server status: {response.status_code}")
            print(response.text)
            return

        res_data = response.json()
        if not res_data.get("success"):
            print("Server failed to initialize video:", res_data.get("error"))
            return

        session_id = res_data["session_id"]
        print(f"Session successfully initialized. Session ID: {session_id}")

        # 2. Add Prompt to Frame 0
        print(f"\n2. Adding prompt '{PROMPT}' to Frame 0...")
        prompt_url = f"{endpoint_base}/video/add_prompt"
        prompt_payload = {
            "session_id": session_id,
            "frame_index": 0,
            "text": PROMPT
        }
        
        response = requests.post(prompt_url, headers=headers, json=prompt_payload)
        if response.status_code != 200:
            print(f"Failed to add prompt. Server status: {response.status_code}")
            return
            
        res_data = response.json()
        if not res_data.get("success"):
            print("Server failed to register prompt:", res_data.get("error"))
            return

        obj_ids = res_data["outputs"].get("obj_ids", [])
        print(f"Prompt registered. Tracking object IDs: {obj_ids}")

        # 3. Propagate Tracking Across All Video Frames
        print("\n3. Propagating tracking across all video frames (this may take a moment)...")
        propagate_url = f"{endpoint_base}/video/propagate"
        
        response = requests.post(propagate_url, headers=headers, json={"session_id": session_id})
        if response.status_code != 200:
            print(f"Failed to propagate tracking. Server status: {response.status_code}")
            return

        res_data = response.json()
        if not res_data.get("success"):
            print("Propagation failed:", res_data.get("error"))
            return

        # 4. Save Output Masks
        results = res_data.get("results", [])
        print(f"Propagation complete. Saving masks for {len(results)} frames to '{OUTPUT_DIR}'...")

        for frame_entry in results:
            frame_idx = frame_entry["frame_index"]
            outputs = frame_entry["outputs"]
            frame_obj_ids = outputs.get("obj_ids", [])
            masks_b64 = outputs.get("masks_base64", [])

            for idx, mask_b64 in enumerate(masks_b64):
                obj_id = frame_obj_ids[idx] if idx < len(frame_obj_ids) else f"unknown_{idx}"
                
                # Decode and save mask image
                mask_data = base64.b64decode(mask_b64)
                mask_img = Image.open(BytesIO(mask_data))
                
                out_filename = f"frame_{frame_idx:04d}_obj_{obj_id}.png"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                mask_img.save(out_path)

        print(f"Successfully processed and saved frames.")

    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Failed to reach the server.")
    except Exception as e:
        print(f"\nAn error occurred during video processing: {str(e)}")

    finally:
        # 5. Clean Up Session (Always run to prevent Colab GPU Out Of Memory issues)
        if session_id:
            print(f"\n5. Cleaning up and closing session {session_id} on server...")
            close_url = f"{endpoint_base}/video/close_session"
            try:
                cleanup_resp = requests.post(close_url, headers=headers, json={"session_id": session_id})
                if cleanup_resp.status_code == 200:
                    cleanup_data = cleanup_resp.json()
                    gpu_mem = cleanup_data.get("gpu_mem", {})
                    free_pct = gpu_mem.get("free_pct", "Unknown")
                    print(f"Session closed successfully. Server free GPU VRAM: {free_pct}%")
                else:
                    print("Failed to gracefully close session on server.")
            except Exception as clean_err:
                print(f"Error during cleanup request: {str(clean_err)}")

if __name__ == "__main__":
    run_video_tracking()
    