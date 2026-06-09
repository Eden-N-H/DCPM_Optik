# SAM 3 Colab API Reference

This document outlines how to interact with your SAM 3 Colab API from a local machine. The API supports two distinct operating modes:

1. **Stateless Image PCS (Promptable Concept Segmentation):** Fast, one-off segmentation of static images.
2. **Stateful Video Tracking (SAM 3.1 Multiplex):** Upload a video, initialize a tracker state, add interactive prompts across frames, and propagate tracking across the entire video.

---

## Base Configuration

All requests must include the following header to bypass Ngrok's free-tier HTML warning screen:

```json
{
  "ngrok-skip-browser-warning": "true"
}
```

Ensure your base URL corresponds to the output of the Colab Ngrok cell (e.g., `https://<your-subdomain>.ngrok-free.app`).

---

## 1. Image API (Stateless)

### `POST /image/segment`

Performs one-off concept segmentation on a static image based on a text prompt.

**Content-Type:** `multipart/form-data`

**Parameters:**

- `image` (File): The target image file.
- `prompt` (String): The text prompt (e.g., "a red car").

**Response:**

```json
{
  "success": true,
  "boxes": [[10.5, 20.0, 50.0, 100.0], ...], // [x, y, w, h] format
  "scores": [0.95, ...],
  "masks_base64": ["iVBORw0KGgo...", ...] // Base64 encoded grayscale PNG strings
}
```

---

## 2. Video Tracking API (Stateful)

Video tracking requires opening a session, manipulating the memory states, and closing the session to prevent memory leaks (OOM errors) on the Colab GPU.

### `POST /video/start_session`

Uploads a video to the server and initializes the SAM 3.1 video tracker memory banks.

**Content-Type:** `multipart/form-data`

**Parameters:**

- `video` (File): An `.mp4` video file.

**Response:**

```json
{
  "success": true,
  "session_id": "uuid-string-here"
}
```

---

### `POST /video/add_prompt`

Adds a prompt (text, point, or box) to a specific frame to teach the model what to track.

**Content-Type:** `application/json`

**Payload:**

```json
{
  "session_id": "uuid-string-here",
  "frame_index": 0,
  "text": "the person on the left",
  "points": [
    [300, 450],
    [320, 460]
  ], // Optional: Absolute pixel coordinates
  "point_labels": [1, 0], // Optional: 1 for positive click, 0 for negative
  "obj_id": 1 // Optional: Required if refining an existing object with points
}
```

**Response:**

```json
{
  "success": true,
  "frame_index": 0,
  "outputs": {
      "obj_ids": [1],
      "boxes": [[...]],
      "scores": [0.98],
      "masks_base64": ["..."]
  }
}
```

---

### `POST /video/propagate`

Commands the tracker to propagate the learned prompts forwards and backwards across the entire video.

**Content-Type:** `application/json`

**Payload:**

```json
{
  "session_id": "uuid-string-here"
}
```

**Response:**
Returns an array of results for every frame in the video.

```json
{
  "success": true,
  "results": [
    {
      "frame_index": 0,
      "outputs": {
          "obj_ids": [1],
          "boxes": [[...]],
          "scores": [0.98],
          "masks_base64": ["..."]
      }
    },
    ...
  ]
}
```

---

### `POST /video/close_session` (⚠️ CRITICAL)

Closes the session, deletes the temporary video file from the server, and forcefully runs PyTorch garbage collection to free the GPU VRAM. **If you do not call this, your Colab GPU will crash out of memory after 2-3 videos.**

**Content-Type:** `application/json`

**Payload:**

```json
{
  "session_id": "uuid-string-here"
}
```

**Response:**

```json
{
  "success": true,
  "gpu_mem": {
    "free_bytes": 10485760000,
    "total_bytes": 16106127360,
    "free_pct": 65.1
  }
}
```

---

## Local Client Examples

### Example 1: Image Segmentation

```python
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "https://<your-subdomain>.ngrok-free.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}

def segment_image():
    files = {'image': open('dog.jpg', 'rb')}
    data = {'prompt': 'the brown dog'}

    res = requests.post(f"{API_URL}/image/segment", headers=HEADERS, files=files, data=data).json()

    for i, mask_b64 in enumerate(res['masks_base64']):
        mask_bytes = base64.b64decode(mask_b64)
        Image.open(BytesIO(mask_bytes)).save(f"image_mask_{i}.png")
        print(f"Saved Mask {i} | Score: {res['scores'][i]}")

if __name__ == "__main__":
    segment_image()
```

### Example 2: Complete Video Tracking Workflow

```python
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "https://<your-subdomain>.ngrok-free.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}

def track_video():
    # 1. Start Session
    print("Uploading video...")
    files = {'video': open('sample.mp4', 'rb')}
    res = requests.post(f"{API_URL}/video/start_session", headers=HEADERS, files=files).json()
    session_id = res['session_id']
    print(f"Session started: {session_id}")

    try:
        # 2. Add Prompt to Frame 0
        print("Adding prompt to frame 0...")
        payload = {
            "session_id": session_id,
            "frame_index": 0,
            "text": "the red car"
        }
        res = requests.post(f"{API_URL}/video/add_prompt", headers=HEADERS, json=payload).json()
        print(f"Prompt added. Found {len(res['outputs']['obj_ids'])} objects on frame 0.")

        # 3. Propagate across video
        print("Propagating tracking across all frames...")
        res = requests.post(f"{API_URL}/video/propagate", headers=HEADERS, json={"session_id": session_id}).json()

        for frame_data in res['results']:
            frame_idx = frame_data['frame_index']
            outputs = frame_data['outputs']

            # Save the mask for obj_id 1 if it exists
            if 1 in outputs['obj_ids']:
                list_idx = outputs['obj_ids'].index(1)
                mask_b64 = outputs['masks_base64'][list_idx]

                mask_bytes = base64.b64decode(mask_b64)
                Image.open(BytesIO(mask_bytes)).save(f"frame_{frame_idx:04d}_mask.png")

        print("Tracking complete and masks saved!")

    finally:
        # 4. ALWAYS close the session to free GPU VRAM!
        print("Closing session and cleaning up VRAM...")
        res = requests.post(f"{API_URL}/video/close_session", headers=HEADERS, json={"session_id": session_id}).json()
        print("VRAM stats:", res.get("gpu_mem", "Unknown"))

if __name__ == "__main__":
    track_video()
```
