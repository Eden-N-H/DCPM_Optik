# Colab Hybrid Relay Workflow

This pipeline uses a Hybrid Relay Architecture. Do not manually upload code and run scripts in Colab. 

* **Local Machine (Orchestrator):** Runs the Flask UI, manages state, tracks telemetry.
* **Google Drive (Bridge):** Stores the zipped dataset and saves checkpoints.
* **Google Colab (Worker):** Provides the GPU, pulls data, pings your local machine for instructions, and runs the heavy lifting.

Follow these steps exactly.

### Step 1: Prepare & Upload Data
Reading raw images directly from Google Drive is incredibly slow. The worker expects a zipped dataset so it can extract it directly to Colab's high-speed NVMe storage.

1. Generate your dataset locally: 
   ```bash
   python -m src.main data --output-dir ./data/road_quality
   ```
2. Zip the `road_quality` folder and name it `dataset.zip`.
3. Create a folder in your Google Drive (e.g., `MyDrive/Shared_DCPM`).
4. Upload `dataset.zip` into that folder.

### Step 2: Start the Local Orchestrator
Your local machine needs to act as the command center.

1. Start the web UI:
   ```bash
   python -m src.main web --port 5000
   ```
2. Expose port 5000 to the internet using [Ngrok](https://ngrok.com/) (or localtunnel):
   ```bash
   ngrok http 5000
   ```
3. Copy the Ngrok Forwarding URL (e.g., `https://<your-id>.ngrok-free.app`).

### Step 3: Generate the Notebook
Generate the exact Colab environment setup file using the built-in CLI.

```bash
python -m src.main colab --output Colab_Pipeline.ipynb
```

### Step 4: Execute on Colab
1. Upload `Colab_Pipeline.ipynb` to Google Colab.
2. In Colab, change the runtime to **T4 GPU** (or better).
3. Scroll to the **Worker Configuration** cell.
4. Paste your Ngrok URL into `ORCHESTRATOR_URL`.
5. Enter your Drive folder path in `SHARED_DRIVE_PATH` (e.g., `/content/drive/MyDrive/Shared_DCPM`).
6. Click **Run All**.

---

### How the Worker Behaves (No-Touch Operation)
* **Auth:** It will prompt you to authorize Google Drive access.
* **Extraction:** It copies `dataset.zip` from Drive to `/content/data` and extracts it locally to bypass Drive I/O bottlenecks.
* **Telemetry:** It pings your local Ngrok URL. You will see training progress live on your local Flask dashboard (`http://localhost:5000`).
* **Auto-Resume:** If Colab disconnects, simply reconnect and Run All again. The worker automatically finds the latest `.pt` file in your Drive's `checkpoints/` folder and resumes training exactly where it left off.
