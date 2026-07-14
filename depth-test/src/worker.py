"""Worker client that runs in Colab, communicates with orchestrator, and executes tasks.

Responsible for:
1. Contacting the orchestrator for task assignments.
2. Zipping/Unzipping datasets from Google Drive to local NVMe for fast I/O.
3. Spawning the main.py subprocess.
"""

import sys
import os
import time
import subprocess
import requests
import zipfile
import shutil
import glob
from pathlib import Path


def extract_dataset(shared_drive_path: Path, dataset_zip_name: str, local_data_path: Path):
    """Copy dataset from Drive and extract to local storage for speed."""
    zip_path = shared_drive_path / dataset_zip_name
    
    if not zip_path.exists():
        print(f"Dataset zip not found at {zip_path}. Ensure it is uploaded to the Shared Drive.")
        sys.exit(1)
        
    if not local_data_path.exists():
        print(f"Copying and extracting {dataset_zip_name} to local NVMe...")
        local_data_path.mkdir(parents=True, exist_ok=True)
        # Extract directly from Drive to local
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_data_path)
        print("Extraction complete.")
    else:
        print("Local dataset already exists. Skipping extraction.")


def get_latest_checkpoint(shared_drive_path: Path) -> str:
    """Find the most recent checkpoint in the shared drive."""
    ckpt_dir = shared_drive_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    checkpoints = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pt")))
    if checkpoints:
        return checkpoints[-1]
    return ""


def start_worker(orchestrator_url: str, shared_drive_path: str, worker_id: str, local_data_path: str = "data"):
    print(f"Starting Worker {worker_id}...")
    shared_drive_path = Path(shared_drive_path)
    local_data_path = Path(local_data_path)
    
    # 1. Register with Orchestrator
    try:
        response = requests.post(
            f"{orchestrator_url}/api/relay/register", 
            json={"worker_id": worker_id},
            timeout=10
        )
        task_data = response.json()
    except Exception as e:
        print(f"Failed to reach Orchestrator: {e}")
        print("Will attempt offline fallback using Drive files directly.")
        task_data = {"action": "run", "task": "train", "dataset_zip": "dataset.zip"}

    if task_data.get("action") == "wait":
        print("Orchestrator says Wait. Exiting.")
        sys.exit(0)

    # 2. Stage Data
    dataset_zip = task_data.get("dataset_zip", "dataset.zip")
    extract_dataset(shared_drive_path, dataset_zip, local_data_path)

    # 3. Find Resume Checkpoint
    latest_ckpt = get_latest_checkpoint(shared_drive_path)
    resume_flag = ["--resume", latest_ckpt] if latest_ckpt else []

    # 4. Construct Subprocess Command
    webhook_url = f"{orchestrator_url}/api/relay/telemetry"
    ckpt_out_dir = str(shared_drive_path / "checkpoints")
    
    cmd = [
        sys.executable, "-m", "src.main", task_data.get("task", "train"),
        "--config", "configs/default.yaml",
        "--output-dir", ckpt_out_dir,
        "--webhook-url", webhook_url,
        "--worker-id", worker_id
    ] + resume_flag

    # Add data path override so it uses the fast local NVMe copy
    cmd.append(f"--data.root={str(local_data_path / 'road_quality')}")

    print("Running command:", " ".join(cmd))
    
    # 5. Execute
    proc = subprocess.Popen(cmd)
    proc.wait()
    
    print(f"Process ended with code {proc.returncode}")
