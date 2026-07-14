"""Worker compute node that runs in Colab, communicates with orchestrator, and executes tasks sequentially."""

import sys
import os
import time
import subprocess
import requests
import zipfile
import shutil
import glob
from pathlib import Path


def extract_zip(zip_path: Path, extract_to: Path):
    """Safely extracts a ZIP directly to local high-speed storage."""
    if not zip_path.exists():
        print(f"Zip not found at {zip_path}. Ensure it is uploaded to the Shared Drive.")
        sys.exit(1)
        
    if not extract_to.exists():
        print(f"Extracting {zip_path.name} to {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Data already staged at {extract_to}. Skipping extraction.")


def get_latest_checkpoint(shared_drive_path: Path, prefix="") -> str:
    """Find the most recent checkpoint in the shared drive."""
    shared_drive_path.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(glob.glob(str(shared_drive_path / f"{prefix}*.pt")))
    return checkpoints[-1] if checkpoints else ""


def start_worker(orchestrator_url: str, shared_drive_path: str, worker_id: str):
    print(f"Starting Worker Node {worker_id}...")
    shared_drive_path = Path(shared_drive_path)
    local_data_path = Path("/content/data")
    
    while True:
        # 1. Ask Orchestrator for Task
        try:
            response = requests.post(f"{orchestrator_url}/api/relay/register", json={"worker_id": worker_id}, timeout=10)
            task_data = response.json()
        except Exception as e:
            print(f"Failed to reach Orchestrator: {e}")
            time.sleep(10)
            continue

        action = task_data.get("action")
        if action == "wait":
            print("No pending tasks. Waiting...")
            time.sleep(10)
            continue
            
        task = task_data.get("task")
        task_id = task_data.get("task_id")
        dataset_zip = task_data.get("dataset_zip", "dataset.zip")
        print(f"Received task: {task} (Queue ID: {task_id})")

        # 2. Smart Data Staging (Direct from Drive to Local NVMe)
        if task in ["train_cyclegan", "translate"]:
            extract_zip(shared_drive_path / dataset_zip, local_data_path)
        elif task == "train":
            trans_zip = shared_drive_path / "dataset_translated.zip"
            if trans_zip.exists():
                extract_zip(trans_zip, local_data_path / "road_quality_translated")
            else:
                extract_zip(shared_drive_path / dataset_zip, local_data_path)

        # 3. Construct Subprocess
        webhook_url = f"{orchestrator_url}/api/relay/telemetry"
        cmd = [
            sys.executable, "-m", "src.main", task,
            "--config", "configs/default.yaml",
            "--webhook-url", webhook_url,
            "--worker-id", worker_id
        ]

        # 4. Attach Task-Specific Flags
        if task == "train_cyclegan":
            ckpt_dir = shared_drive_path / "checkpoints" / "cyclegan"
            cmd.extend(["--output-dir", str(ckpt_dir)])
            latest = get_latest_checkpoint(ckpt_dir)
            if latest: cmd.extend(["--resume", latest])
            cmd.append(f"--data.root={str(local_data_path / 'road_quality')}")
            cmd.append(f"--real-data={str(shared_drive_path / 'real_images')}")
            
        elif task == "translate":
            ckpt_dir = shared_drive_path / "checkpoints" / "cyclegan"
            latest = get_latest_checkpoint(ckpt_dir)
            if latest: cmd.extend(["--checkpoint", latest])
            cmd.extend([
                "--input-dir", str(local_data_path / 'road_quality'),
                "--output-dir", str(local_data_path / 'road_quality_translated')
            ])
            
        elif task == "train":
            ckpt_dir = shared_drive_path / "checkpoints" / "multitask"
            cmd.extend(["--output-dir", str(ckpt_dir)])
            latest = get_latest_checkpoint(ckpt_dir)
            if latest: cmd.extend(["--resume", latest])
            
            trans_data = local_data_path / 'road_quality_translated'
            if trans_data.exists():
                cmd.append(f"--data.root={str(trans_data)}")
            else:
                cmd.append(f"--data.root={str(local_data_path / 'road_quality')}")

        print("Executing command:", " ".join(cmd))
        
        # 5. Run Execute
        proc = subprocess.Popen(cmd)
        proc.wait()
        
        # 6. Post-Task State Preservation
        if task == "translate" and proc.returncode == 0:
            print("Zipping translated dataset to Google Drive for safety...")
            shutil.make_archive(
                str(shared_drive_path / "dataset_translated"), 
                'zip', 
                str(local_data_path / "road_quality_translated")
            )
        
        # 7. Post Telemetry
        status = "completed" if proc.returncode == 0 else "failed"
        try:
            requests.post(webhook_url, json={"worker_id": worker_id, "task_id": task_id, "status": status}, timeout=5)
        except Exception:
            pass
