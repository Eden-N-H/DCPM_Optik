"""Flask Web UI Backend - SQLite-backed Subprocess & Relay Dispatcher."""

import os
import sys
import uuid
import time
import json
import sqlite3
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

from src.utils.config import ConfigLoader

app = Flask(__name__)

TASKS_DIR = Path(".tasks")
CHECKPOINTS_DIR = Path("checkpoints")
DATA_DIR = Path("data")
DB_PATH = TASKS_DIR / "tasks.db"

def init_db():
    """Initialize the SQLite database for task tracking and relay orchestration."""
    TASKS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Local UI tasks
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            pid INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    # Relay workers registry
    c.execute('''
        CREATE TABLE IF NOT EXISTS relay_workers (
            worker_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            current_epoch INTEGER,
            train_loss REAL,
            val_loss REAL,
            last_ping TEXT NOT NULL
        )
    ''')
    
    # Relay state (What the remote workers should do)
    c.execute('''
        CREATE TABLE IF NOT EXISTS relay_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            active_task TEXT,
            dataset_zip TEXT,
            target_epochs INTEGER
        )
    ''')
    # Insert default state
    c.execute("INSERT OR IGNORE INTO relay_state (id, active_task, dataset_zip, target_epochs) VALUES (1, 'train', 'dataset.zip', 200)")
    
    conn.commit()
    conn.close()

def update_task_status(task_id: str, status: str, pid: int = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    if pid is not None:
        c.execute("UPDATE tasks SET status = ?, pid = ?, updated_at = ? WHERE id = ?", (status, pid, now, task_id))
    else:
        c.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?", (status, now, task_id))
    conn.commit()
    conn.close()

def expand_dot_notation(flat_dict):
    """Convert flat dict with dot.notation.keys to a nested dictionary."""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split('.')
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convert numeric strings where appropriate
        if isinstance(value, str):
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        current[parts[-1]] = value
    return nested

def run_subprocess(task_id: str, cmd: list, workspace: Path):
    """Background thread to monitor the subprocess and update the DB upon exit."""
    log_file = workspace / "stdout.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=f, stderr=subprocess.STDOUT)
        update_task_status(task_id, "running", proc.pid)
        proc.wait()
        
    final_status = "completed" if proc.returncode == 0 else "failed"
    update_task_status(task_id, final_status)


# ---------------------------------------------------------------------------
# HTML Page Routes
# ---------------------------------------------------------------------------

@app.route('/')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get local tasks
    c.execute("SELECT id, type, status, created_at FROM tasks ORDER BY created_at DESC")
    tasks = [{"id": row[0], "type": row[1], "status": row[2], "created": row[3]} for row in c.fetchall()]
    
    # Get relay workers
    c.execute("SELECT worker_id, status, current_epoch, train_loss, last_ping FROM relay_workers ORDER BY last_ping DESC")
    workers = [{"id": r[0], "status": r[1], "epoch": r[2], "loss": r[3], "ping": r[4]} for r in c.fetchall()]
    
    conn.close()
    
    # Renders the HTML template and passes both local tasks and remote workers
    return render_template('index.html', tasks=tasks, relay_workers=workers)

@app.route('/data')
def data_view():
    return render_template('data.html')

@app.route('/train')
def train_view():
    checkpoints = [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]
    return render_template('train.html', checkpoints=checkpoints)

@app.route('/evaluate')
def evaluate_view():
    checkpoints = [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]
    return render_template('evaluate.html', checkpoints=checkpoints)

@app.route('/reconstruct')
def reconstruct_view():
    checkpoints = [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]
    return render_template('reconstruct.html', checkpoints=checkpoints)

@app.route('/visualize')
def visualize_view():
    checkpoints = [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]
    return render_template('visualize.html', checkpoints=checkpoints)

@app.route('/quicktest')
def quicktest_view():
    checkpoints = [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]
    return render_template('quicktest.html', checkpoints=checkpoints)

@app.route('/task/<task_id>')
def task_view(task_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT type, status, created_at FROM tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "Task not found", 404
    return render_template('task.html', task_id=task_id, type=row[0], status=row[1], created=row[2])


# ---------------------------------------------------------------------------
# Relay Orchestrator API Routes
# ---------------------------------------------------------------------------

@app.route('/api/relay/register', methods=['POST'])
def relay_register():
    """Colab worker asks for instructions."""
    data = request.json
    worker_id = data.get("worker_id")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO relay_workers (worker_id, status, current_epoch, train_loss, val_loss, last_ping) VALUES (?, ?, 0, 0, 0, ?)",
              (worker_id, "registered", datetime.now().isoformat()))
              
    c.execute("SELECT active_task, dataset_zip FROM relay_state WHERE id = 1")
    state = c.fetchone()
    conn.commit()
    conn.close()
    
    if not state or not state[0]:
        return jsonify({"action": "wait"})
        
    return jsonify({
        "action": "run",
        "task": state[0],
        "dataset_zip": state[1],
    })

@app.route('/api/relay/telemetry', methods=['POST'])
def relay_telemetry():
    """Colab worker reports progress or interruption."""
    data = request.json
    worker_id = data.get("worker_id")
    status = data.get("status")
    epoch = data.get("epoch", 0)
    train_loss = data.get("train_loss", 0.0)
    val_loss = data.get("val_loss", 0.0)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE relay_workers 
        SET status = ?, current_epoch = ?, train_loss = ?, val_loss = ?, last_ping = ?
        WHERE worker_id = ?
    ''', (status, epoch, train_loss, val_loss, datetime.now().isoformat(), worker_id))
    conn.commit()
    conn.close()
    
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Action API Routes (Spawning Local Subprocesses)
# ---------------------------------------------------------------------------

@app.route('/api/run/data', methods=['POST'])
def run_data():
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    form_data = request.form.to_dict()
    nested_overrides = expand_dot_notation(form_data)
    
    config = ConfigLoader()
    merged = config._deep_merge(config.config, nested_overrides)
    
    config_path = workspace / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(merged, f)
        
    output_dir = DATA_DIR / "road_quality"
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "data", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [sys.executable, "-m", "src.main", "data", "--config", str(config_path), "--output-dir", str(output_dir)]
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    
    return jsonify({"task_id": task_id})

@app.route('/api/run/train', methods=['POST'])
def run_train():
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    form_data = request.form.to_dict()
    resume_ckpt = form_data.pop("resume_checkpoint", "")
    
    nested_overrides = expand_dot_notation(form_data)
    config = ConfigLoader()
    merged = config._deep_merge(config.config, nested_overrides)
    
    config_path = workspace / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(merged, f)
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "train", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [sys.executable, "-m", "src.main", "train", "--config", str(config_path), "--output-dir", str(CHECKPOINTS_DIR)]
    if resume_ckpt:
        cmd.extend(["--resume", str(CHECKPOINTS_DIR / resume_ckpt)])
        
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route('/api/run/evaluate', methods=['POST'])
def run_evaluate():
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    form_data = request.form.to_dict()
    checkpoint = form_data.get("checkpoint", "")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "evaluate", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [sys.executable, "-m", "src.main", "evaluate", "--config", "configs/default.yaml", "--checkpoint", str(CHECKPOINTS_DIR / checkpoint)]
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route('/api/run/reconstruct', methods=['POST'])
def run_reconstruct():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    checkpoint = request.form.get("checkpoint", "")
    
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    filepath = workspace / secure_filename(file.filename)
    file.save(str(filepath))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "reconstruct", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [
        sys.executable, "-m", "src.main", "reconstruct", 
        "--config", "configs/default.yaml", 
        "--checkpoint", str(CHECKPOINTS_DIR / checkpoint),
        "--input", str(filepath),
        "--output", str(workspace)
    ]
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route('/api/run/visualize', methods=['POST'])
def run_visualize():
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    form_data = request.form.to_dict()
    cg_ckpt = form_data.get("cyclegan_ckpt", "")
    mt_ckpt = form_data.get("multitask_ckpt", "")
    samples = form_data.get("samples", "5")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "visualize", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [
        sys.executable, "-m", "src.main", "visualize", 
        "--config", "configs/default.yaml", 
        "--multitask-ckpt", str(CHECKPOINTS_DIR / mt_ckpt),
        "--samples", str(samples),
        "--output-dir", str(workspace)
    ]
    if cg_ckpt:
        cmd.extend(["--cyclegan-ckpt", str(CHECKPOINTS_DIR / cg_ckpt)])
        
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route('/api/run/quicktest', methods=['POST'])
def run_quicktest():
    task_id = str(uuid.uuid4())
    workspace = TASKS_DIR / task_id
    workspace.mkdir()
    
    form_data = request.form.to_dict()
    cg_ckpt = form_data.get("cyclegan_ckpt", "")
    mt_ckpt = form_data.get("multitask_ckpt", "")
    samples = form_data.get("samples", "5")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, "quicktest", "queued", datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    cmd = [
        sys.executable, "-m", "src.main", "quicktest", 
        "--config", "configs/default.yaml", 
        "--samples", str(samples),
        "--output-dir", str(workspace)
    ]
    if cg_ckpt:
        cmd.extend(["--cyclegan-ckpt", str(CHECKPOINTS_DIR / cg_ckpt)])
    if mt_ckpt:
        cmd.extend(["--multitask-ckpt", str(CHECKPOINTS_DIR / mt_ckpt)])
        
    threading.Thread(target=run_subprocess, args=(task_id, cmd, workspace), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route('/api/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT pid, status FROM tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    if row and row[0] and row[1] == "running":
        try:
            import signal
            os.kill(row[0], signal.SIGTERM)
            update_task_status(task_id, "failed")
        except Exception:
            update_task_status(task_id, "completed")
    conn.close()
    return jsonify({"success": True})

@app.route('/api/tasks/<task_id>/files')
def list_task_files(task_id):
    """Return a list of .png files generated by a task (used by visualization page)."""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return jsonify({"files": []})
    files = sorted([f.name for f in task_dir.glob("*.png")])
    return jsonify({"files": files})


# ---------------------------------------------------------------------------
# Streaming & Downloads
# ---------------------------------------------------------------------------

@app.route('/stream/logs/<task_id>')
def stream_logs(task_id):
    """SSE endpoint to stream the stdout.log file."""
    log_file = TASKS_DIR / task_id / "stdout.log"
    
    def generate():
        wait_time = 0
        while not log_file.exists() and wait_time < 5:
            time.sleep(0.5)
            wait_time += 0.5
            
        if not log_file.exists():
            yield f"data: {json.dumps({'log': 'Log file not found.\\n', 'status': 'failed'})}\n\n"
            return
            
        with open(log_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
                    row = c.fetchone()
                    conn.close()
                    
                    status = row[0] if row else "unknown"
                    if status in ["completed", "failed"]:
                        yield f"data: {json.dumps({'log': '', 'status': status})}\n\n"
                        break
                        
                    time.sleep(0.5)
                    continue
                yield f"data: {json.dumps({'log': line, 'status': 'running'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    safe_filename = secure_filename(filename)
    directory = os.path.abspath(TASKS_DIR / task_id)
    return send_from_directory(directory, safe_filename)


def start_server(args):
    init_db()
    app.template_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    app.run(host=args.host, port=args.port, debug=False)
