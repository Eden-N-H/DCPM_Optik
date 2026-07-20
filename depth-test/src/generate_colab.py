"""Module to programmatically generate a Google Colab notebook."""
import json
from pathlib import Path

def create_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split('\n')]}

def create_code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.split('\n')]}

def generate_colab_notebook(output_path: str = "Colab_Pipeline.ipynb"):
    cells = []
    cells.append(create_markdown_cell(
        "# DCPM Optik - Automated Relay Worker Node\n"
        "This notebook acts as an infinite-loop remote GPU worker. It asks your local orchestrator "
        "for the next step in the pipeline (CycleGAN Training, Dataset Translation, DeepLabV3+ Training), "
        "processes it, and automatically writes the outputs back to your Google Drive.\n\n"
        "**Instructions:**\n"
        "1. Ensure you have `dataset.zip` and a `real_images/` folder in your Shared Drive.\n"
        "2. Paste your Ngrok Orchestrator URL.\n"
        "3. Run All."
    ))

    cells.append(create_code_cell(
        "from google.colab import drive\n"
        "import os\nimport uuid\n\n"
        "drive.mount('/content/drive')\n\n"
        "WORKSPACE_DIR = '/content/DCPM_Optik_Workspace'\n"
        "REPO_URL = 'https://github.com/Eden-N-H/DCPM_Optik.git'\n\n"
        "os.makedirs(WORKSPACE_DIR, exist_ok=True)\n"
        "os.chdir(WORKSPACE_DIR)\n\n"
        "if not os.path.exists('DCPM_Optik'):\n"
        "    !git clone {REPO_URL}\n"
        "else:\n"
        "    %cd DCPM_Optik\n"
        "    !git pull\n"
        "    %cd ..\n\n"
        "os.chdir('DCPM_Optik/depth-test')\n"
        "!pip install -r requirements.txt"
    ))

    cells.append(create_code_cell(
        "#@title Worker Configuration\n"
        "ORCHESTRATOR_URL = \"https://your-ngrok-url.ngrok.io\" #@param {type:\"string\"}\n"
        "SHARED_DRIVE_PATH = \"/content/drive/MyDrive/Shared_DCPM\" #@param {type:\"string\"}\n\n"
        "worker_id = f'colab-{str(uuid.uuid4())[:8]}'\n\n"
        "!python -m src.main worker --orchestrator-url {ORCHESTRATOR_URL} --shared-drive-path {SHARED_DRIVE_PATH} --worker-id {worker_id}"
    ))

    notebook = {
        "cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 4
    }

    out_path = Path(output_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
