"""Module to programmatically generate a Google Colab notebook.

This notebook uses the Hybrid Relay architecture. It connects to the orchestrator,
mounts the Shared Google Drive, and executes the worker.py script.
"""

import json
from pathlib import Path


def create_markdown_cell(source: str) -> dict:
    """Create a markdown cell dictionary for a Jupyter Notebook."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }


def create_code_cell(source: str) -> dict:
    """Create a code cell dictionary for a Jupyter Notebook."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }


def generate_colab_notebook(output_path: str = "Colab_Pipeline.ipynb"):
    """Generates the .ipynb file containing the Relay Worker execution."""

    cells = []

    # Cell 1: Intro
    cells.append(create_markdown_cell(
        "# DCPM Optik - Relay Worker Node\n"
        "This notebook serves as a remote GPU worker for the DCPM Pipeline. "
        "It mounts your shared Google Drive, contacts your local orchestrator, "
        "and automatically resumes training where the last node left off.\n\n"
        "**Instructions:**\n"
        "1. Mount Drive.\n"
        "2. Paste your Ngrok Orchestrator URL.\n"
        "3. Run All."
    ))

    # Cell 2: Mount Drive & Setup Repo
    cells.append(create_code_cell(
        "from google.colab import drive\n"
        "import os\n"
        "import uuid\n\n"
        "# Mount Google Drive\n"
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

    # Cell 3: Form variables & Run Worker
    cells.append(create_markdown_cell("### Start Relay Worker"))
    cells.append(create_code_cell(
        "#@title Worker Configuration\n"
        "ORCHESTRATOR_URL = \"https://your-ngrok-url.ngrok.io\" #@param {type:\"string\"}\n"
        "SHARED_DRIVE_PATH = \"/content/drive/MyDrive/Shared_DCPM\" #@param {type:\"string\"}\n\n"
        "worker_id = f'colab-{str(uuid.uuid4())[:8]}'\n"
        "print(f'Starting Worker ID: {worker_id}')\n\n"
        "!python -m src.main worker --orchestrator-url {ORCHESTRATOR_URL} --shared-drive-path {SHARED_DRIVE_PATH} --worker-id {worker_id}"
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path(output_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Successfully generated Relay Colab notebook at: {out_path.absolute()}")
