"""Module to programmatically generate a Google Colab notebook.

This notebook is configured to run the full pipeline on a Colab T4 GPU,
using Google Drive for persistent storage of the large dataset and checkpoints,
and incorporating auto-resume logic for long training sessions.
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
    """Generates the .ipynb file containing the end-to-end pipeline execution."""

    cells = []

    # Cell 1: Intro
    cells.append(create_markdown_cell(
        "# DCPM Optik - Road Quality Analysis Pipeline\n"
        "This notebook runs the full synthetic data generation, multi-task training, "
        "and evaluation pipeline.\n\n"
        "**Note:** Because the default configuration generates 16,036 images and trains for 200 epochs, "
        "this notebook mounts your Google Drive to ensure persistent storage. If your Colab session "
        "disconnects (e.g., hitting the 12-hour limit), simply reconnect and **Run All** again. "
        "The training cell is programmed to automatically resume from the latest checkpoint."
    ))

    # Cell 2: Mount Drive & Setup Repo
    cells.append(create_code_cell(
        "from google.colab import drive\n"
        "import os\n\n"
        "# Mount Google Drive for persistent storage\n"
        "drive.mount('/content/drive')\n\n"
        "WORKSPACE_DIR = '/content/drive/MyDrive/DCPM_Optik_Workspace'\n"
        "REPO_URL = 'https://github.com/Eden-N-H/DCPM_Optik.git'\n\n"
        "os.makedirs(WORKSPACE_DIR, exist_ok=True)\n"
        "os.chdir(WORKSPACE_DIR)\n\n"
        "# Clone or update the repository\n"
        "if not os.path.exists('DCPM_Optik'):\n"
        "    !git clone {REPO_URL}\n"
        "else:\n"
        "    %cd DCPM_Optik\n"
        "    !git pull\n"
        "    %cd ..\n\n"
        "# Change to the correct sub-directory\n"
        "os.chdir('DCPM_Optik/depth-test')\n\n"
        "# Install python dependencies\n"
        "!pip install -r requirements.txt"
    ))

    # Cell 3: Install Blender 4.0
    cells.append(create_markdown_cell("### System Dependencies (Blender)\nThe dataset builder requires Blender to render scenes. We download the portable 4.0 binary to avoid outdated `apt-get` packages."))
    cells.append(create_code_cell(
        "import os\n\n"
        "# Download and extract Blender 4.0 to local Colab storage (fast, ephemeral)\n"
        "BLENDER_DIR = '/content/blender-4.0.2-linux-x64'\n"
        "if not os.path.exists(BLENDER_DIR):\n"
        "    !wget -nc https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz -O /content/blender.tar.xz\n"
        "    !tar -xf /content/blender.tar.xz -C /content/\n\n"
        "# Add Blender to system PATH so the subprocess in dataset_builder.py can find it\n"
        "os.environ['PATH'] += f':{BLENDER_DIR}'\n"
        "!blender --version"
    ))

    # Cell 4: Data Generation
    cells.append(create_markdown_cell("### 1. Data Generation\nGenerates the full 16,036 synthetic dataset. If the dataset already exists in your Drive, you can skip this cell to save time."))
    cells.append(create_code_cell(
        "!python -m src.main data --config configs/default.yaml --output-dir ./data/road_quality"
    ))

    # Cell 5: Training with Auto-Resume
    cells.append(create_markdown_cell("### 2. Multi-Task Model Training\nExecutes the domain adaptation and multi-task learning. Automatically detects and resumes from the latest checkpoint if interrupted."))
    cells.append(create_code_cell(
        "import os\n"
        "import glob\n\n"
        "# Find the latest checkpoint to resume from\n"
        "checkpoints = sorted(glob.glob('./checkpoints/checkpoint_epoch_*.pt'))\n"
        "resume_flag = f'--resume {checkpoints[-1]}' if checkpoints else ''\n\n"
        "if resume_flag:\n"
        "    print(f'Resuming training from: {checkpoints[-1]}')\n\n"
        "!python -m src.main train --config configs/default.yaml {resume_flag}"
    ))

    # Cell 6: Evaluation
    cells.append(create_markdown_cell("### 3. Evaluation\nEvaluates the best model on the test split."))
    cells.append(create_code_cell(
        "!python -m src.main evaluate --config configs/default.yaml --checkpoint ./checkpoints/best_model.pt"
    ))

    # Cell 7: 3D Reconstruction
    cells.append(create_markdown_cell("### 4. 3D Reconstruction\nRun the SLAM/BEV mapping reconstruction pipeline on the generated test set images."))
    cells.append(create_code_cell(
        "!python -m src.main reconstruct --config configs/default.yaml --checkpoint ./checkpoints/best_model.pt --input ./data/road_quality/test/dashcam --output ./reconstruction_output"
    ))

    # Cell 8: Visualization
    cells.append(create_markdown_cell("### 5. Visualization\nGenerates storyboard grids comparing Ground Truth, translated CycleGAN outputs, and Multi-Task predictions."))
    cells.append(create_code_cell(
        "!python -m src.main visualize --config configs/default.yaml --multitask-ckpt ./checkpoints/best_model.pt --samples 5 --output-dir ./visualizations"
    ))

    # Cell 9: Display Results in Colab
    cells.append(create_markdown_cell("### View Results"))
    cells.append(create_code_cell(
        "import glob\n"
        "from IPython.display import Image, display\n\n"
        "vis_files = sorted(glob.glob('./visualizations/grid_*.png'))\n"
        "for f in vis_files:\n"
        "    display(Image(filename=f))\n"
        "    print('-' * 100)"
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
    
    print(f"Successfully generated Colab notebook at: {out_path.absolute()}")

