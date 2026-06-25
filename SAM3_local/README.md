# SAM 3 YOLO Auto-Labeler (Local Version)

A local version of the SAM 3 annotation tool that runs entirely on your machine — no Google Colab needed.

## Requirements

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (RTX 3060 or better recommended)
- CUDA toolkit installed
- Hugging Face account with SAM 3 model access

## Setup

### 1. Install Dependencies

```bash
cd SAM3_local
pip install -r requirements.txt
```

### 2. Clone and Install SAM 3

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

### 3. Set Environment Variables

Create a `.env` file in this folder (or set these in your terminal):

```
HF_TOKEN=your_huggingface_token_here
```

You can also set them as system environment variables.

### 4. Configure Data Paths

Edit `config.json` to set your input/output directories:

```json
{
    "input_dir": "path/to/your/raw_images",
    "output_dir": "path/to/your/dataset",
    "classes": [...]
}
```

### 5. Run the Server

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

## Features

- **Annotation Studio**: Interactive polygon annotation with SAM 3 auto-labeling
- **Data Management**: Upload, organize, rename, move, copy, and delete images
- **Class Ontology**: Define YOLO classes with text prompts and colors
- **Bulk Processing**: Auto-label entire queues with progress tracking
- **YOLO Export**: Saves annotations in YOLO polygon format (normalized coordinates)

## Folder Structure

```
SAM3_local/
├── app.py              # Flask backend + SAM 3 inference
├── config.json         # Default configuration
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI
└── data/
    ├── raw_images/     # Default input directory
    └── dataset/        # Default output (YOLO format)
        ├── images/
        └── labels/
```
