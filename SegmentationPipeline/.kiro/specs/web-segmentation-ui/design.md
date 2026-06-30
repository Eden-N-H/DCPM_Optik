# Design Document: Web Segmentation UI

## Overview

A locally-deployed web application that wraps the existing road defect segmentation pipeline with a browser-based drag-and-drop interface. The system uses a Flask backend that pre-loads the YOLO and SAM2 models at startup, processes uploaded image batches sequentially, and reports progress to a vanilla HTML/CSS/JavaScript frontend via polling.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│  Browser (Frontend)                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  index.html + style.css + app.js              │  │
│  │  - Drop Zone for file/folder upload           │  │
│  │  - Progress Bar with status text              │  │
│  │  - Image Gallery with modal viewer            │  │
│  └───────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (localhost:5000)
┌────────────────────▼────────────────────────────────┐
│  Flask Backend (web_app.py)                          │
│  ┌───────────────────────────────────────────────┐  │
│  │  Routes:                                       │  │
│  │  GET  /            → serve index.html          │  │
│  │  GET  /static/*    → serve CSS/JS assets       │  │
│  │  POST /api/upload  → accept image batch        │  │
│  │  GET  /api/progress→ return processing status  │  │
│  │  GET  /api/results → return result image URLs  │  │
│  │  GET  /output/*    → serve annotated images    │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Pipeline Integration:                         │  │
│  │  - Pre-loaded YOLO + SAM2 models              │  │
│  │  - Reuses existing pipeline components         │  │
│  │  - Generates annotated images to temp output   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### File Structure

```
web/
├── web_app.py          # Flask backend server
├── static/
│   ├── style.css       # Frontend styles
│   └── app.js          # Frontend JavaScript logic
├── templates/
│   └── index.html      # Main HTML page
├── uploads/            # Temporary upload storage (gitignored)
└── output/             # Annotated result images (gitignored)
```

## Components and Interfaces

### Backend Components

#### FlaskApp (web/web_app.py)
- **Responsibility**: HTTP server, route handling, static file serving
- **Interface**: REST API endpoints (see API Design section)
- **Dependencies**: Flask, pipeline components

#### PipelineProcessor (web/web_app.py)
- **Responsibility**: Wraps existing pipeline components for per-image processing
- **Interface**: `process_image(image_path: str, output_dir: str) -> ProcessResult`
- **Dependencies**: Preprocessor, YOLODetector, SAM2Segmenter, PostSegmentationVerifier, MeasurementEngine

#### BatchManager (web/web_app.py)
- **Responsibility**: Manages batch state, progress tracking, background thread coordination
- **Interface**: 
  - `start_batch(image_paths: List[str]) -> str` (returns batch_id)
  - `get_progress(batch_id: str) -> ProgressState`
  - `get_results(batch_id: str) -> List[ImageResult]`
- **Dependencies**: PipelineProcessor, threading.Lock

### Frontend Components

#### DropZone (app.js)
- **Responsibility**: File/folder drag-and-drop handling, file filtering, upload initiation
- **Interface**: DOM event handlers on the drop zone element

#### ProgressTracker (app.js)
- **Responsibility**: Polling /api/progress, updating progress bar UI
- **Interface**: `startPolling(batchId)`, `stopPolling()`

#### ResultsGallery (app.js)
- **Responsibility**: Rendering result images, modal viewer
- **Interface**: `displayResults(results)`, `openModal(imageUrl)`

## Data Models

### ProcessResult
```python
@dataclass
class ProcessResult:
    filename: str
    success: bool
    output_path: Optional[str]  # Path to annotated image if success
    defects_found: int
    error: Optional[str]  # Error message if failed
```

### ProgressState
```python
@dataclass
class ProgressState:
    batch_id: str
    total: int
    completed: int
    failed: List[str]  # Filenames of failed images
    status: str  # "processing" | "complete"
```

### ImageResult
```python
@dataclass
class ImageResult:
    filename: str
    output_url: Optional[str]
    status: str  # "success" | "failed"
    defects_found: int
    error: Optional[str]
```

## API Design

### POST /api/upload

Accepts a multipart form upload with multiple image files.

**Request:** `multipart/form-data` with field name `images` (multiple files)

**Response (200):**
```json
{
  "batch_id": "uuid-string",
  "total_images": 4,
  "status": "processing"
}
```

**Response (400):**
```json
{
  "error": "No valid images provided"
}
```

**Response (409):**
```json
{
  "error": "A batch is already being processed"
}
```

### GET /api/progress?batch_id={id}

Returns the current processing progress for a batch.

**Response (200):**
```json
{
  "batch_id": "uuid-string",
  "total": 4,
  "completed": 2,
  "failed": [],
  "status": "processing"
}
```

### GET /api/results?batch_id={id}

Returns the list of result images after processing completes.

**Response (200):**
```json
{
  "batch_id": "uuid-string",
  "results": [
    {
      "filename": "image1.jpg",
      "output_url": "/output/batch-uuid/image1.jpg",
      "status": "success",
      "defects_found": 3
    }
  ]
}
```

### GET /output/{batch_id}/{filename}

Serves the annotated output image file directly.

## Error Handling

### Backend Error Handling

| Error Scenario | Handling Strategy |
|---|---|
| Model load failure at startup | Log error, exit with non-zero code |
| Invalid/corrupted image file | Skip image, mark as failed, continue batch |
| Image exceeds 50 MB limit | Skip image, mark as failed in progress |
| Pipeline exception during processing | Catch, log, mark image as failed, continue |
| Batch already processing (409) | Return conflict response, reject new batch |
| Unknown batch_id in progress/results | Return 404 response |

### Frontend Error Handling

| Error Scenario | Handling Strategy |
|---|---|
| Backend unreachable | Display "Server not available" error message |
| Network error during upload | Display error with retry option |
| Polling fails | Retry 3 times, then display error |
| No valid images dropped | Display "No valid images found" message |
| Batch exceeds 100 images | Display limit message, reject submission |

## Testing Strategy

### Unit Tests
- File extension filtering logic (property: only .jpg/.jpeg/.png retained)
- Batch size validation (property: >100 rejected, ≤100 accepted)
- Progress state management (property: completed monotonically non-decreasing)

### Integration Tests
- Upload endpoint accepts valid images and returns batch_id
- Progress endpoint reflects actual processing state
- Results endpoint returns correct output URLs
- End-to-end flow with test images

### Manual Testing
- Drag-and-drop behavior in Chrome/Firefox/Safari
- Modal viewer responsiveness
- Progress bar animation smoothness

## Correctness Properties

### Property 1: File Filter Correctness

For any collection of files with mixed extensions, the frontend file filter SHALL retain only files with extensions .jpg, .jpeg, or .png (case-insensitive) and reject all other files. The count of retained files SHALL equal the count of files in the original set whose extension matches the supported set.

**Validates: Requirements 3.2, 3.3**

### Property 2: Progress Monotonicity

For any batch of N images, the progress endpoint SHALL report a `completed` count that is monotonically non-decreasing over time, with `completed` always in the range [0, N] and `completed + len(failed)` never exceeding `total`.

**Validates: Requirements 5.2, 5.3**

### Property 3: Batch Resilience

For any batch containing a mix of valid and invalid images, the number of successfully processed results plus the number of failed results SHALL equal the total number of images submitted. No image SHALL be silently dropped.

**Validates: Requirements 4.3, 7.3**

### Property 4: Batch Size Enforcement

For any attempted submission with more than 100 images, the Frontend SHALL reject the submission before sending to the Backend. For any submission with 100 or fewer images, the submission SHALL proceed.

**Validates: Requirements 8.1, 8.2**

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend framework | Flask | Lightweight, well-suited for local tools, already in Python ecosystem |
| Frontend | Vanilla HTML/CSS/JS | No build step, simple deployment, user's stated preference |
| Background processing | `threading.Thread` | Simple for single-user local deployment, avoids Celery complexity |
| Progress mechanism | Polling (2s interval) | Simpler than WebSockets for this use case, sufficient update rate |
| Image serving | Flask static file serving | Direct file serving, no additional web server needed |

## Assumptions and Constraints

- Single user, local deployment only (no authentication needed)
- Models are already downloaded and available at configured paths
- Processing is sequential (one image at a time) due to GPU memory constraints
- The existing pipeline components are not modified; the web app wraps them
- Browser must support the File System Access API or `webkitdirectory` for folder drops
