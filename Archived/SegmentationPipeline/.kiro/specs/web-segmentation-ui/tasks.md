# Implementation Plan: Web Segmentation UI

## Overview

Build a local web application (Flask backend + vanilla HTML/CSS/JS frontend) that wraps the existing road defect segmentation pipeline with a drag-and-drop interface, progress tracking, and result gallery.

## Tasks

- [x] 1. Create Flask backend server with model pre-loading
  - [x] 1.1 Create `web/web_app.py` with Flask app initialization and project structure
  - [x] 1.2 Configure server to start on localhost:5000
  - [x] 1.3 Load YOLO and SAM2 models at startup using existing pipeline components
  - [x] 1.4 Log server URL and model load confirmation on successful startup
  - [x] 1.5 Exit with error and log message if model loading fails
  - [x] 1.6 Initialize Preprocessor, Verifier, and MeasurementEngine at startup

- [x] 2. Create frontend HTML/CSS template
  - [x] 2.1 Create `web/templates/index.html` with Drop Zone, Progress section, and Gallery section
  - [x] 2.2 Create `web/static/style.css` with all styling (no external CDN dependencies)
  - [x] 2.3 Display Drop Zone prominently in idle state with drag-and-drop instructions
  - [x] 2.4 Add hidden Progress section with progress bar element and status text
  - [x] 2.5 Add Gallery section with CSS grid layout and responsive columns
  - [x] 2.6 Add modal overlay component for full-resolution image viewing
  - [x] 2.7 Add Flask route `GET /` to serve index.html template
  - [x] 2.8 Verify Flask serves static CSS and JS assets at `/static/*`

- [x] 3. Implement drag-and-drop and file upload logic
  - [x] 3.1 Create `web/static/app.js` with Drop Zone event handlers (dragenter, dragover, dragleave, drop) and visual feedback
  - [x] 3.2 Filter dropped files to include only .jpg, .jpeg, .png extensions (case-insensitive)
  - [x] 3.3 Support folder drops via webkitGetAsEntry/webkitdirectory
  - [x] 3.4 Display "No valid images found" message if no supported images after filtering
  - [x] 3.5 Validate batch size client-side (reject if >100 images with user message)
  - [x] 3.6 Upload valid images via multipart FormData POST to /api/upload
  - [x] 3.7 Transition UI to processing state after successful upload initiation

- [x] 4. Implement upload API endpoint
  - [x] 4.1 Create `POST /api/upload` route accepting multipart form data with field name `images`
  - [x] 4.2 Return 400 if no supported image files provided
  - [x] 4.3 Return 400 if batch exceeds 100 images
  - [x] 4.4 Skip individual files exceeding 50 MB, mark as failed
  - [x] 4.5 Save valid uploaded images to `web/uploads/{batch_id}/` directory
  - [x] 4.6 Generate unique batch_id (UUID) and initialize progress tracking state
  - [x] 4.7 Start background thread for batch processing
  - [x] 4.8 Return 409 if a batch is already being processed
  - [x] 4.9 Return JSON response with batch_id, total_images, and status

- [x] 5. Implement background image processing
  - [x] 5.1 Process each image through Preprocessor → YOLO → SAM2 → Verifier in background thread
  - [x] 5.2 Generate annotated images with mask overlays, bounding boxes, contours, and labels (reuse run_and_visualize.py logic)
  - [x] 5.3 Annotate images with no detections with "No defects detected" label
  - [x] 5.4 Save output images to `web/output/{batch_id}/{filename}`
  - [x] 5.5 Update progress state (with thread lock) after each image completes
  - [x] 5.6 Log and record failed images without stopping the batch
  - [x] 5.7 Set batch status to "complete" when all images are processed
  - [x] 5.8 Clean up upload directory after processing completes

- [x] 6. Implement progress and results API endpoints
  - [x] 6.1 Create `GET /api/progress?batch_id={id}` returning JSON with total, completed, failed, and status
  - [x] 6.2 Return status "processing" while in progress, "complete" when done
  - [x] 6.3 Create `GET /api/results?batch_id={id}` returning result details for each image
  - [x] 6.4 Include filename, output_url, status (success/failed), and defects_found in results
  - [x] 6.5 Create `GET /output/{batch_id}/{filename}` route to serve annotated output images
  - [x] 6.6 Return 404 for unknown batch_id values

- [x] 7. Implement frontend progress polling and results display
  - [x] 7.1 Poll `/api/progress` every 2 seconds after upload completes
  - [x] 7.2 Update progress bar to reflect current completion percentage
  - [x] 7.3 Display "X of Y images processed" text label alongside progress bar
  - [x] 7.4 Stop polling and transition to results view when status is "complete"
  - [x] 7.5 Fetch `/api/results` and populate gallery with image thumbnails
  - [x] 7.6 Open full-resolution image in modal on thumbnail click
  - [x] 7.7 Close modal via close button, backdrop click, or Escape key
  - [x] 7.8 Show failed images in gallery with error indicator and message
  - [x] 7.9 Display user-friendly error messages for network errors during upload/polling
  - [x] 7.10 Add "Process Another Batch" button that resets UI to idle state

- [x] 8. Integration testing and project cleanup
  - [x] 8.1 Verify end-to-end flow: drop images → progress updates → results displayed
  - [x] 8.2 Create `web/README.md` with setup and usage instructions
  - [x] 8.3 Add Flask to project dependencies (pyproject.toml or requirements.txt)
  - [x] 8.4 Update `.gitignore` to exclude `web/uploads/` and `web/output/` directories
  - [x] 8.5 Verify server handles graceful shutdown (Ctrl+C) without orphan threads

## Task Dependency Graph

```json
{
  "waves": [
    {"tasks": ["1", "2", "3"]},
    {"tasks": ["4"]},
    {"tasks": ["5"]},
    {"tasks": ["6"]},
    {"tasks": ["7"]},
    {"tasks": ["8"]}
  ]
}
```

Tasks 1, 2, and 3 can be developed in parallel (wave 1). Task 4 depends on Task 1. Task 5 depends on Task 4. Task 6 depends on Task 5. Task 7 depends on Tasks 3 and 6. Task 8 depends on all other tasks.

## Notes

- The backend reuses existing pipeline components without modification
- Models are loaded once at startup and shared across all requests
- Only one batch can be processed at a time (single-user local deployment)
- Output images are served directly by Flask (no separate static file server needed)
- The `web/` directory is self-contained and does not modify existing project files
