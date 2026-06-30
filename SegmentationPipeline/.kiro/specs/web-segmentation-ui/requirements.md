# Requirements Document

## Introduction

A locally-deployed web application that provides a browser-based interface for the existing road defect segmentation pipeline. Users can drag and drop a folder of images into the interface, which processes each image through the YOLO detection and SAM2 segmentation pipeline, displays a progress bar during processing, and presents the annotated output images with segmentation mask overlays.

## Glossary

- **Web_App**: The local web application consisting of a Python backend server and an HTML/JavaScript frontend served at localhost.
- **Backend**: A Python HTTP server (Flask or FastAPI) that serves the frontend static files and provides REST API endpoints for image upload and processing.
- **Frontend**: The browser-based user interface built with vanilla HTML, CSS, and JavaScript, served by the Backend.
- **Pipeline**: The existing road defect segmentation pipeline comprising YOLO detection and SAM2 segmentation stages.
- **Progress_Bar**: A visual UI element in the Frontend that indicates the percentage of images processed in the current batch.
- **Batch**: A collection of images submitted together via a single drag-and-drop operation.
- **Annotated_Image**: An output image with segmentation masks, bounding boxes, contours, and class labels overlaid on the original image.
- **Drop_Zone**: The designated area in the Frontend where users drag and drop image folders or files.

## Requirements

### Requirement 1: Backend Server Startup

**User Story:** As a user, I want to start the web application with a single command, so that I can quickly begin processing images without complex setup.

#### Acceptance Criteria

1. WHEN the user executes the server start command, THE Backend SHALL start an HTTP server on localhost port 5000.
2. WHEN the Backend starts successfully, THE Backend SHALL log a message indicating the server is running and the URL to access the Frontend.
3. WHEN the Backend starts, THE Backend SHALL load the YOLO and SAM2 models into memory so that subsequent image processing requests do not incur model loading delays.
4. IF the models fail to load during startup, THEN THE Backend SHALL log a descriptive error message and exit with a non-zero status code.

### Requirement 2: Frontend Serving

**User Story:** As a user, I want to access the segmentation interface by navigating to a URL in my browser, so that I do not need to install additional software.

#### Acceptance Criteria

1. WHEN a browser navigates to http://localhost:5000, THE Backend SHALL serve the index.html Frontend page.
2. THE Frontend SHALL render without requiring any external CDN resources or internet connection.
3. THE Frontend SHALL display the Drop_Zone prominently on page load.

### Requirement 3: Drag-and-Drop Image Upload

**User Story:** As a user, I want to drag and drop a folder of images into the interface, so that I can submit multiple images for processing in one action.

#### Acceptance Criteria

1. WHEN the user drags files over the Drop_Zone, THE Frontend SHALL visually indicate that the zone is ready to accept a drop by changing its border style.
2. WHEN the user drops a folder or image files onto the Drop_Zone, THE Frontend SHALL collect all files with extensions .jpg, .jpeg, or .png.
3. WHEN the user drops files that include unsupported formats, THE Frontend SHALL ignore non-image files and process only supported image files.
4. IF the user drops an empty folder or no supported images are found, THEN THE Frontend SHALL display a message stating that no valid images were found.
5. WHEN valid images are collected, THE Frontend SHALL upload the images to the Backend via a multipart HTTP POST request.

### Requirement 4: Image Processing via Pipeline

**User Story:** As a user, I want each uploaded image to be processed through the segmentation pipeline, so that I get annotated results showing detected defects.

#### Acceptance Criteria

1. WHEN the Backend receives uploaded images, THE Backend SHALL process each image sequentially through the Pipeline (preprocessing, YOLO detection, SAM2 segmentation, verification, and visualization).
2. WHEN an image is processed successfully, THE Backend SHALL generate an Annotated_Image with segmentation masks, bounding boxes, contours, and class labels overlaid.
3. IF an individual image fails to process, THEN THE Backend SHALL log the error, skip that image, and continue processing the remaining images in the Batch.
4. WHEN an image contains no detected defects, THE Backend SHALL generate an Annotated_Image with a "No defects detected" label overlaid on the original.

### Requirement 5: Processing Progress Reporting

**User Story:** As a user, I want to see a progress bar while my images are being processed, so that I know how far along the batch is and can estimate remaining time.

#### Acceptance Criteria

1. WHEN a Batch is submitted for processing, THE Frontend SHALL display the Progress_Bar with an initial value of 0%.
2. WHILE images are being processed, THE Backend SHALL provide a progress endpoint that reports the number of images completed and the total number of images in the Batch.
3. WHILE the Batch is in progress, THE Frontend SHALL poll the progress endpoint and update the Progress_Bar to reflect the current completion percentage.
4. WHEN all images in the Batch are processed, THE Frontend SHALL set the Progress_Bar to 100% and display a completion message.
5. THE Frontend SHALL display a text label alongside the Progress_Bar showing "X of Y images processed".

### Requirement 6: Result Display

**User Story:** As a user, I want to view the annotated output images in the browser after processing completes, so that I can inspect the segmentation results without navigating the file system.

#### Acceptance Criteria

1. WHEN all images in a Batch are processed, THE Frontend SHALL display a scrollable gallery of Annotated_Images.
2. WHEN the user clicks on an Annotated_Image in the gallery, THE Frontend SHALL display the image at full resolution in a modal or expanded view.
3. THE Frontend SHALL display the original filename beneath each Annotated_Image in the gallery.
4. WHEN no defects are detected in any image of the Batch, THE Frontend SHALL still display all Annotated_Images showing the "No defects detected" label.

### Requirement 7: Error Handling and User Feedback

**User Story:** As a user, I want clear error messages when something goes wrong, so that I can understand and resolve issues.

#### Acceptance Criteria

1. IF the Backend is unreachable when the Frontend attempts to upload images, THEN THE Frontend SHALL display an error message stating that the server is not available.
2. IF the upload request fails due to a network error, THEN THE Frontend SHALL display an error message and allow the user to retry the upload.
3. WHEN some images in a Batch fail processing, THE Frontend SHALL display a summary indicating which images failed alongside the successfully processed results.
4. IF all images in a Batch fail processing, THEN THE Frontend SHALL display an error message stating that processing failed and suggest checking server logs.

### Requirement 8: File Size and Batch Constraints

**User Story:** As a user, I want the system to handle reasonable batch sizes, so that I can process typical inspection sets without issues.

#### Acceptance Criteria

1. THE Backend SHALL accept Batches of up to 100 images per upload request.
2. IF a Batch exceeds 100 images, THEN THE Frontend SHALL display a message informing the user of the limit and reject the submission.
3. THE Backend SHALL accept individual image files up to 50 MB in size.
4. IF an individual image exceeds the 50 MB size limit, THEN THE Backend SHALL skip that image and report it as failed in the progress response.
