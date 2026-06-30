// Road Defect Segmentation - Web UI Application
"use strict";

(function () {
  // === DOM References ===
  const dropZone = document.getElementById("drop-zone");
  const errorArea = document.getElementById("error-area");
  const errorMessage = document.getElementById("error-message");
  const errorDismissBtn = document.getElementById("error-dismiss-btn");
  const progressSection = document.getElementById("progress-section");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");
  const gallerySection = document.getElementById("gallery-section");
  const galleryGrid = document.getElementById("gallery-grid");
  const processAnotherBtn = document.getElementById("process-another-btn");
  const modalOverlay = document.getElementById("modal-overlay");
  const modalBackdrop = document.getElementById("modal-backdrop");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalImage = document.getElementById("modal-image");

  // === Constants ===
  const VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"];
  const MAX_BATCH_SIZE = 300;
  const POLL_INTERVAL_MS = 2000;

  // === State ===
  let currentBatchId = null;
  let pollTimerId = null;

  // === Utility Functions ===

  /**
   * Check if a filename has a valid image extension (case-insensitive).
   */
  function isValidImageFile(filename) {
    const lower = filename.toLowerCase();
    return VALID_EXTENSIONS.some(function (ext) {
      return lower.endsWith(ext);
    });
  }

  /**
   * Show an error message in the error area.
   */
  function showError(message) {
    errorMessage.textContent = message;
    errorArea.classList.remove("hidden");
  }

  /**
   * Hide the error area.
   */
  function hideError() {
    errorArea.classList.add("hidden");
    errorMessage.textContent = "";
  }

  /**
   * Transition to the processing/progress state.
   */
  function showProgressState() {
    dropZone.classList.add("hidden");
    hideError();
    progressSection.classList.remove("hidden");
    gallerySection.classList.add("hidden");
    progressBar.value = 0;
    progressBar.max = 100;
    progressText.textContent = "0 of 0 images processed";
  }

  /**
   * Transition to the results/gallery state.
   */
  function showResultsState() {
    progressSection.classList.add("hidden");
    gallerySection.classList.remove("hidden");
    processAnotherBtn.classList.remove("hidden");
  }

  /**
   * Reset the UI to idle state (drop zone visible).
   */
  function resetToIdle() {
    currentBatchId = null;
    stopPolling();
    dropZone.classList.remove("hidden");
    dropZone.classList.remove("drag-over");
    hideError();
    progressSection.classList.add("hidden");
    gallerySection.classList.add("hidden");
    processAnotherBtn.classList.add("hidden");
    galleryGrid.innerHTML = "";
    progressBar.value = 0;
    progressText.textContent = "0 of 0 images processed";
  }

  // === Drop Zone Event Handlers ===

  function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag-over");
  }

  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag-over");
  }

  function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-over");
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-over");
    hideError();

    var items = e.dataTransfer.items;
    if (items && items.length > 0) {
      collectFilesFromItems(items).then(function (files) {
        processCollectedFiles(files);
      });
    } else {
      // Fallback for browsers without DataTransferItem support
      var files = e.dataTransfer.files;
      var fileArray = Array.from(files).filter(function (f) {
        return isValidImageFile(f.name);
      });
      processCollectedFiles(fileArray);
    }
  }

  // === File Collection (with folder recursion) ===

  /**
   * Collect files from DataTransferItemList, supporting folder recursion
   * via webkitGetAsEntry.
   */
  function collectFilesFromItems(items) {
    var entries = [];
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      if (item.webkitGetAsEntry) {
        var entry = item.webkitGetAsEntry();
        if (entry) {
          entries.push(entry);
        }
      } else if (item.kind === "file") {
        // Direct file without entry API - wrap as promise
        var file = item.getAsFile();
        if (file && isValidImageFile(file.name)) {
          entries.push(file);
        }
      }
    }

    if (entries.length === 0) {
      return Promise.resolve([]);
    }

    // Process entries (could be files or directories)
    var promises = entries.map(function (entry) {
      if (entry instanceof File) {
        return Promise.resolve([entry]);
      }
      return readEntryRecursive(entry);
    });

    return Promise.all(promises).then(function (results) {
      var allFiles = [];
      results.forEach(function (fileList) {
        fileList.forEach(function (f) {
          allFiles.push(f);
        });
      });
      return allFiles;
    });
  }

  /**
   * Recursively read a FileSystemEntry (file or directory).
   */
  function readEntryRecursive(entry) {
    if (entry.isFile) {
      return new Promise(function (resolve) {
        entry.file(function (file) {
          if (isValidImageFile(file.name)) {
            resolve([file]);
          } else {
            resolve([]);
          }
        }, function () {
          resolve([]);
        });
      });
    } else if (entry.isDirectory) {
      return readDirectoryEntries(entry.createReader());
    }
    return Promise.resolve([]);
  }

  /**
   * Read all entries from a directory reader, handling batched results.
   */
  function readDirectoryEntries(reader) {
    return new Promise(function (resolve) {
      var allFiles = [];

      function readBatch() {
        reader.readEntries(function (entries) {
          if (entries.length === 0) {
            resolve(allFiles);
            return;
          }
          var promises = entries.map(function (entry) {
            return readEntryRecursive(entry);
          });
          Promise.all(promises).then(function (results) {
            results.forEach(function (files) {
              files.forEach(function (f) {
                allFiles.push(f);
              });
            });
            // Continue reading (directories can return results in batches)
            readBatch();
          });
        }, function () {
          resolve(allFiles);
        });
      }

      readBatch();
    });
  }

  // === File Validation and Upload ===

  /**
   * Process the collected files: validate and upload.
   */
  function processCollectedFiles(files) {
    // Filter to valid images only
    var validFiles = files.filter(function (f) {
      return isValidImageFile(f.name);
    });

    if (validFiles.length === 0) {
      showError("No valid images found. Please drop .jpg, .jpeg, or .png files.");
      return;
    }

    if (validFiles.length > MAX_BATCH_SIZE) {
      showError(
        "Batch exceeds the limit of " + MAX_BATCH_SIZE + " images. " +
        "You selected " + validFiles.length + " images. Please reduce the batch size."
      );
      return;
    }

    uploadFiles(validFiles);
  }

  /**
   * Upload valid image files via multipart FormData POST to /api/upload.
   */
  function uploadFiles(files) {
    var formData = new FormData();
    files.forEach(function (file) {
      formData.append("images", file);
    });

    showProgressState();
    progressText.textContent = "Uploading " + files.length + " images...";

    fetch("/api/upload", {
      method: "POST",
      body: formData,
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (data) {
            throw new Error(data.error || "Upload failed with status " + response.status);
          });
        }
        return response.json();
      })
      .then(function (data) {
        currentBatchId = data.batch_id;
        progressBar.max = data.total_images;
        progressBar.value = 0;
        progressText.textContent = "0 of " + data.total_images + " images processed";
        startPolling();
      })
      .catch(function (err) {
        resetToIdle();
        if (err.message && err.message.indexOf("Failed to fetch") !== -1) {
          showError("Server not available. Please ensure the backend is running.");
        } else {
          showError(err.message || "An error occurred during upload.");
        }
      });
  }

  // === Progress Polling ===

  /**
   * Start polling /api/progress every 2 seconds.
   */
  function startPolling() {
    stopPolling();
    pollProgress();
    pollTimerId = setInterval(pollProgress, POLL_INTERVAL_MS);
  }

  /**
   * Stop the progress polling interval.
   */
  function stopPolling() {
    if (pollTimerId !== null) {
      clearInterval(pollTimerId);
      pollTimerId = null;
    }
  }

  /**
   * Fetch progress and update the UI.
   */
  var pollRetryCount = 0;
  var MAX_POLL_RETRIES = 3;

  function pollProgress() {
    if (!currentBatchId) return;

    fetch("/api/progress?batch_id=" + encodeURIComponent(currentBatchId))
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Progress request failed");
        }
        return response.json();
      })
      .then(function (data) {
        pollRetryCount = 0;
        var completed = data.completed || 0;
        var total = data.total || 0;

        progressBar.max = total;
        progressBar.value = completed;
        progressText.textContent = completed + " of " + total + " images processed";

        if (data.status === "complete") {
          stopPolling();
          progressBar.value = total;
          progressText.textContent = total + " of " + total + " images processed";
          fetchResults();
        }
      })
      .catch(function () {
        pollRetryCount++;
        if (pollRetryCount >= MAX_POLL_RETRIES) {
          stopPolling();
          showError("Lost connection to server. Please check that the backend is running.");
        }
      });
  }

  // === Results Display ===

  /**
   * Fetch results from /api/results and populate the gallery.
   */
  function fetchResults() {
    if (!currentBatchId) return;

    fetch("/api/results?batch_id=" + encodeURIComponent(currentBatchId))
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Failed to fetch results");
        }
        return response.json();
      })
      .then(function (data) {
        displayResults(data.results || []);
        showResultsState();
      })
      .catch(function () {
        showError("Failed to load results. Please try again.");
      });
  }

  /**
   * Populate the gallery grid with result thumbnails.
   */
  function displayResults(results) {
    galleryGrid.innerHTML = "";

    results.forEach(function (result) {
      var item = document.createElement("div");
      item.className = "gallery-item";

      if (result.status === "failed") {
        item.classList.add("failed");
      }

      // Image thumbnail
      var img = document.createElement("img");
      if (result.status === "success" && result.output_url) {
        img.src = result.output_url;
        img.alt = "Annotated result for " + result.filename;
      } else {
        // Placeholder for failed images
        img.src = "";
        img.alt = "Processing failed for " + result.filename;
      }

      // Info section
      var info = document.createElement("div");
      info.className = "gallery-item-info";

      var filename = document.createElement("p");
      filename.className = "gallery-item-filename";
      filename.textContent = result.filename;
      filename.title = result.filename;

      var defects = document.createElement("p");
      defects.className = "gallery-item-defects";

      if (result.status === "failed") {
        defects.textContent = "Error: " + (result.error || "Processing failed");
      } else {
        var count = result.defects_found || 0;
        defects.textContent = count === 0
          ? "No defects detected"
          : count + " defect" + (count !== 1 ? "s" : "") + " found";
      }

      info.appendChild(filename);
      info.appendChild(defects);
      item.appendChild(img);
      item.appendChild(info);

      // Click to open modal (only for successful results)
      if (result.status === "success" && result.output_url) {
        item.addEventListener("click", function () {
          openModal(result.output_url, result.filename);
        });
      }

      galleryGrid.appendChild(item);
    });
  }

  // === Modal ===

  /**
   * Open the modal with a full-resolution image.
   */
  function openModal(imageUrl, filename) {
    modalImage.src = imageUrl;
    modalImage.alt = "Full resolution: " + filename;
    modalOverlay.classList.remove("hidden");
    document.body.style.overflow = "hidden";
  }

  /**
   * Close the modal.
   */
  function closeModal() {
    modalOverlay.classList.add("hidden");
    modalImage.src = "";
    document.body.style.overflow = "";
  }

  // === Event Listeners Setup ===

  // Drop zone events
  dropZone.addEventListener("dragenter", handleDragEnter);
  dropZone.addEventListener("dragover", handleDragOver);
  dropZone.addEventListener("dragleave", handleDragLeave);
  dropZone.addEventListener("drop", handleDrop);

  // Prevent default drag behavior on the document to avoid
  // the browser opening dropped files
  document.addEventListener("dragover", function (e) {
    e.preventDefault();
  });
  document.addEventListener("drop", function (e) {
    e.preventDefault();
  });

  // Error dismiss button
  errorDismissBtn.addEventListener("click", hideError);

  // Process Another Batch button
  processAnotherBtn.addEventListener("click", resetToIdle);

  // Modal close handlers
  modalCloseBtn.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !modalOverlay.classList.contains("hidden")) {
      closeModal();
    }
  });
  // === Model Upload ===
  const modelUploadBtn = document.getElementById("model-upload-btn");
  const modelFileInput = document.getElementById("model-file-input");
  const modelName = document.getElementById("model-name");

  // Fetch current model name and confidence on load
  fetch("/api/model")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      modelName.textContent = "Model: " + data.model_name;
      if (data.confidence_threshold !== undefined) {
        confidenceInput.value = Math.round(data.confidence_threshold * 100);
      }
    })
    .catch(function () {
      modelName.textContent = "Model: unknown";
    });

  // === Confidence Threshold ===
  const confidenceInput = document.getElementById("confidence-input");

  confidenceInput.addEventListener("change", function () {
    var pct = parseFloat(confidenceInput.value);
    if (isNaN(pct) || pct < 0 || pct > 100) {
      showError("Confidence must be between 0 and 100.");
      return;
    }
    var threshold = pct / 100;

    fetch("/api/confidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence_threshold: threshold }),
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (data) {
            throw new Error(data.error || "Failed to update confidence");
          });
        }
        return response.json();
      })
      .then(function () {
        hideError();
      })
      .catch(function (err) {
        showError(err.message || "Failed to update confidence threshold.");
      });
  });

  modelUploadBtn.addEventListener("click", function () {
    modelFileInput.click();
  });

  modelFileInput.addEventListener("change", function () {
    var file = modelFileInput.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pt")) {
      showError("Only .pt model files are supported.");
      return;
    }

    modelName.textContent = "Model: loading...";
    var formData = new FormData();
    formData.append("model", file);

    fetch("/api/model", { method: "POST", body: formData })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (data) {
            throw new Error(data.error || "Model upload failed");
          });
        }
        return response.json();
      })
      .then(function (data) {
        modelName.textContent = "Model: " + data.model_name;
        hideError();
      })
      .catch(function (err) {
        modelName.textContent = "Model: load failed";
        showError(err.message || "Failed to load model.");
      });

    // Reset input so the same file can be re-selected
    modelFileInput.value = "";
  });
})();
