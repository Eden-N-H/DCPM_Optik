document.addEventListener("DOMContentLoaded", () => {
    console.log("script.js loaded successfully");

    let modelFile = null;
    let imageFiles = [];

    let map = null;
    let geoJsonLayer = null;
    let mapMarkers = {};

    let fullResults = [];
    let fullGeojson = {
        type: "FeatureCollection",
        features: []
    };

    let appResults = [];
    let currentIndex = 0;
    let currentDirection = "front";

    // -----------------------------------------------------------------
    // Safe DOM helpers
    // -----------------------------------------------------------------

    function byId(id) {
        return document.getElementById(id);
    }

    function firstExisting(ids) {
        for (const id of ids) {
            const el = byId(id);
            if (el) {
                return el;
            }
        }
        return null;
    }

    function buttonByText(text) {
        const buttons = Array.from(document.querySelectorAll("button"));

        return buttons.find((btn) => {
            return btn.textContent.trim().toLowerCase().includes(text.toLowerCase());
        }) || null;
    }

    function show(el) {
        if (el) {
            el.classList.remove("hidden");
        }
    }

    function hide(el) {
        if (el) {
            el.classList.add("hidden");
        }
    }

    function setDisabled(el, disabled) {
        if (el) {
            el.disabled = disabled;
        }
    }

    function valueOf(ids, fallback = "") {
        const el = firstExisting(ids);
        return el ? el.value : fallback;
    }

    function checkedOf(ids, fallback = false) {
        const el = firstExisting(ids);
        return el ? el.checked : fallback;
    }

    function safeOn(el, event, handler) {
        if (el) {
            el.addEventListener(event, handler);
        }
    }

    function safeSetText(ids, value) {
        const el = Array.isArray(ids) ? firstExisting(ids) : byId(ids);
        if (el) {
            el.textContent = value;
        }
    }

    function safeSetSrc(ids, value) {
        const el = Array.isArray(ids) ? firstExisting(ids) : byId(ids);
        if (el) {
            el.src = value || "";
        }
    }

    // -----------------------------------------------------------------
    // Main UI element detection
    // -----------------------------------------------------------------

    const btnProcess =
        firstExisting([
            "process-btn",
            "btn-process",
            "process-uploads-btn",
            "btn-process-uploads",
            "btn-process-upload"
        ]) || buttonByText("Process Uploads");

    const btnScanPipeline =
        firstExisting([
            "scan-pipeline-btn",
            "btn-scan-pipeline",
            "btn-scan",
            "pipeline-btn",
            "process-pipeline-btn"
        ]) || buttonByText("Scan Pipeline");

    const btnNewJob =
        firstExisting([
            "btn-new-job",
            "new-job-btn",
            "btn-reset"
        ]) || buttonByText("New Job");

    const btnSaveProject =
        firstExisting([
            "btn-save-project",
            "save-project-btn"
        ]);

    const inputLoadProject =
        firstExisting([
            "in-load-project",
            "load-project-input"
        ]);

    const selLocation =
        firstExisting([
            "sel-location",
            "location-select"
        ]);

    const uploadPanel =
        firstExisting([
            "upload-panel",
            "panel-upload",
            "main-panel"
        ]);

    const workspace =
        firstExisting([
            "workspace",
            "results-panel"
        ]);

    const loading =
        firstExisting([
            "loading",
            "loading-panel",
            "loader"
        ]);

    if (!btnProcess) {
        console.error("Process Uploads button was not found. Check the button ID in index.html.");
    }

    if (!btnScanPipeline) {
        console.warn("Scan Pipeline button was not found. Pipeline scan will be unavailable.");
    }

    // -----------------------------------------------------------------
    // Map setup
    // -----------------------------------------------------------------

    function initMap() {
        const mapEl = byId("map");

        if (!mapEl) {
            console.warn("Map element not found. Skipping map setup.");
            return;
        }

        if (typeof L === "undefined") {
            console.warn("Leaflet is not loaded. Skipping map setup.");
            return;
        }

        if (!map) {
            map = L.map("map").setView([-32.06, 151.90], 15);

            L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
                maxZoom: 24,
                maxNativeZoom: 17,
                attribution: "Tiles &copy; Esri"
            }).addTo(map);
        }
    }

    // -----------------------------------------------------------------
    // Upload dropzone setup
    // -----------------------------------------------------------------

    function setupDz(dzId, inputId, nameId, isMulti, callback) {
        const dz = byId(dzId);
        const input = byId(inputId);
        const nameLabel = byId(nameId);

        if (!dz || !input || !nameLabel) {
            console.warn(`Missing upload element: ${dzId}, ${inputId}, or ${nameId}`);
            return;
        }

        dz.addEventListener("click", () => {
            input.click();
        });

        input.addEventListener("change", () => {
            handleFiles(input.files, isMulti, nameLabel, callback);
        });

        dz.addEventListener("dragover", (event) => {
            event.preventDefault();
            dz.classList.add("active");
        });

        dz.addEventListener("dragleave", () => {
            dz.classList.remove("active");
        });

        dz.addEventListener("drop", (event) => {
            event.preventDefault();
            dz.classList.remove("active");

            input.files = event.dataTransfer.files;
            handleFiles(event.dataTransfer.files, isMulti, nameLabel, callback);
        });
    }

    function handleFiles(files, isMulti, nameLabel, callback) {
        if (!files || files.length === 0) {
            return;
        }

        let selected;

        if (isMulti) {
            selected = Array.from(files);
            nameLabel.textContent = `${selected.length} file(s) selected`;
        } else {
            selected = files[0];
            nameLabel.textContent = selected.name;
        }

        nameLabel.classList.remove("hidden");

        callback(selected);
        updateButtons();
    }

    function updateButtons() {
        const hasModel = modelFile !== null;
        const hasImages = imageFiles.length > 0;

        setDisabled(btnProcess, !(hasModel && hasImages));
        setDisabled(btnScanPipeline, !hasModel);
    }

    setupDz("dz-model", "in-model", "name-model", false, (file) => {
        modelFile = file;
        console.log("Model selected:", file.name);
        updateButtons();
    });

    setupDz("dz-image", "in-image", "name-image", true, (files) => {
        imageFiles = files;
        console.log("Manual media selected:", files.map((f) => f.name));
        updateButtons();
    });

    updateButtons();

    // -----------------------------------------------------------------
    // Settings form data
    // -----------------------------------------------------------------

    function appendCommonSettings(formData) {
        formData.append("cam_height", valueOf(["cam-height", "cam_height"], "1.6"));
        formData.append("frame_skip", valueOf(["frame-skip", "frame_skip"], "30"));

        formData.append("is_360", checkedOf([
            "is-360",
            "is_360",
            "chk-360",
            "check-360",
            "equirectangular-360"
        ], true));

        formData.append("gps_snap", checkedOf([
            "gps-snap",
            "gps_snap",
            "chk-gps-snap",
            "video-gps-snap-sync"
        ], false));

        formData.append("last_lat", "0.0");
        formData.append("last_lon", "0.0");
        formData.append("last_loc_id", "1");
    }

    // -----------------------------------------------------------------
    // Button actions
    // -----------------------------------------------------------------

    safeOn(btnProcess, "click", async () => {
        if (!modelFile) {
            alert("Please upload a YOLO .pt model first.");
            return;
        }

        if (imageFiles.length === 0) {
            alert("Please upload images or a video first.");
            return;
        }

        const formData = new FormData();

        formData.append("model", modelFile);

        imageFiles.forEach((file) => {
            formData.append("images", file);
        });

        appendCommonSettings(formData);

        await startProcessingRequest("/process_uploads", formData);
    });

    safeOn(btnScanPipeline, "click", async () => {
        if (!modelFile) {
            alert("Please upload a YOLO .pt model first.");
            return;
        }

        const formData = new FormData();

        formData.append("model", modelFile);
        appendCommonSettings(formData);

        await startProcessingRequest("/process_pipeline_folder", formData);
    });

    safeOn(btnNewJob, "click", () => {
        location.reload();
    });

    async function startProcessingRequest(endpoint, formData) {
        console.log("Sending request to:", endpoint);

        show(loading);
        hide(workspace);
        setDisabled(btnProcess, true);
        setDisabled(btnScanPipeline, true);

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || "Unknown server error");
            }

            console.log("Initial server response:", data);

            if (data.results && data.geojson) {
                loadWorkspace(data.results, data.geojson);
                return;
            }

            if (!data.task_id) {
                throw new Error("Server did not return a task_id.");
            }

            await listenToProcessingStream(data);

        } catch (error) {
            console.error(error);
            alert("Processing failed: " + error.message);
            show(uploadPanel);
            hide(workspace);
        } finally {
            hide(loading);
            updateButtons();
        }
    }

    function listenToProcessingStream(initialData) {
        return new Promise((resolve, reject) => {
            fullResults = [];

            fullGeojson = initialData.initial_trail || {
                type: "FeatureCollection",
                features: []
            };

            const taskId = initialData.task_id;
            const total = initialData.total_images || 0;

            console.log("Listening to task stream:", taskId);

            const source = new EventSource(`/stream/${taskId}`);

            source.onmessage = (event) => {
                const msg = JSON.parse(event.data);

                console.log("Stream message:", msg);

                if (msg.type === "update") {
                    const result = msg.data;

                    fullResults.push(result);

                    if (result.geojson) {
                        mergeGeojson(fullGeojson, result.geojson);
                    }

                    safeSetText(
                        ["loading-text", "progress-text"],
                        `Processing ${fullResults.length} of ${total || "?"}`
                    );
                }

                if (msg.type === "complete") {
                    source.close();

                    if (fullResults.length === 0) {
                        reject(new Error("Processing completed, but no results were returned."));
                        return;
                    }

                    loadWorkspace(fullResults, fullGeojson);
                    resolve();
                }

                if (msg.type === "error") {
                    source.close();
                    reject(new Error(msg.message || "Processing stream failed."));
                }
            };

            source.onerror = () => {
                source.close();
                reject(new Error("Connection to processing stream was lost."));
            };
        });
    }

    function mergeGeojson(target, incoming) {
        if (!incoming) {
            return;
        }

        if (Array.isArray(incoming)) {
            target.features.push(...incoming);
            return;
        }

        if (incoming.type === "FeatureCollection") {
            target.features.push(...(incoming.features || []));
            return;
        }

        if (incoming.type === "Feature") {
            target.features.push(incoming);
        }
    }

    // -----------------------------------------------------------------
    // Save / load state JSON
    // -----------------------------------------------------------------

    safeOn(btnSaveProject, "click", () => {
        if (fullResults.length === 0) {
            alert("No results available to save.");
            return;
        }

        const projectData = {
            results: fullResults,
            geojson: fullGeojson
        };

        const blob = new Blob(
            [JSON.stringify(projectData, null, 2)],
            { type: "application/json" }
        );

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");

        a.href = url;
        a.download = "dcpm_project_state.json";
        a.click();

        URL.revokeObjectURL(url);
    });

    safeOn(inputLoadProject, "change", (event) => {
        const file = event.target.files[0];

        if (!file) {
            return;
        }

        const reader = new FileReader();

        reader.onload = (readerEvent) => {
            try {
                const data = JSON.parse(readerEvent.target.result);

                if (!data.results || !data.geojson) {
                    throw new Error("Invalid project JSON format.");
                }

                loadWorkspace(data.results, data.geojson);

            } catch (error) {
                alert("Error loading project: " + error.message);
            }
        };

        reader.readAsText(file);
    });

    // -----------------------------------------------------------------
    // Workspace management
    // -----------------------------------------------------------------

    function loadWorkspace(resultsData, geojsonData) {
        fullResults = resultsData || [];
        fullGeojson = geojsonData || {
            type: "FeatureCollection",
            features: []
        };

        console.log("Loading workspace:", fullResults, fullGeojson);

        initMap();

        hide(uploadPanel);
        show(workspace);
        show(btnSaveProject);

        if (map) {
            setTimeout(() => {
                map.invalidateSize();
            }, 200);
        }

        renderMap(fullGeojson);
        populateLocations();
        setView("front");
    }

    function populateLocations() {
        if (!selLocation) {
            console.warn("Location selector not found.");
            appResults = fullResults;
            currentIndex = 0;
            updateCarousel(true);
            return;
        }

        const locations = [...new Set(
            fullResults.map((result) => result.location || "Location 1")
        )];

        selLocation.innerHTML = locations
            .map((location) => `<option value="${escapeHtml(location)}">${escapeHtml(location)}</option>`)
            .join("");

        selLocation.onchange = () => {
            appResults = fullResults.filter((result) => {
                return (result.location || "Location 1") === selLocation.value;
            });

            currentIndex = 0;
            updateCarousel(true);
        };

        if (locations.length > 0) {
            selLocation.value = locations[0];
            selLocation.dispatchEvent(new Event("change"));
        }
    }

    // -----------------------------------------------------------------
    // Map rendering
    // -----------------------------------------------------------------

    function renderMap(geoJsonData) {
        if (!map) {
            return;
        }

        if (geoJsonLayer) {
            map.removeLayer(geoJsonLayer);
        }

        mapMarkers = {};

        if (!geoJsonData || !geoJsonData.features || geoJsonData.features.length === 0) {
            return;
        }

        geoJsonLayer = L.geoJSON(geoJsonData, {
            style: function (feature) {
                if (feature.geometry.type === "Polygon") {
                    const fillColor = feature.properties.view === "rear" ? "#f59e0b" : "#ffaa00";

                    return {
                        color: "#ff0000",
                        weight: 2,
                        fillColor: fillColor,
                        fillOpacity: 0.5
                    };
                }

                if (feature.geometry.type === "LineString") {
                    return {
                        color: "#00b4d8",
                        weight: 3,
                        dashArray: "5, 10"
                    };
                }

                return {};
            },

            pointToLayer: function (feature, latlng) {
                const marker = L.circleMarker(latlng, {
                    radius: 5,
                    fillColor: "#3b82f6",
                    color: "#ffffff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9
                });

                if (feature.properties && feature.properties.type === "camera") {
                    mapMarkers[feature.properties.filename] = marker;
                }

                return marker;
            },

            onEachFeature: function (feature, layer) {
                if (feature.geometry.type === "Polygon") {
                    layer.bindPopup(`
                        <b>${escapeHtml(feature.properties.class || "Defect")}</b><br>
                        View: ${escapeHtml(feature.properties.view || "N/A")}<br>
                        Area: ${escapeHtml(String(feature.properties.area_sqm || "N/A"))} m²
                    `);
                } else if (feature.geometry.type === "Point") {
                    layer.bindPopup(`
                        <b>Photo Location</b><br>
                        ${escapeHtml(feature.properties.filename || "Unknown")}
                    `);
                }
            }
        }).addTo(map);

        try {
            const bounds = geoJsonLayer.getBounds();

            if (bounds.isValid()) {
                map.fitBounds(bounds, {
                    padding: [50, 50]
                });
            }
        } catch (error) {
            console.warn("Could not fit map bounds:", error);
        }
    }

    // -----------------------------------------------------------------
    // Front / rear view toggle
    // -----------------------------------------------------------------

    function setView(direction) {
        currentDirection = direction;

        const frontContainer = byId("container-bev-front");
        const rearContainer = byId("container-bev-rear");
        const activeLabel = byId("label-active-view");

        if (frontContainer && rearContainer) {
            if (direction === "front") {
                frontContainer.classList.add("border-blue-500", "ring-2", "ring-blue-100");
                frontContainer.classList.remove("border-transparent");

                rearContainer.classList.remove("border-blue-500", "ring-2", "ring-blue-100");
                rearContainer.classList.add("border-transparent");

                if (activeLabel) {
                    activeLabel.textContent = "Front View Active";
                }
            } else {
                rearContainer.classList.add("border-blue-500", "ring-2", "ring-blue-100");
                rearContainer.classList.remove("border-transparent");

                frontContainer.classList.remove("border-blue-500", "ring-2", "ring-blue-100");
                frontContainer.classList.add("border-transparent");

                if (activeLabel) {
                    activeLabel.textContent = "Rear View Active";
                }
            }
        }

        updateCarousel(false);
    }

    safeOn(byId("container-bev-front"), "click", () => {
        setView("front");
    });

    safeOn(byId("container-bev-rear"), "click", () => {
        setView("rear");
    });

    // -----------------------------------------------------------------
    // Carousel update
    // -----------------------------------------------------------------

    function updateCarousel(panMap = true) {
        if (!appResults || appResults.length === 0) {
            return;
        }

        const current = appResults[currentIndex];

        if (!current || !current.views) {
            return;
        }

        const activeViewData = current.views[currentDirection] || current.views.front;

        safeSetText("carousel-counter", `Image ${currentIndex + 1} of ${appResults.length}`);
        safeSetText("carousel-filename", current.original_name || "Unknown image");
        safeSetText(
            "carousel-telemetry",
            `Pitch: ${current.pitch}° | Lat: ${current.lat} | Lon: ${current.lon}`
        );

        safeSetSrc("img-bev-front", current.views.front ? current.views.front.bev_url : "");
        safeSetSrc("img-bev-rear", current.views.rear ? current.views.rear.bev_url : "");
        safeSetSrc("img-rect", activeViewData ? activeViewData.rect_url : "");

        renderDefectsTable(activeViewData ? activeViewData.defects || [] : []);

        const activeMarker = mapMarkers[current.original_name] || mapMarkers[current.filename];

        if (map && activeMarker) {
            if (panMap) {
                map.setView(activeMarker.getLatLng(), 20, {
                    animate: true
                });
            }

            activeMarker.openPopup();
        }

        setDisabled(byId("btn-prev"), currentIndex === 0);
        setDisabled(byId("btn-next"), currentIndex === appResults.length - 1);
    }

    function renderDefectsTable(defects) {
        const tbody = byId("table-defects");

        if (!tbody) {
            return;
        }

        tbody.innerHTML = "";

        if (!defects || defects.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="3" class="p-2 text-center text-gray-500">
                        No defects in this view
                    </td>
                </tr>
            `;
            return;
        }

        defects.forEach((defect) => {
            const className = defect.class || defect.class_name || "Defect";
            const confidence = defect.conf !== undefined ? `${(defect.conf * 100).toFixed(0)}%` : "N/A";
            const area = defect.area_sqm !== undefined ? `${defect.area_sqm} m²` : "N/A";

            const row = document.createElement("tr");

            row.innerHTML = `
                <td class="p-2">${escapeHtml(className)}</td>
                <td class="p-2 text-gray-500">${confidence}</td>
                <td class="p-2 font-bold text-red-600">${area}</td>
            `;

            tbody.appendChild(row);
        });
    }

    // -----------------------------------------------------------------
    // Carousel navigation
    // -----------------------------------------------------------------

    safeOn(byId("btn-prev"), "click", () => {
        if (currentIndex > 0) {
            currentIndex--;
            updateCarousel(true);
        }
    });

    safeOn(byId("btn-next"), "click", () => {
        if (currentIndex < appResults.length - 1) {
            currentIndex++;
            updateCarousel(true);
        }
    });

    // -----------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }
});