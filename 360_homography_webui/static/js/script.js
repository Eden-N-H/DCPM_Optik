document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let imageFiles = [];
    let manifestFile = null;

    let map = null;
    let geoJsonLayer = null;
    let mapMarkers = {};

    let fullResults = [];
    let fullGeojson = null;
    let appResults = [];
    let currentIndex = 0;
    let currentDirection = "front";

    const btnProcess = document.getElementById("process-btn");
    const selLocation = document.getElementById("sel-location");

    // -----------------------------------------------------------------
    // Map setup
    // -----------------------------------------------------------------

    function initMap() {
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
        const dz = document.getElementById(dzId);
        const input = document.getElementById(inputId);
        const nameLabel = document.getElementById(nameId);

        if (!dz || !input || !nameLabel) {
            console.warn(`Missing upload element: ${dzId}`);
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
        updateProcessButton();
    }

    function updateProcessButton() {
        const hasModel = modelFile !== null;
        const hasImages = imageFiles.length > 0;
        const hasManifest = manifestFile !== null;

        btnProcess.disabled = !(hasModel && (hasImages || hasManifest));
    }

    function clearManifestSelection() {
        manifestFile = null;

        const manifestInput = document.getElementById("in-manifest");
        const manifestLabel = document.getElementById("name-manifest");

        if (manifestInput) {
            manifestInput.value = "";
        }

        if (manifestLabel) {
            manifestLabel.textContent = "";
            manifestLabel.classList.add("hidden");
        }
    }

    function clearManualSelection() {
        imageFiles = [];

        const imageInput = document.getElementById("in-image");
        const imageLabel = document.getElementById("name-image");

        if (imageInput) {
            imageInput.value = "";
        }

        if (imageLabel) {
            imageLabel.textContent = "";
            imageLabel.classList.add("hidden");
        }
    }

    setupDz("dz-model", "in-model", "name-model", false, (file) => {
        modelFile = file;
        updateProcessButton();
    });

    setupDz("dz-image", "in-image", "name-image", true, (files) => {
        // Manual 360 image upload mode selected
        imageFiles = files;

        // Clear manifest mode so old pipeline data is not used accidentally
        clearManifestSelection();

        console.log("Manual 360 image mode selected. Manifest cleared.");
        updateProcessButton();
    });

    setupDz("dz-manifest", "in-manifest", "name-manifest", false, (file) => {
        // Manifest pipeline mode selected
        manifestFile = file;

        // Clear manual image uploads so the mode is not ambiguous
        clearManualSelection();

        console.log("Manifest mode selected. Manual images cleared.");
        updateProcessButton();
    });

    // -----------------------------------------------------------------
    // Submit processing request
    // -----------------------------------------------------------------

    btnProcess.onclick = async () => {
        if (!modelFile) {
            alert("Please upload a YOLO model first.");
            return;
        }

        if (!manifestFile && imageFiles.length === 0) {
            alert("Please upload 360 images or a homography manifest CSV.");
            return;
        }

        const formData = new FormData();

        formData.append("model", modelFile);
        formData.append("cam_height", document.getElementById("cam-height").value || "1.6");

        let endpoint = "";

        if (manifestFile && imageFiles.length === 0) {
            endpoint = "/process_manifest";
            formData.append("manifest", manifestFile);

            console.log("Running manifest mode...");
        } else if (!manifestFile && imageFiles.length > 0) {
            endpoint = "/process";

            imageFiles.forEach((file) => {
                formData.append("images", file);
            });

            console.log("Running manual 360 image mode...");
        } else {
            alert("Please choose either manifest mode or manual upload mode, not both.");
            updateProcessButton();
            return;
        }

        document.getElementById("loading").classList.remove("hidden");
        document.getElementById("upload-panel").classList.add("hidden");
        document.getElementById("workspace").classList.add("hidden");
        btnProcess.disabled = true;

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || "Unknown server error");
            }

            if (!data.results || data.results.length === 0) {
                throw new Error("Processing finished, but no valid results were returned.");
            }

            console.log("Server response:", data);

            loadWorkspace(data.results, data.geojson);

            if (data.skipped_count && data.skipped_count > 0) {
                console.warn("Skipped frames:", data.skipped_frames);
                alert(`Processing completed with ${data.skipped_count} skipped frame(s). Check the browser console or Flask terminal for details.`);
            }

        } catch (error) {
            console.error(error);
            alert("Processing failed: " + error.message);
            document.getElementById("upload-panel").classList.remove("hidden");
        } finally {
            document.getElementById("loading").classList.add("hidden");
            updateProcessButton();
        }
    };

    // -----------------------------------------------------------------
    // Save / load project JSON
    // -----------------------------------------------------------------

    document.getElementById("btn-save-project").onclick = () => {
        if (fullResults.length === 0) {
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
        a.download = "dcpm_360_project.json";
        a.click();

        URL.revokeObjectURL(url);
    };

    document.getElementById("in-load-project").addEventListener("change", (event) => {
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
        fullResults = resultsData;
        fullGeojson = geojsonData;

        initMap();

        document.getElementById("upload-panel").classList.add("hidden");
        document.getElementById("workspace").classList.remove("hidden");
        document.getElementById("btn-save-project").classList.remove("hidden");

        setTimeout(() => {
            map.invalidateSize();
        }, 200);

        renderMap(fullGeojson);
        populateLocations();
        setView("front");
    }

    function populateLocations() {
        const locations = [...new Set(fullResults.map((result) => result.location || "Location 1"))];

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
            initMap();
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

                    marker.on("click", () => {
                        const target = fullResults.find((result) => {
                            return result.original_name === feature.properties.filename;
                        });

                        if (target) {
                            if (selLocation.value !== target.location) {
                                selLocation.value = target.location;

                                appResults = fullResults.filter((result) => {
                                    return result.location === target.location;
                                });
                            }

                            currentIndex = appResults.findIndex((result) => {
                                return result.original_name === target.original_name;
                            });

                            if (currentIndex < 0) {
                                currentIndex = 0;
                            }

                            updateCarousel(false);
                        }
                    });
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

        const frontContainer = document.getElementById("container-bev-front");
        const rearContainer = document.getElementById("container-bev-rear");
        const activeLabel = document.getElementById("label-active-view");

        if (direction === "front") {
            frontContainer.classList.add("border-blue-500", "ring-2", "ring-blue-100");
            frontContainer.classList.remove("border-transparent");

            rearContainer.classList.remove("border-blue-500", "ring-2", "ring-blue-100");
            rearContainer.classList.add("border-transparent");

            activeLabel.textContent = "Front View Active";
        } else {
            rearContainer.classList.add("border-blue-500", "ring-2", "ring-blue-100");
            rearContainer.classList.remove("border-transparent");

            frontContainer.classList.remove("border-blue-500", "ring-2", "ring-blue-100");
            frontContainer.classList.add("border-transparent");

            activeLabel.textContent = "Rear View Active";
        }

        updateCarousel(false);
    }

    document.getElementById("container-bev-front").onclick = () => {
        setView("front");
    };

    document.getElementById("container-bev-rear").onclick = () => {
        setView("rear");
    };

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

        const activeViewData = current.views[currentDirection];

        document.getElementById("carousel-counter").textContent = `Image ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name || "Unknown image";
        document.getElementById("carousel-telemetry").textContent =
            `Pitch: ${current.pitch}° | Lat: ${current.lat} | Lon: ${current.lon}`;

        document.getElementById("img-bev-front").src = current.views.front.bev_url || "";
        document.getElementById("img-bev-rear").src = current.views.rear.bev_url || "";

        document.getElementById("img-rect").src = activeViewData.rect_url || "";

        renderDefectsTable(activeViewData.defects || []);

        const activeMarker = mapMarkers[current.original_name];

        if (activeMarker) {
            if (panMap) {
                map.setView(activeMarker.getLatLng(), 20, {
                    animate: true
                });
            }

            activeMarker.openPopup();
        }

        document.getElementById("btn-prev").disabled = currentIndex === 0;
        document.getElementById("btn-next").disabled = currentIndex === appResults.length - 1;
    }

    function renderDefectsTable(defects) {
        const tbody = document.getElementById("table-defects");
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

    document.getElementById("btn-prev").onclick = () => {
        if (currentIndex > 0) {
            currentIndex--;
            updateCarousel(true);
        }
    };

    document.getElementById("btn-next").onclick = () => {
        if (currentIndex < appResults.length - 1) {
            currentIndex++;
            updateCarousel(true);
        }
    };

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