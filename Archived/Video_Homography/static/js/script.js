// Video_Homography/static/js/script.js

let modelFile = null;
let imageFiles = [];
let manifestFile = null;

let results = [];
let currentIndex = 0;

let map = null;
let geojsonLayer = null;

// ---------------------------------------------------------------------
// Page elements
// ---------------------------------------------------------------------

const btnProcess = document.getElementById("process-btn");
const loading = document.getElementById("loading");
const workspace = document.getElementById("workspace");

const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");

const carouselCounter = document.getElementById("carousel-counter");
const carouselFilename = document.getElementById("carousel-filename");
const carouselTelemetry = document.getElementById("carousel-telemetry");

const imgRect = document.getElementById("img-rect");
const imgBev = document.getElementById("img-bev");
const tableDefects = document.getElementById("table-defects");

// ---------------------------------------------------------------------
// Dropzone setup
// ---------------------------------------------------------------------

function setupDz(dropzoneId, inputId, nameId, multiple, callback) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    const nameLabel = document.getElementById(nameId);

    if (!dropzone || !input || !nameLabel) {
        console.warn(`Missing dropzone setup element: ${dropzoneId}`);
        return;
    }

    dropzone.addEventListener("click", () => {
        input.click();
    });

    input.addEventListener("change", () => {
        handleFiles(input.files, multiple, nameLabel, callback);
    });

    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("active");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("active");
    });

    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("active");

        const droppedFiles = event.dataTransfer.files;
        input.files = droppedFiles;

        handleFiles(droppedFiles, multiple, nameLabel, callback);
    });
}


function handleFiles(files, multiple, nameLabel, callback) {
    if (!files || files.length === 0) {
        return;
    }

    let selected;

    if (multiple) {
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
    const hasManualInput = imageFiles.length > 0;
    const hasManifest = manifestFile !== null;

    btnProcess.disabled = !(hasModel && (hasManualInput || hasManifest));
}

// ---------------------------------------------------------------------
// Initialise upload boxes
// ---------------------------------------------------------------------

setupDz("dz-model", "in-model", "name-model", false, (file) => {
    modelFile = file;
});

setupDz("dz-image", "in-image", "name-image", true, (files) => {
    imageFiles = files;
});

setupDz("dz-manifest", "in-manifest", "name-manifest", false, (file) => {
    manifestFile = file;
});

// ---------------------------------------------------------------------
// Pipeline submit
// ---------------------------------------------------------------------

btnProcess.addEventListener("click", async () => {
    if (!modelFile) {
        alert("Please upload a YOLO model file first.");
        return;
    }

    if (!manifestFile && imageFiles.length === 0) {
        alert("Please upload either images/video or a homography manifest CSV.");
        return;
    }

    const formData = new FormData();

    formData.append("model", modelFile);

    const camHeight = document.getElementById("cam-height").value || "1.6";
    formData.append("cam_height", camHeight);

    let endpoint = "/process";

    if (manifestFile) {
        endpoint = "/process_manifest";
        formData.append("manifest", manifestFile);
        console.log("Running manifest pipeline mode...");
    } else {
        endpoint = "/process";

        imageFiles.forEach((file) => {
            formData.append("files", file);
        });

        const gpsSnapElement = document.getElementById("gps-snap");
        if (gpsSnapElement) {
            formData.append("gps_snap", gpsSnapElement.checked ? "true" : "false");
        }

        console.log("Running manual image/video upload mode...");
    }

    loading.classList.remove("hidden");
    workspace.classList.add("hidden");
    btnProcess.disabled = true;

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Unknown server error");
        }

        console.log("Server response:", data);

        results = data.results || [];
        currentIndex = 0;

        if (results.length === 0) {
            alert("Processing finished, but no valid results were returned. Check the terminal for skipped frames or errors.");
            return;
        }

        workspace.classList.remove("hidden");

        renderCurrentResult();
        initialiseMap();
        renderMap(data.geojson);

        if (data.skipped_count && data.skipped_count > 0) {
            console.warn("Skipped frames:", data.skipped_frames);
            alert(`Processing completed with ${data.skipped_count} skipped frame(s). Check the browser console or Flask terminal for details.`);
        }

    } catch (error) {
        console.error(error);
        alert(`Processing failed: ${error.message}`);
    } finally {
        loading.classList.add("hidden");
        updateProcessButton();
    }
});

// ---------------------------------------------------------------------
// Carousel rendering
// ---------------------------------------------------------------------

function renderCurrentResult() {
    if (!results || results.length === 0) {
        return;
    }

    const item = results[currentIndex];

    carouselCounter.textContent = `Item ${currentIndex + 1} of ${results.length}`;
    carouselFilename.textContent = item.original_name || "Unknown frame";

    const pitch = item.pitch !== undefined ? item.pitch : "N/A";
    const lat = item.lat !== undefined ? item.lat : "N/A";
    const lon = item.lon !== undefined ? item.lon : "N/A";

    carouselTelemetry.textContent = `Pitch: ${pitch}° | Lat: ${lat} | Lon: ${lon}`;

    imgRect.src = item.rect_url || "";
    imgBev.src = item.bev_url || "";

    renderDefectsTable(item.defects || []);

    btnPrev.disabled = currentIndex === 0;
    btnNext.disabled = currentIndex === results.length - 1;
}


function renderDefectsTable(defects) {
    tableDefects.innerHTML = "";

    if (!defects || defects.length === 0) {
        tableDefects.innerHTML = `
            <tr>
                <td class="p-2 text-gray-500">No defects detected in this frame.</td>
            </tr>
        `;
        return;
    }

    defects.forEach((defect, index) => {
        const row = document.createElement("tr");

        const className =
            defect.class ||
            defect.class_name ||
            defect.name ||
            defect.label ||
            `Defect ${index + 1}`;

        const confidence =
            defect.confidence !== undefined
                ? Number(defect.confidence).toFixed(2)
                : defect.conf !== undefined
                    ? Number(defect.conf).toFixed(2)
                    : "N/A";

        const area =
            defect.area_m2 !== undefined
                ? `${Number(defect.area_m2).toFixed(3)} m²`
                : defect.area !== undefined
                    ? `${Number(defect.area).toFixed(3)}`
                    : "N/A";

        const depth =
            defect.depth_mm !== undefined
                ? `${Number(defect.depth_mm).toFixed(1)} mm`
                : defect.depth !== undefined
                    ? `${Number(defect.depth).toFixed(1)}`
                    : "N/A";

        row.innerHTML = `
            <td class="p-2 font-medium text-gray-700">${index + 1}. ${escapeHtml(className)}</td>
            <td class="p-2 text-gray-600">Conf: ${confidence}</td>
            <td class="p-2 text-gray-600">Area: ${area}</td>
            <td class="p-2 text-gray-600">Depth: ${depth}</td>
        `;

        tableDefects.appendChild(row);
    });
}


btnPrev.addEventListener("click", () => {
    if (currentIndex > 0) {
        currentIndex--;
        renderCurrentResult();
    }
});


btnNext.addEventListener("click", () => {
    if (currentIndex < results.length - 1) {
        currentIndex++;
        renderCurrentResult();
    }
});

// ---------------------------------------------------------------------
// Map rendering
// ---------------------------------------------------------------------

function initialiseMap() {
    if (map !== null) {
        return;
    }

    map = L.map("map").setView([-33.8688, 151.2093], 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 22,
        attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
}


function renderMap(geojson) {
    if (!map) {
        initialiseMap();
    }

    if (geojsonLayer) {
        map.removeLayer(geojsonLayer);
    }

    if (!geojson || !geojson.features || geojson.features.length === 0) {
        return;
    }

    geojsonLayer = L.geoJSON(geojson, {
        pointToLayer: function (feature, latlng) {
            const type = feature.properties?.type || "";

            if (type === "camera") {
                return L.circleMarker(latlng, {
                    radius: 5,
                    color: "#2563eb",
                    fillColor: "#3b82f6",
                    fillOpacity: 0.9,
                });
            }

            return L.circleMarker(latlng, {
                radius: 6,
                color: "#dc2626",
                fillColor: "#ef4444",
                fillOpacity: 0.9,
            });
        },

        style: function (feature) {
            const type = feature.properties?.type || "";

            if (type === "trail") {
                return {
                    color: "#2563eb",
                    weight: 4,
                    opacity: 0.8,
                };
            }

            return {
                color: "#dc2626",
                weight: 2,
                opacity: 0.8,
            };
        },

        onEachFeature: function (feature, layer) {
            const props = feature.properties || {};
            let popup = "";

            Object.keys(props).forEach((key) => {
                popup += `<strong>${escapeHtml(key)}:</strong> ${escapeHtml(String(props[key]))}<br>`;
            });

            if (popup) {
                layer.bindPopup(popup);
            }
        },
    }).addTo(map);

    try {
        const bounds = geojsonLayer.getBounds();

        if (bounds.isValid()) {
            map.fitBounds(bounds, {
                padding: [30, 30],
            });
        }
    } catch (error) {
        console.warn("Could not fit map bounds:", error);
    }
}

// ---------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}