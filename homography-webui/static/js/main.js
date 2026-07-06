import { state } from './state.js';
import { toggleMapLayerVisibility, clearOrthomosaics, initMap, updateMapSource, fitMapToBounds, addOrthomosaicShingle } from './map.js';
import { setupDz, checkCanProcess, setView, handleMapClick, refreshLocationsUI, updateCarousel, toggleWarningsModal, toggleMapView, initResizers, autoFitSplitters, setupCalibrationUI, initDrawMode, openFullscreen, initFullscreenModal } from './ui.js';
import { executeJob, triggerZipExport, cancelJob } from './api.js';

document.addEventListener("DOMContentLoaded", () => {
    
    initResizers();
    setupCalibrationUI();
    initDrawMode();
    initFullscreenModal();

    document.getElementById("rect-image-wrapper").addEventListener("click", () => {
        openFullscreen('rect');
    });

    fetch("/classes").then(r => r.json()).then(data => {
        window.modelClasses = data;
        const optionsHtml = data.map(c => `<option value="${c}">${c}</option>`).join('');
        if(document.getElementById("draw-class-select")) document.getElementById("draw-class-select").innerHTML = optionsHtml;
        if(document.getElementById("change-class-select")) document.getElementById("change-class-select").innerHTML = optionsHtml;
    }).catch(e => console.error("Failed to load classes", e));

    document.getElementById("sel-media-type").addEventListener("change", (e) => {
        const val = e.target.value;
        const is360 = val.startsWith("360");
        const hasTelemetry = val.startsWith("360") || val === "standard-video";
        document.getElementById("chk-is-360").checked = is360;
        document.getElementById("chk-has-telemetry").checked = hasTelemetry;
    });

    setupDz("dz-model", "in-model", "name-model", false, f => { state.modelFile = f; checkCanProcess(); });
    setupDz("dz-image", "in-image", "name-image", true, f => { state.imageFiles = f; checkCanProcess(); });

    state.isModelLoaded = true;
    document.getElementById("status-model").classList.remove("hidden");
    document.getElementById("name-model").textContent = "RMCC_8_classes.pt (pre-loaded)";
    document.getElementById("name-model").classList.remove("hidden");
    document.getElementById("lbl-model").textContent = "Model ready (drop .pt to swap)";
    checkCanProcess();

    document.getElementById("process-btn").onclick = () => executeJob();
    document.getElementById("btn-cancel-job").onclick = () => cancelJob();
    
    document.getElementById("btn-export-zip").onclick = () => triggerZipExport(
        "/export-zip", 
        "btn-export-zip", 
        `<img src="https://api.iconify.design/svg-spinners/180-ring.svg?color=white" class="w-4 h-4 inline align-middle -mt-0.5 mr-1"> Compiling RAW...`, 
        "DCPM_RAW_Export.zip"
    );
    
    document.getElementById("btn-export-flat-zip").onclick = () => triggerZipExport(
        "/export-flat-zip", 
        "btn-export-flat-zip", 
        `<img src="https://api.iconify.design/svg-spinners/180-ring.svg?color=white" class="w-4 h-4 inline align-middle -mt-0.5 mr-1"> Compiling Flattened...`, 
        "DCPM_FLAT_Export.zip"
    );

    const chkLayerFront = document.getElementById("chk-layer-front");
    const chkLayerRear = document.getElementById("chk-layer-rear");
    chkLayerFront.addEventListener('change', (e) => toggleMapLayerVisibility('front', e.target.checked));
    chkLayerRear.addEventListener('change', (e) => toggleMapLayerVisibility('rear', e.target.checked));
    
    document.getElementById("btn-toggle-map").onclick = () => toggleMapView();
    document.getElementById('container-bev-front').onclick = () => setView('front');
    document.getElementById('container-bev-rear').onclick = () => setView('rear');

    document.getElementById("btn-prev").onclick = () => { if (state.currentIndex > 0) { state.currentIndex--; updateCarousel(true); } };
    document.getElementById("btn-next").onclick = () => { if (state.currentIndex < state.appResults.length - 1) { state.currentIndex++; updateCarousel(true); } };

    document.getElementById("btn-show-warnings").onclick = () => toggleWarningsModal(true);
    document.getElementById("btn-close-warnings").onclick = () => toggleWarningsModal(false);

    document.getElementById("btn-save-project").onclick = () => {
        if (state.fullResults.length === 0) return;
        const blob = new Blob([JSON.stringify({ is_360: state.appIs360, results: state.fullResults, geojson: state.fullGeojson })], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = "dcpm_project.json"; a.click(); URL.revokeObjectURL(url);
    };

    document.getElementById("in-load-project").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const data = JSON.parse(ev.target.result);
                if (!data.results || !data.geojson) throw new Error("Invalid project format.");
                
                state.appIs360 = data.is_360 !== undefined ? data.is_360 : true;
                chkLayerFront.checked = true;
                chkLayerRear.checked = true;

                const containerBevRear = document.getElementById("container-bev-rear");
                const layerTogglePanel = document.getElementById("layer-toggle-panel");

                if (!state.appIs360) { 
                    containerBevRear.classList.add("hidden"); 
                    layerTogglePanel.classList.add("hidden");
                    layerTogglePanel.classList.remove("flex");
                    setView('front'); 
                } else { 
                    containerBevRear.classList.remove("hidden");
                    layerTogglePanel.classList.remove("hidden");
                    layerTogglePanel.classList.add("flex");
                }

                state.fullResults = data.results;
                state.fullGeojson = data.geojson;
                
                if (state.fullResults.length > 0) {
                    const lastRec = state.fullResults[state.fullResults.length - 1];
                    state.stateLastLat = lastRec.lat; state.stateLastLon = lastRec.lon;
                    state.stateLastLocId = parseInt(lastRec.location.replace("Location ", "")) || 1;
                }

                document.getElementById("upload-panel").classList.add("hidden");
                document.getElementById("workspace").classList.remove("hidden");
                document.getElementById("btn-save-project").classList.remove("hidden");
                document.getElementById("btn-export-zip").classList.remove("hidden");
                document.getElementById("btn-export-flat-zip").classList.remove("hidden");
                document.getElementById("btn-toggle-map").classList.remove("hidden"); 
                
                state.layoutPrefs.mapOn.isManual = false;
                state.layoutPrefs.mapOff.isManual = false;

                clearOrthomosaics();
                state.nodesGeoJson = { type: "FeatureCollection", features: [] };
                let trailCoords = [];

                state.fullResults.forEach(img => {
                    if (img.lat !== null && img.lon !== null) {
                        trailCoords.push([img.lon, img.lat]);
                        state.nodesGeoJson.features.push({
                            type: "Feature",
                            geometry: { type: "Point", coordinates: [img.lon, img.lat] },
                            properties: { original_name: img.original_name, location: img.location, processed: true, active: false }
                        });
                    }
                });

                if (trailCoords.length > 1) {
                    state.trailGeoJson = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "LineString", coordinates: trailCoords }, properties: {} }] };
                } else {
                    state.trailGeoJson = { type: "FeatureCollection", features: [] };
                }

                initMap(handleMapClick);

                setTimeout(() => {
                    if (state.map) state.map.resize();
                    updateMapSource('defects-source', state.fullGeojson);
                    updateMapSource('nodes-source', state.nodesGeoJson);
                    updateMapSource('trail-source', state.trailGeoJson);

                    state.fullResults.forEach(r => addOrthomosaicShingle(r, true, true));

                    if (state.fullGeojson.features.length > 0) fitMapToBounds(state.fullGeojson);
                    else if (state.trailGeoJson.features.length > 0) fitMapToBounds(state.trailGeoJson);

                    refreshLocationsUI(); 
                    setView('front'); 
                    if (state.appResults.length > 0) updateCarousel(true);
                }, 300);
            } catch (err) { alert("Error loading project: " + err.message); }
        };
        reader.readAsText(file);
    });
});