import { state } from './state.js';
import { toggleMapLayerVisibility, clearOrthomosaics, initMap, updateMapSource, fitMapToBounds, addOrthomosaicShingle } from './map.js';
import { setupDz, checkCanProcess, setView, handleMapClick, refreshLocationsUI, updateCarousel, toggleWarningsModal, toggleMapView, initResizers, autoFitSplitters, setupCalibrationUI, initDrawMode, openFullscreen, initFullscreenModal, setupDiagnosticsUI, combineDefectSegments } from './ui.js';
import { executeJob, triggerZipExport, cancelJob, exportProject, importProject } from './api.js';

document.addEventListener("DOMContentLoaded", () => {
    
    initResizers();
    setupCalibrationUI();
    setupDiagnosticsUI();
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

    document.getElementById("chk-skip-ai").addEventListener("change", checkCanProcess);

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
        const projectState = { is_360: state.appIs360, results: state.fullResults, geojson: state.fullGeojson };
        exportProject(projectState, "btn-save-project", "dcpm_project.dcpmproj");
    };
    
    const btnGroupDefectsHeader = document.getElementById("btn-group-defects");
    if (btnGroupDefectsHeader) {
        btnGroupDefectsHeader.onclick = () => combineDefectSegments([btnGroupDefectsHeader]);
    }

    document.getElementById("in-load-project").addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const loaderDiv = document.createElement('div');
        loaderDiv.style.position = 'fixed';
        loaderDiv.style.inset = '0';
        loaderDiv.style.backgroundColor = 'rgba(255,255,255,0.85)';
        loaderDiv.style.zIndex = '9999';
        loaderDiv.style.display = 'flex';
        loaderDiv.style.alignItems = 'center';
        loaderDiv.style.justifyContent = 'center';
        loaderDiv.style.fontFamily = 'monospace';
        loaderDiv.style.fontSize = '1.2rem';
        loaderDiv.style.fontWeight = 'bold';
        loaderDiv.innerHTML = 'EXTRACTING PROJECT DATA...';
        document.body.appendChild(loaderDiv);
        
        try {
            const data = await importProject(file);
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
            document.getElementById("btn-analyze-passes").classList.remove("hidden");
            if (document.getElementById("btn-group-defects")) {
                document.getElementById("btn-group-defects").classList.remove("hidden");
            }
            
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
        } catch (err) { 
            alert("Error loading project: " + err.message); 
        } finally {
            document.body.removeChild(loaderDiv);
            e.target.value = ''; // Reset input to allow loading the same file twice
        }
    });
});

