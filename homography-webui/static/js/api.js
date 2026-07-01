import { state } from './state.js';
import { initMap, clearOrthomosaics, addOrthomosaicShingle, updateMapSource, fitMapToBounds } from './map.js';
import { refreshLocationsUI, updateCarousel, setView, checkCanProcess, handleMapClick, addWarning } from './ui.js';

export async function triggerZipExport(endpoint, btnId, loadingText, filename) {
    if (state.fullResults.length === 0) return;
    const btn = document.getElementById(btnId); const originalText = btn.textContent;
    btn.textContent = loadingText; btn.disabled = true;
    try {
        const res = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ results: state.fullResults }) });
        if (!res.ok) throw new Error("Failed to compile ZIP file");
        const blob = await res.blob(); const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
    } catch (err) { alert(err.message); } finally { btn.textContent = originalText; btn.disabled = false; }
}

export async function cancelJob() {
    if (!state.currentTaskId) return;
    if (!confirm("Are you sure you want to stop processing? Images completed so far will be saved.")) return;
    const btn = document.getElementById("btn-cancel-job");
    btn.disabled = true; btn.textContent = "Cancelling...";
    try {
        await fetch(`/cancel/${state.currentTaskId}`, { method: 'POST' });
    } catch(e) { console.error("Cancel request failed", e); }
}

export async function fetchGridPreview(filename, view, calibrationConfig) {
    const res = await fetch("/preview_grid", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, view, calibration: calibrationConfig })
    });
    const data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || "Failed to preview");
    return data.image;
}

export async function autoDetectVP(filename, view, calibrationConfig) {
    const res = await fetch("/auto_vp", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, view, calibration: calibrationConfig })
    });
    return await res.json();
}

export async function clickManualVP(filename, view, calibrationConfig, px, py) {
    const res = await fetch("/click_vp", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, view, calibration: calibrationConfig, px, py })
    });
    return await res.json();
}

export async function recalculateProject(calibrationConfig) {
    const res = await fetch("/recalculate_bev", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ calibration: calibrationConfig, results: state.fullResults })
    });
    const data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || "Failed to recalculate");
    return data.results;
}

function startSSE(taskId, totalImages) {
    const source = new EventSource(`/stream/${taskId}`);
    let processedCount = 0; let startTime = Date.now();
    const telemetryHud = document.getElementById("telemetry-hud");

    source.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === "error" || msg.type === "complete" || msg.type === "cancelled") {
            source.close();
            state.currentTaskId = null;
            document.getElementById("progress-container").classList.add("hidden");
            checkCanProcess();
            state.imageFiles = [];
            document.getElementById("name-image").textContent = "Completed.";
            
            if (msg.type === "error") {
                alert(`Background Task Error: ${msg.message}`);
            } else if (state.fullResults.length === 0) {
                alert(msg.type === "cancelled" ? "Process cancelled before any frames finished." : "Processing complete, but 0 frames were successfully extracted. Check warnings.");
                document.getElementById("workspace").classList.add("hidden");
                document.getElementById("upload-panel").classList.remove("hidden");
                document.getElementById("btn-save-project").classList.add("hidden");
                document.getElementById("btn-export-zip").classList.add("hidden");
                document.getElementById("btn-export-flat-zip").classList.add("hidden");
                document.getElementById("btn-toggle-map").classList.add("hidden");
            } else if (msg.type === "cancelled") {
                addWarning("⚠️ User Cancelled Process. Partial results have been saved.");
            }
            return;
        }

        if (msg.type === "health_report") {
            telemetryHud.classList.remove("hidden");
            const hr = msg.data;
            let gpsColor = hr.gps_score > 80 ? 'text-green-400' : 'text-orange-400';
            let imuColor = hr.imu_score > 90 ? 'text-green-400' : 'text-orange-400';

            const hudLine = document.createElement("div");
            hudLine.innerHTML = `<span class="font-bold text-gray-300 mr-1">[${msg.original_name}]</span> GPS: <span class="${gpsColor} font-bold">${hr.gps_score.toFixed(0)}%</span> | IMU: <span class="${imuColor} font-bold">${hr.imu_score.toFixed(0)}%</span> | Drift: <span class="text-gray-300">${hr.metrics.avg_gps_speed_error_ms.toFixed(2)}m/s</span>`;
            telemetryHud.appendChild(hudLine);
            
            if (hr.warnings.length > 0) hr.warnings.forEach(w => addWarning(`[${msg.original_name} Telemetry] ${w}`));
            return;
        }

        if (msg.type === "item_error") {
            addWarning(`Skipped ${msg.original_name}: ${msg.message}`);
            if (!msg.is_video) {
                processedCount++;
                const pct = totalImages > 0 ? (processedCount / totalImages) * 100 : 100;
                document.getElementById("progress-bar").style.width = `${pct}%`;
                document.getElementById("progress-text").textContent = `Segmenting ${processedCount} of ${totalImages}`;
            }
            return;
        }

        if (msg.type === "update") {
            const r = msg.data;
            state.fullResults.push(r);
            addOrthomosaicShingle(r, document.getElementById("chk-layer-front").checked, document.getElementById("chk-layer-rear").checked);
            
            let hasDefects = r.geojson && r.geojson.length > 0;
            if (hasDefects) {
                state.fullGeojson.features.push(...r.geojson);
                updateMapSource('defects-source', state.fullGeojson);
            }

            const fIndex = state.nodesGeoJson.features.findIndex(f => f.properties.original_name === r.original_name);
            if (fIndex > -1) {
                state.nodesGeoJson.features[fIndex].properties.processed = true;
                state.nodesGeoJson.features[fIndex].properties.location = r.location;
            } else if (r.lat !== null && r.lon !== null) {
                state.nodesGeoJson.features.push({ type: "Feature", geometry: { type: "Point", coordinates: [r.lon, r.lat] }, properties: { original_name: r.original_name, location: r.location, processed: true, active: false }});
            }
            updateMapSource('nodes-source', state.nodesGeoJson);

            processedCount++;
            const pct = totalImages > 0 ? (processedCount / totalImages) * 100 : 100;
            document.getElementById("progress-bar").style.width = `${pct}%`;
            document.getElementById("progress-text").textContent = `Segmenting ${processedCount} of ${totalImages}`;

            const elapsedSec = (Date.now() - startTime) / 1000;
            const remainSec = Math.ceil((totalImages - processedCount) * (elapsedSec / processedCount));
            document.getElementById("eta-text").textContent = `ETA: ${Math.floor(remainSec / 60)}m ${remainSec % 60}s`;

            refreshLocationsUI();

            const selLocation = document.getElementById("sel-location");
            if (selLocation.value === r.location) {
                state.appResults = state.fullResults.filter(x => x.location === r.location);
                if (state.appResults.length > 0) {
                    document.getElementById("carousel-counter").textContent = `Item ${state.currentIndex + 1} of ${state.appResults.length}`;
                    document.getElementById("btn-next").disabled = (state.currentIndex === state.appResults.length - 1);
                }
            }
            if (state.fullResults.length === 1) updateCarousel(true);
        }
    };
}

export async function executeJob() {
    const fd = new FormData();
    if (state.modelFile) fd.append("model", state.modelFile);
    
    const chkIs360 = document.getElementById("chk-is-360");
    const chkLayerFront = document.getElementById("chk-layer-front");
    const chkLayerRear = document.getElementById("chk-layer-rear");
    const layerTogglePanel = document.getElementById("layer-toggle-panel");
    const containerBevRear = document.getElementById("container-bev-rear");

    // Standard Options
    fd.append("cam_height", document.getElementById("cam-height").value);
    fd.append("is_360", chkIs360.checked ? "true" : "false");
    fd.append("draw_grid", document.getElementById("chk-draw-grid").checked ? "true" : "false");
    fd.append("interval_m", document.getElementById("interval-m").value);
    
    // Advanced Options
    fd.append("comp_roll", document.getElementById("chk-comp-roll").checked ? "true" : "false");
    fd.append("comp_pitch", document.getElementById("chk-comp-pitch").checked ? "true" : "false");
    fd.append("undistort", document.getElementById("chk-undistort").checked ? "true" : "false");
    fd.append("ego_mask", document.getElementById("chk-ego-mask").checked ? "true" : "false");
    fd.append("conf_thresh", document.getElementById("num-conf").value);
    
    if(state.stateLastLat !== null) fd.append("last_lat", state.stateLastLat);
    if(state.stateLastLon !== null) fd.append("last_lon", state.stateLastLon);
    fd.append("last_loc_id", state.stateLastLocId);
    
    state.imageFiles.forEach(f => fd.append("images", f));

    state.appIs360 = chkIs360.checked;
    chkLayerFront.checked = true;
    chkLayerRear.checked = true;
    
    if (!state.appIs360) { 
        containerBevRear.classList.add("hidden"); 
        layerTogglePanel.classList.add("hidden"); layerTogglePanel.classList.remove("flex");
        setView('front'); 
    } else { 
        containerBevRear.classList.remove("hidden");
        layerTogglePanel.classList.remove("hidden"); layerTogglePanel.classList.add("flex");
    }

    document.getElementById("upload-panel").classList.add("hidden");
    document.getElementById("workspace").classList.remove("hidden");
    document.getElementById("btn-save-project").classList.remove("hidden");
    document.getElementById("btn-export-zip").classList.remove("hidden");
    document.getElementById("btn-export-flat-zip").classList.remove("hidden");
    document.getElementById("btn-toggle-map").classList.remove("hidden");
    document.getElementById("progress-container").classList.remove("hidden");
    
    state.warningCount = 0;
    state.layoutPrefs.mapOn.isManual = false;
    state.layoutPrefs.mapOff.isManual = false;
    
    document.getElementById("warnings-badge").textContent = "0";
    document.getElementById("warnings-badge").classList.add("hidden");
    document.getElementById("btn-show-warnings").classList.add("hidden");
    document.getElementById("warnings-list").innerHTML = "";
    document.getElementById("no-warnings-msg").classList.remove("hidden");
    
    const btnCancel = document.getElementById("btn-cancel-job");
    btnCancel.disabled = false; btnCancel.textContent = "Stop / Cancel";
    
    const telemetryHud = document.getElementById("telemetry-hud");
    telemetryHud.innerHTML = ""; telemetryHud.classList.add("hidden");
    document.getElementById("process-btn").disabled = true;
    
    clearOrthomosaics(); 
    state.fullGeojson = { type: "FeatureCollection", features: [] };
    state.nodesGeoJson = { type: "FeatureCollection", features: [] };
    state.trailGeoJson = { type: "FeatureCollection", features: [] };
    state.fullResults = []; state.appResults = [];

    initMap(handleMapClick);

    try {
        const res = await fetch("/process", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || "Unknown server error");

        state.stateLastLat = data.last_lat;
        state.stateLastLon = data.last_lon;
        state.stateLastLocId = data.last_loc_id;
        state.currentTaskId = data.task_id;

        if (data.initial_trail && data.initial_trail.features) {
            state.trailGeoJson = data.initial_trail;
            updateMapSource('trail-source', state.trailGeoJson);
            fitMapToBounds(state.trailGeoJson);
        }

        data.initial_state.forEach(img => {
            if(img.lat !== null && img.lon !== null) {
                state.nodesGeoJson.features.push({
                    type: "Feature", geometry: { type: "Point", coordinates: [img.lon, img.lat] },
                    properties: { original_name: img.original_name, location: img.location, processed: false, active: false }
                });
            }
        });
        updateMapSource('nodes-source', state.nodesGeoJson);
        setTimeout(() => { if (state.map) state.map.resize(); }, 300);
        
        startSSE(data.task_id, data.total_images);
    } catch (e) {
        alert(e.message); 
        document.getElementById("upload-panel").classList.remove("hidden"); 
        document.getElementById("progress-container").classList.add("hidden");
        document.getElementById("btn-toggle-map").classList.add("hidden");
        checkCanProcess();
    }
}