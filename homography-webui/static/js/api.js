import { state } from './state.js';
import { initMap, clearOrthomosaics, addOrthomosaicShingle, updateMapSource, fitMapToBounds, setPassPairsData } from './map.js';
import { refreshLocationsUI, updateCarousel, setView, checkCanProcess, handleMapClick, addWarning } from './ui.js';

export async function exportProject(projectState, btnId, filename) {
    const btn = document.getElementById(btnId);
    const originalText = btn.innerHTML;
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> SAVING...`; 
    btn.disabled = true;
    try {
        const res = await fetch("/export-project", {
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(projectState) 
        });
        if (!res.ok) throw new Error("Failed to export project");
        const blob = await res.blob(); 
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a'); 
        a.href = url; a.download = filename; 
        document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
    } catch (err) { 
        alert(err.message); 
    } finally { 
        btn.innerHTML = originalText; btn.disabled = false; 
    }
}

export async function importProject(file) {
    const fd = new FormData();
    fd.append("project_zip", file);
    const res = await fetch("/import-project", {
        method: "POST",
        body: fd
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || "Failed to import project");
    return data.project_state;
}

export async function triggerZipExport(endpoint, btnId, filename) {
    if (state.fullResults.length === 0) return;
    const btn = document.getElementById(btnId); 
    const originalText = btn.innerHTML;
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> PACKAGING...`; 
    btn.disabled = true;
    try {
        const res = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ results: state.fullResults }) });
        if (!res.ok) throw new Error("Failed to compile ZIP file");
        const blob = await res.blob(); const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
    } catch (err) { alert(err.message); } finally { btn.innerHTML = originalText; btn.disabled = false; }
}

export async function cancelJob() {
    if (!state.currentTaskId) return;
    if (!confirm("Confirm hard abort? Completed frames will be retained.")) return;
    const btn = document.getElementById("btn-cancel-job");
    btn.disabled = true; btn.innerHTML = "ABORTING...";
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

export async function fetchFrameTrace(filename) {
    const res = await fetch(`/trace/${encodeURIComponent(filename)}`);
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || "No trace data available for this frame");
    return data;
}

export async function fetchPassDiagnostics(minIndexGap, maxDistM) {
    const res = await fetch("/diagnose_passes", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results: state.fullResults, min_index_gap: minIndexGap, max_dist_m: maxDistM })
    });
    const data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || "Failed to run pass diagnostics");
    return data;
}

// Generates a SAM2 polygon preview dynamically for a single frame's rect/BEV space
export async function fetchSam2Preview(filename, view, calibrationConfig, points) {
    const res = await fetch("/preview_sam2", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, view, calibration: calibrationConfig, points })
    });
    const data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || "Failed to generate SAM2 mask.");
    return data.points;
}

// Generates a SAM2 polygon preview directly on a stitched multi-frame
// corridor image (used when the mask editor has more than one frame
// loaded). Points are normalized 0-1 relative to the corridor image
// itself, NOT to any single source frame's rect/BEV space.
export async function fetchSam2PreviewCorridor(corridorUrl, points) {
    const res = await fetch("/preview_sam2_corridor", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corridor_url: corridorUrl, points })
    });
    const data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || "Failed to generate SAM2 mask on corridor.");
    return data.points;
}

function startSSE(taskId, totalImages) {
    const source = new EventSource(`/stream/${taskId}`);
    let processedCount = 0; let startTime = Date.now();
    const telemetryHud = document.getElementById("telemetry-hud");

    document.getElementById("progress-bar").max = totalImages > 0 ? totalImages : 100;

    source.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === "error" || msg.type === "complete" || msg.type === "cancelled") {
            source.close();
            state.currentTaskId = null;
            document.getElementById("progress-container").classList.add("hidden");
            checkCanProcess();
            state.imageFiles = [];
            document.getElementById("name-image").textContent = "EXECUTION COMPLETE.";
            
            if (msg.type === "error") {
                alert(`SYSTEM ERROR: ${msg.message}`);
            } else if (state.fullResults.length === 0) {
                alert(msg.type === "cancelled" ? "Process aborted. No outputs generated." : "Process completed, but 0 frames successfully extracted. Review logs.");
                document.getElementById("workspace").classList.add("hidden");
                document.getElementById("upload-panel").classList.remove("hidden");
                document.getElementById("btn-save-project").classList.add("hidden");
                document.getElementById("btn-export-zip").classList.add("hidden");
                document.getElementById("btn-export-flat-zip").classList.add("hidden");
                document.getElementById("btn-toggle-map").classList.add("hidden");
                document.getElementById("btn-analyze-passes").classList.add("hidden");
                document.getElementById("btn-group-defects").classList.add("hidden");
            } else if (msg.type === "cancelled") {
                addWarning(`[SYSTEM] User aborted process. Partial state saved.`);
            }
            return;
        }

        if (msg.type === "health_report") {
            telemetryHud.classList.remove("hidden");
            const hr = msg.data;
            let gpsColor = hr.gps_score > 80 ? 'text-green' : 'text-orange';
            let imuColor = hr.imu_score > 90 ? 'text-green' : 'text-orange';

            const hudLine = document.createElement("div");
            hudLine.innerHTML = `<strong>[${msg.original_name}]</strong> GPS: <span class="${gpsColor}">${hr.gps_score.toFixed(0)}%</span> | IMU: <span class="${imuColor}">${hr.imu_score.toFixed(0)}%</span> | Drift: ${hr.metrics.avg_gps_speed_error_ms.toFixed(2)}m/s`;
            telemetryHud.appendChild(hudLine);
            
            if (hr.warnings.length > 0) {
                hr.warnings.forEach(w => addWarning(`[${msg.original_name} TELEMETRY] ${w}`));
            }
            return;
        }

        if (msg.type === "item_error") {
            addWarning(`[SKIPPED] ${msg.original_name}: ${msg.message}`);
            if (!msg.is_video) {
                processedCount++;
                document.getElementById("progress-bar").value = processedCount;
                document.getElementById("progress-text").textContent = `SEGMENTING ${processedCount} / ${totalImages}`;
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
            document.getElementById("progress-bar").value = processedCount;
            document.getElementById("progress-text").textContent = `SEGMENTING ${processedCount} / ${totalImages}`;

            const elapsedSec = (Date.now() - startTime) / 1000;
            const remainSec = Math.ceil((totalImages - processedCount) * (elapsedSec / processedCount));
            document.getElementById("eta-text").textContent = `ETA: ${Math.floor(remainSec / 60)}M ${remainSec % 60}S`;

            refreshLocationsUI();

            const selLocation = document.getElementById("sel-location");
            if (selLocation.value === r.location) {
                state.appResults = state.fullResults.filter(x => x.location === r.location);
                if (state.appResults.length > 0) {
                    document.getElementById("carousel-counter").textContent = `ITEM ${state.currentIndex + 1} OF ${state.appResults.length}`;
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
    fd.append("media_type", document.getElementById("sel-media-type").value);
    fd.append("has_telemetry", document.getElementById("chk-has-telemetry").checked ? "true" : "false");
    fd.append("cam_height", document.getElementById("cam-height").value);
    fd.append("is_360", chkIs360.checked ? "true" : "false");
    fd.append("draw_grid", document.getElementById("chk-draw-grid").checked ? "true" : "false");
    fd.append("interval_m", document.getElementById("interval-m").value);
    
    // Advanced Options
    fd.append("undistort", document.getElementById("chk-undistort").checked ? "true" : "false");
    fd.append("ego_mask", document.getElementById("chk-ego-mask").checked ? "true" : "false");
    fd.append("skip_ai", document.getElementById("chk-skip-ai").checked ? "true" : "false");
    fd.append("conf_thresh", document.getElementById("num-conf").value);

    // Grid Size Options
    fd.append("z_near", document.getElementById("grid-z-near").value);
    fd.append("z_far", document.getElementById("grid-z-far").value);
    fd.append("lane_width", document.getElementById("grid-lane-width").value);
    fd.append("gps_lag_sec", document.getElementById("gps-lag-sec").value);

    // GPS antenna -> camera lever-arm offset
    const camOffFwd = document.getElementById("cam-offset-forward");
    const camOffRight = document.getElementById("cam-offset-right");
    fd.append("cam_offset_forward_m", camOffFwd ? camOffFwd.value : "0.0");
    fd.append("cam_offset_right_m", camOffRight ? camOffRight.value : "0.0");
    
    if(state.stateLastLat !== null) fd.append("last_lat", state.stateLastLat);
    if(state.stateLastLon !== null) fd.append("last_lon", state.stateLastLon);
    fd.append("last_loc_id", state.stateLastLocId);
    
    state.imageFiles.forEach(f => fd.append("images", f));

    state.appIs360 = chkIs360.checked;
    chkLayerFront.checked = true;
    chkLayerRear.checked = true;
    
    if (!state.appIs360) { 
        containerBevRear.classList.add("hidden"); 
        layerTogglePanel.classList.add("hidden");
        setView('front'); 
    } else { 
        containerBevRear.classList.remove("hidden");
        layerTogglePanel.classList.remove("hidden"); 
    }

    document.getElementById("upload-panel").classList.add("hidden");
    document.getElementById("workspace").classList.remove("hidden");
    document.getElementById("btn-save-project").classList.remove("hidden");
    document.getElementById("btn-export-zip").classList.remove("hidden");
    document.getElementById("btn-export-flat-zip").classList.remove("hidden");
    document.getElementById("btn-toggle-map").classList.remove("hidden");
    document.getElementById("btn-analyze-passes").classList.remove("hidden");
    document.getElementById("btn-group-defects").classList.remove("hidden");
    document.getElementById("progress-container").classList.remove("hidden");
    
    state.warningCount = 0;
    state.layoutPrefs.mapOn.isManual = false;
    state.layoutPrefs.mapOff.isManual = false;
    
    document.getElementById("warnings-badge").textContent = "0";
    document.getElementById("btn-show-warnings").classList.add("hidden");
    document.getElementById("warnings-list").innerHTML = "";
    document.getElementById("no-warnings-msg").classList.remove("hidden");
    
    const btnCancel = document.getElementById("btn-cancel-job");
    btnCancel.disabled = false; btnCancel.innerHTML = `<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg> ABORT`;
    
    const telemetryHud = document.getElementById("telemetry-hud");
    telemetryHud.innerHTML = ""; telemetryHud.classList.add("hidden");
    document.getElementById("process-btn").disabled = true;
    document.getElementById("progress-bar").value = 0;
    
    clearOrthomosaics(); 
    state.fullGeojson = { type: "FeatureCollection", features: [] };
    state.nodesGeoJson = { type: "FeatureCollection", features: [] };
    state.trailGeoJson = { type: "FeatureCollection", features: [] };
    state.passPairsGeoJson = { type: "FeatureCollection", features: [] };
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
        document.getElementById("btn-analyze-passes").classList.add("hidden");
        document.getElementById("btn-group-defects").classList.add("hidden");
        checkCanProcess();
    }
}

