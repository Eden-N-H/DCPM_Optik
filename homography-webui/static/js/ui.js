import { state } from './state.js';
import { updateMapSource, clearOrthomosaics, addOrthomosaicShingle } from './map.js';
import { stringToColor } from './utils.js';
import { fetchGridPreview, recalculateProject, autoDetectVP, clickManualVP } from './api.js';

export function handleMapClick(originalName) {
    const target = state.fullResults.find(r => r.original_name === originalName);
    if (target) {
        const selLocation = document.getElementById("sel-location");
        if (selLocation.value !== target.location) {
            selLocation.value = target.location;
            state.appResults = state.fullResults.filter(r => r.location === target.location);
        }
        state.currentIndex = state.appResults.findIndex(r => r.original_name === target.original_name);
        updateCarousel(false);
    }
}

export function checkCanProcess() {
    const btnProcess = document.getElementById("process-btn");
    const hasModel = (state.isModelLoaded || state.modelFile !== null);
    btnProcess.disabled = !(hasModel && state.imageFiles.length > 0);
}

const handleFiles = (files, isMulti, callback, nameElement) => {
    if (!files || !files.length) return;
    if (isMulti) { 
        callback(Array.from(files)); 
        nameElement.textContent = `${files.length} items queued`; 
    } 
    else { 
        callback(files[0]); 
        nameElement.textContent = files[0].name; 
        document.getElementById("status-model").classList.remove("hidden"); 
        state.isModelLoaded = true; 
    }
    nameElement.classList.remove("hidden");
    checkCanProcess();
};

export function setupDz(dzId, inId, nameId, isMulti, callback) {
    const dz = document.getElementById(dzId);
    const inp = document.getElementById(inId);
    const nm = document.getElementById(nameId);
    
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add("bg-blue-50", "border-blue-500"); });
    dz.addEventListener('dragenter', (e) => { e.preventDefault(); dz.classList.add("bg-blue-50", "border-blue-500"); });
    dz.addEventListener('dragleave', (e) => { e.preventDefault(); dz.classList.remove("bg-blue-50", "border-blue-500"); });
    
    dz.addEventListener('drop', (e) => { 
        e.preventDefault(); 
        dz.classList.remove("bg-blue-50", "border-blue-500"); 
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            inp.files = e.dataTransfer.files; 
            handleFiles(e.dataTransfer.files, isMulti, callback, nm); 
        }
    });
    
    inp.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files, isMulti, callback, nm);
        }
    });
}

export function refreshLocationsUI() {
    const selLocation = document.getElementById("sel-location");
    const locations = [...new Set(state.fullResults.map(r => r.location))];
    const currentSelection = selLocation.value;
    
    selLocation.innerHTML = locations.map(loc => `<option value="${loc}">${loc}</option>`).join("");
    if (locations.includes(currentSelection)) selLocation.value = currentSelection;
    else if (locations.length > 0) selLocation.value = locations[0];

    selLocation.onchange = () => { 
        state.appResults = state.fullResults.filter(r => r.location === selLocation.value); 
        state.currentIndex = 0; 
        updateCarousel(true); 
    };
    
    state.appResults = state.fullResults.filter(r => r.location === selLocation.value);
    if(state.appResults.length > 0 && document.getElementById("img-rect").classList.contains("hidden")) updateCarousel(false);
}

export function setView(dir) {
    state.currentDirection = dir;
    const contF = document.getElementById('container-bev-front'), contR = document.getElementById('container-bev-rear');
    const activeLabel = document.getElementById('label-active-view');
    
    if (dir === 'front') {
        contF.classList.add('border-blue-500', 'ring-2'); contF.classList.remove('border-transparent');
        contR.classList.remove('border-blue-500', 'ring-2'); contR.classList.add('border-transparent');
        activeLabel.textContent = 'Front View Active';
    } else {
        contR.classList.add('border-blue-500', 'ring-2'); contR.classList.remove('border-transparent');
        contF.classList.remove('border-blue-500', 'ring-2'); contF.classList.add('border-transparent');
        activeLabel.textContent = 'Rear View Active';
    }
    updateCarousel(false);
}

export function updateCarousel(panMap = true) {
    if (state.appResults.length === 0) return;
    const current = state.appResults[state.currentIndex];
    
    const imgRect = document.getElementById("img-rect");
    const imgBevFront = document.getElementById("img-bev-front");
    const imgBevRear = document.getElementById("img-bev-rear");

    document.getElementById("placeholder-rect").classList.add("hidden");
    imgRect.classList.remove("hidden");
    imgBevFront.classList.remove("hidden");
    if (state.appIs360) imgBevRear.classList.remove("hidden");

    const activeViewData = current.views[state.currentDirection] || current.views['front'];
    document.getElementById("carousel-counter").textContent = `Item ${state.currentIndex + 1} of ${state.appResults.length}`;
    document.getElementById("carousel-filename").textContent = current.original_name;
    document.getElementById("carousel-telemetry").textContent = `Pitch: ${current.pitch}° | Roll: ${current.roll}°`;

    imgRect.onload = () => { autoFitSplitters(true); };

    const ts = Date.now();
    
    imgBevFront.src = current.views['front'].bev_url + `?t=${ts}`;
    if (state.appIs360 && current.views['rear']) imgBevRear.src = current.views['rear'].bev_url + `?t=${ts}`;
    imgRect.src = activeViewData.rect_url + `?t=${ts}`;

    document.getElementById("table-defects").innerHTML = activeViewData.defects.map(d => `<tr><td class="p-2"><span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: ${d.color || stringToColor(d.class)}; border: 1px solid #ccc;"></span>${d.class}</td><td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td><td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td></tr>`).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No detections</td></tr>`;

    state.activeMarkerFilename = current.original_name;
    state.nodesGeoJson.features.forEach(f => { f.properties.active = (f.properties.original_name === state.activeMarkerFilename); });
    updateMapSource('nodes-source', state.nodesGeoJson);

    if (panMap && state.mapLoaded && current.lat !== null && state.isMapVisible) {
        state.map.flyTo({ center: [current.lon, current.lat], zoom: 20, speed: 1.5 });
    }

    document.getElementById("btn-prev").disabled = (state.currentIndex === 0);
    document.getElementById("btn-next").disabled = (state.currentIndex === state.appResults.length - 1);
}

export function addWarning(message) {
    const list = document.getElementById("warnings-list");
    const badge = document.getElementById("warnings-badge");
    const btn = document.getElementById("btn-show-warnings");
    document.getElementById("no-warnings-msg").classList.add("hidden");
    
    const li = document.createElement("li");
    li.textContent = message;
    list.appendChild(li);
    
    state.warningCount++;
    badge.textContent = state.warningCount;
    badge.classList.remove("hidden");
    btn.classList.remove("hidden");
}

export function toggleWarningsModal(show) {
    const modal = document.getElementById("warnings-modal");
    if (show) modal.classList.remove("hidden");
    else modal.classList.add("hidden");
}

export function setupCalibrationUI() {
    const btnOpen = document.getElementById("btn-open-calibrate");
    const btnClose = document.getElementById("btn-close-calibrate");
    const btnCancel = document.getElementById("btn-cancel-calibrate");
    const btnApply = document.getElementById("btn-apply-calibrate");
    const btnAutoVP = document.getElementById("btn-auto-vp");
    
    const modal = document.getElementById("calibrate-modal");
    const previewImg = document.getElementById("img-calibrate-preview");
    const loader = document.getElementById("calibrate-loader");
    
    // Swapped x_range for lane_width
    const inputs = ['pitch_offset', 'roll_offset', 'yaw_offset', 'fov', 'cam_height', 'z_near', 'z_far', 'lane_width'];
    
    const getCalibrationValues = () => {
        let calib = {};
        inputs.forEach(id => {
            const el = document.getElementById(`calib-${id}`);
            if (el) calib[id] = parseFloat(el.value);
        });
        return calib;
    };

    const updateLabels = () => {
        inputs.forEach(id => {
            const el = document.getElementById(`calib-${id}`);
            const lbl = document.getElementById(`lbl-${id}`);
            if (el && lbl) {
                const unit = ['pitch_offset', 'roll_offset', 'yaw_offset', 'fov'].includes(id) ? '°' : 'm';
                lbl.textContent = `${el.value}${unit}`;
            }
        });
    };

    const setCalibrationValues = (calib) => {
        inputs.forEach(id => {
            const el = document.getElementById(`calib-${id}`);
            if (el && calib[id] !== undefined) {
                let val = calib[id];
                if(el.min) val = Math.max(parseFloat(el.min), val);
                if(el.max) val = Math.min(parseFloat(el.max), val);
                el.value = val;
            }
        });
        updateLabels();
    };

    const triggerPreviewUpdate = async () => {
        loader.classList.remove("hidden");
        try {
            const current = state.appResults[state.currentIndex];
            const b64 = await fetchGridPreview(current.filename, state.currentDirection, getCalibrationValues());
            previewImg.src = b64;
        } catch(err) { console.error(err); }
        finally { loader.classList.add("hidden"); }
    };

    const close = () => modal.classList.add("hidden");
    
    btnOpen.onclick = () => {
        if(state.appResults.length === 0) return;
        const current = state.appResults[state.currentIndex];
        const baseCalib = current.views[state.currentDirection].calibration || {
            pitch_offset: 0, roll_offset: 0, yaw_offset: 0,
            fov: 100, cam_height: 1.6, z_near: 1.9, z_far: 10.0, lane_width: 8.0
        };
        setCalibrationValues(baseCalib);
        previewImg.src = current.views[state.currentDirection].rect_url;
        modal.classList.remove("hidden");
    };
    
    btnClose.onclick = close;
    btnCancel.onclick = close;

    let previewTimeout = null;
    inputs.forEach(id => {
        const el = document.getElementById(`calib-${id}`);
        if (el) {
            el.addEventListener("input", () => {
                updateLabels();
                clearTimeout(previewTimeout);
                previewTimeout = setTimeout(triggerPreviewUpdate, 350);
            });
        }
    });

    btnAutoVP.onclick = async () => {
        btnAutoVP.disabled = true;
        btnAutoVP.textContent = "Analyzing...";
        try {
            const current = state.appResults[state.currentIndex];
            const data = await autoDetectVP(current.filename, state.currentDirection, getCalibrationValues());
            if (data.success) {
                setCalibrationValues(data.calibration);
                triggerPreviewUpdate();
            } else {
                alert(data.error);
            }
        } catch(err) { console.error(err); }
        finally {
            btnAutoVP.disabled = false;
            btnAutoVP.textContent = "🪄 Auto-Detect VP";
        }
    };

    previewImg.addEventListener('click', async (e) => {
        const rect = previewImg.getBoundingClientRect();
        
        const nw = previewImg.naturalWidth;
        const nh = previewImg.naturalHeight;
        if(!nw || !nh) return;
        
        const scale = Math.min(rect.width / nw, rect.height / nh);
        const wRendered = nw * scale;
        const hRendered = nh * scale;
        
        const xOffset = (rect.width - wRendered) / 2;
        const yOffset = (rect.height - hRendered) / 2;
        
        const clickX = e.clientX - rect.left - xOffset;
        const clickY = e.clientY - rect.top - yOffset;
        
        if (clickX < 0 || clickX > wRendered || clickY < 0 || clickY > hRendered) return; 
        
        const px = clickX / wRendered;
        const py = clickY / hRendered;
        
        loader.classList.remove("hidden");
        try {
            const current = state.appResults[state.currentIndex];
            const data = await clickManualVP(current.filename, state.currentDirection, getCalibrationValues(), px, py);
            if (data.success) {
                setCalibrationValues(data.calibration);
                triggerPreviewUpdate();
            }
        } catch(err) { console.error(err); }
    });

    btnApply.onclick = async () => {
        btnApply.disabled = true; btnApply.textContent = "Re-running AI Inference...";
        btnCancel.disabled = true;
        try {
            const newResults = await recalculateProject(getCalibrationValues());
            state.fullResults = newResults;
            refreshLocationsUI(); 
            
            state.fullGeojson.features = [];
            state.fullResults.forEach(r => state.fullGeojson.features.push(...r.geojson));
            updateMapSource('defects-source', state.fullGeojson);
            
            clearOrthomosaics();
            const chkF = document.getElementById("chk-layer-front").checked;
            const chkR = document.getElementById("chk-layer-rear").checked;
            state.fullResults.forEach(r => addOrthomosaicShingle(r, chkF, chkR));
            
            updateCarousel(false);
            close();
        } catch(err) {
            alert("Failed to recalculate: " + err.message);
        } finally {
            btnApply.disabled = false; btnApply.textContent = "Apply to Project";
            btnCancel.disabled = false;
        }
    };
}

export function autoFitSplitters(adjustMain = false, force = false) {
    const isMapOn = state.isMapVisible;
    const prefs = isMapOn ? state.layoutPrefs.mapOn : state.layoutPrefs.mapOff;
    
    if (!force && prefs.isManual) return; 

    const imgRect = document.getElementById("img-rect");
    const imgBevFront = document.getElementById("img-bev-front");

    if (!imgRect.complete || !imgRect.naturalWidth || !imgBevFront.complete || !imgBevFront.naturalWidth) {
        setTimeout(() => autoFitSplitters(adjustMain, force), 100);
        return;
    }

    const arRect = imgRect.naturalWidth / imgRect.naturalHeight;
    const arBev = imgBevFront.naturalWidth / imgBevFront.naturalHeight;
    const is360 = state.appIs360;

    const perspectiveContainer = document.getElementById("perspective-container");
    const imagePanel = document.getElementById("image-panel");
    const workspace = document.getElementById("workspace");
    
    if (isMapOn) {
        const hRectRel = 1 / arRect;
        const hBevRel = 1 / ((is360 ? 2 : 1) * arBev);
        
        const splitPct = (hRectRel / (hRectRel + hBevRel)) * 100;
        prefs.mediaBasis = `${Math.max(20, Math.min(splitPct, 80))}%`;
        perspectiveContainer.style.flexBasis = prefs.mediaBasis;

        if (adjustMain) {
            const hAvail = workspace.clientHeight - 150;
            const idealW = hAvail / (hRectRel + hBevRel);
            const mainSplitPct = (idealW / workspace.clientWidth) * 100;
            prefs.mainW = `${Math.max(25, Math.min(mainSplitPct, 75))}%`;
            imagePanel.style.width = prefs.mainW;
            if (state.map) state.map.resize();
        }
    } else {
        const wRectRel = arRect;
        const wBevRel = ((is360 ? 0.5 : 1) * arBev);
        
        const splitPct = (wRectRel / (wRectRel + wBevRel)) * 100;
        prefs.mediaBasis = `${Math.max(20, Math.min(splitPct, 80))}%`;
        perspectiveContainer.style.flexBasis = prefs.mediaBasis;
    }

    if (force) prefs.isManual = false; 
}

export function toggleMapView() {
    state.isMapVisible = !state.isMapVisible;
    
    const mapPanel = document.getElementById("map-panel");
    const mainSplitter = document.getElementById("main-splitter");
    const imagePanel = document.getElementById("image-panel");
    
    const mediaLayout = document.getElementById("media-layout-container");
    const mediaSplitter = document.getElementById("media-splitter");
    const perspectiveContainer = document.getElementById("perspective-container");
    const btnToggleMap = document.getElementById("btn-toggle-map");

    if (state.isMapVisible) {
        mapPanel.classList.remove("hidden");
        mainSplitter.classList.remove("hidden");
        
        mediaLayout.classList.remove("flex-row"); mediaLayout.classList.add("flex-col");
        mediaSplitter.classList.remove("w-2", "h-full", "cursor-col-resize");
        mediaSplitter.classList.add("h-2", "w-full", "cursor-row-resize");
        
        btnToggleMap.classList.remove("bg-blue-100", "text-blue-800", "border-blue-300");
        btnToggleMap.classList.add("bg-gray-200", "text-gray-800");

        if (!state.layoutPrefs.mapOn.isManual) {
            autoFitSplitters(true);
        } else {
            imagePanel.style.width = state.layoutPrefs.mapOn.mainW;
            perspectiveContainer.style.flexBasis = state.layoutPrefs.mapOn.mediaBasis;
        }

    } else {
        mapPanel.classList.add("hidden");
        mainSplitter.classList.add("hidden");
        imagePanel.style.width = "100%";
        
        mediaLayout.classList.remove("flex-col"); mediaLayout.classList.add("flex-row");
        mediaSplitter.classList.remove("h-2", "w-full", "cursor-row-resize");
        mediaSplitter.classList.add("w-2", "h-full", "cursor-col-resize");
        
        btnToggleMap.classList.add("bg-blue-100", "text-blue-800", "border-blue-300");
        btnToggleMap.classList.remove("bg-gray-200", "text-gray-800");

        if (!state.layoutPrefs.mapOff.isManual) {
            autoFitSplitters(false);
        } else {
            perspectiveContainer.style.flexBasis = state.layoutPrefs.mapOff.mediaBasis;
        }
    }

    setTimeout(() => {
        if(state.map) state.map.resize();
        updateCarousel(true); 
    }, 50);
}

export function initResizers() {
    const mainSplitter = document.getElementById("main-splitter");
    const imagePanel = document.getElementById("image-panel");
    const workspace = document.getElementById("workspace");
    const mapPanel = document.getElementById("map-panel"); 

    mainSplitter.addEventListener("dblclick", () => autoFitSplitters(true, true));

    let isDraggingMain = false;
    mainSplitter.addEventListener("mousedown", () => {
        isDraggingMain = true;
        document.body.classList.add("select-none");
        mapPanel.style.pointerEvents = "none";
    });

    const mediaSplitter = document.getElementById("media-splitter");
    const perspectiveContainer = document.getElementById("perspective-container");
    const mediaLayout = document.getElementById("media-layout-container");
    
    mediaSplitter.addEventListener("dblclick", () => autoFitSplitters(false, true));

    let isDraggingMedia = false;
    mediaSplitter.addEventListener("mousedown", () => {
        isDraggingMedia = true;
        document.body.classList.add("select-none");
    });

    document.addEventListener("mousemove", (e) => {
        if (isDraggingMain) {
            const rect = workspace.getBoundingClientRect();
            let newWidth = ((e.clientX - rect.left) / rect.width) * 100;
            newWidth = Math.max(25, Math.min(newWidth, 75)); 
            imagePanel.style.width = `${newWidth}%`;
            if (state.map) state.map.resize();
        }
        if (isDraggingMedia) {
            const rect = mediaLayout.getBoundingClientRect();
            if (state.isMapVisible) {
                let newHeight = ((e.clientY - rect.top) / rect.height) * 100;
                newHeight = Math.max(15, Math.min(newHeight, 85));
                perspectiveContainer.style.flexBasis = `${newHeight}%`;
            } else {
                let newWidth = ((e.clientX - rect.left) / rect.width) * 100;
                newWidth = Math.max(15, Math.min(newWidth, 85));
                perspectiveContainer.style.flexBasis = `${newWidth}%`;
            }
        }
    });

    document.addEventListener("mouseup", () => {
        if (isDraggingMain || isDraggingMedia) {
            if (state.isMapVisible) {
                state.layoutPrefs.mapOn.isManual = true;
                state.layoutPrefs.mapOn.mainW = imagePanel.style.width;
                state.layoutPrefs.mapOn.mediaBasis = perspectiveContainer.style.flexBasis;
            } else {
                state.layoutPrefs.mapOff.isManual = true;
                state.layoutPrefs.mapOff.mediaBasis = perspectiveContainer.style.flexBasis;
            }

            isDraggingMain = false;
            isDraggingMedia = false;
            document.body.classList.remove("select-none");
            mapPanel.style.pointerEvents = "auto";
            if (state.map) state.map.resize();
        }
    });
}