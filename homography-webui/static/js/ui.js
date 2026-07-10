import { state } from './state.js';
import { updateMapSource, clearOrthomosaics, addOrthomosaicShingle } from './map.js';
import { stringToColor } from './utils.js';
import { fetchGridPreview, recalculateProject, autoDetectVP, clickManualVP, fetchSam2Preview } from './api.js';

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
    const chkSkipAi = document.getElementById("chk-skip-ai");
    const skipAi = chkSkipAi ? chkSkipAi.checked : false;
    const hasModel = (state.isModelLoaded || state.modelFile !== null);
    
    btnProcess.disabled = !((hasModel || skipAi) && state.imageFiles.length > 0);
}

const handleFiles = (files, isMulti, callback, nameElement) => {
    if (!files || !files.length) return;
    if (isMulti) { 
        callback(Array.from(files)); 
        nameElement.textContent = `${files.length} ITEMS QUEUED`; 
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
    
    // Add proxy click listener to open the file selector natively
    dz.addEventListener('click', () => inp.click());
    
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add("drag-active"); });
    dz.addEventListener('dragenter', (e) => { e.preventDefault(); dz.classList.add("drag-active"); });
    dz.addEventListener('dragleave', (e) => { e.preventDefault(); dz.classList.remove("drag-active"); });
    
    dz.addEventListener('drop', (e) => { 
        e.preventDefault(); 
        dz.classList.remove("drag-active"); 
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
    if (state.currentDirection === dir) {
        openFullscreen('bev');
        return;
    }
    
    state.currentDirection = dir;
    const contF = document.getElementById('container-bev-front');
    const contR = document.getElementById('container-bev-rear');
    const activeLabel = document.getElementById('label-active-view');
    
    if (dir === 'front') {
        contF.classList.add('active-view');
        contR.classList.remove('active-view');
        activeLabel.textContent = 'FRONT ACTIVE';
    } else {
        contR.classList.add('active-view');
        contF.classList.remove('active-view');
        activeLabel.textContent = 'REAR ACTIVE';
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

    imgRect.onload = () => { autoFitSplitters(true); };

    const activeViewData = current.views[state.currentDirection] || current.views['front'];
    document.getElementById("carousel-counter").textContent = `ITEM ${state.currentIndex + 1} OF ${state.appResults.length}`;
    document.getElementById("carousel-filename").textContent = current.original_name;
    document.getElementById("carousel-telemetry").textContent = `P: ${current.pitch}° | R: ${current.roll}°`;

    const ts = Date.now();
    imgBevFront.src = current.views['front'].bev_url.split('?')[0] + `?t=${ts}`;
    if (state.appIs360 && current.views['rear']) {
        imgBevRear.src = current.views['rear'].bev_url.split('?')[0] + `?t=${ts}`;
    }
    imgRect.src = activeViewData.rect_url.split('?')[0] + `?t=${ts}`;

    // Pass the exact backend detection index (d.det_idx) instead of visual array loop idx
    document.getElementById("table-defects").innerHTML = activeViewData.defects.map((d) => `
        <tr>
            <td><span class="color-dot" style="background-color: ${d.color || stringToColor(d.class)};"></span>${d.class}</td>
            <td>${(d.conf*100).toFixed(0)}%</td>
            <td><strong>${d.area_sqm} m²</strong></td>
            <td class="text-right">
                <button onclick="window.startEditDefect(${d.det_idx})" class="action-icon" title="Change Class">
                    <svg viewBox="0 0 24 24"><polygon points="16 3 21 8 8 21 3 21 3 16 16 3"/></svg>
                </button>
                <button onclick="window.startReoutlineDefect(${d.det_idx})" class="action-icon" title="Re-outline">
                    <svg viewBox="0 0 24 24"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
                </button>
                <button onclick="window.deleteDefect(${d.det_idx})" class="action-icon" title="Delete">
                    <svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                </button>
            </td>
        </tr>
    `).join('') || `<tr><td colspan="4" style="text-align:center; padding:10px; color:var(--muted);">NO DETECTIONS FOUND</td></tr>`;

    state.activeMarkerFilename = current.original_name;
    state.nodesGeoJson.features.forEach(f => { f.properties.active = (f.properties.original_name === state.activeMarkerFilename); });
    updateMapSource('nodes-source', state.nodesGeoJson);

    if (panMap && state.mapLoaded && current.lat !== null && state.isMapVisible) {
        state.map.flyTo({ center: [current.lon, current.lat], zoom: 20, speed: 1.5 });
    }

    document.getElementById("btn-prev").disabled = (state.currentIndex === 0);
    document.getElementById("btn-next").disabled = (state.currentIndex === state.appResults.length - 1);
}

export function openFullscreen(type) {
    if (state.appResults.length === 0) return;
    const current = state.appResults[state.currentIndex];
    const viewData = current.views[state.currentDirection] || current.views['front'];
    
    const imgFullscreen = document.getElementById("img-fullscreen");
    if (type === 'rect') {
        imgFullscreen.src = viewData.rect_url.split('?')[0] + `?t=${Date.now()}`;
    } else if (type === 'bev') {
        imgFullscreen.src = viewData.bev_url.split('?')[0] + `?t=${Date.now()}`;
    }
    document.getElementById("fullscreen-modal").showModal();
}

export function initFullscreenModal() {
    document.getElementById("btn-close-fullscreen").onclick = () => {
        document.getElementById("fullscreen-modal").close();
        document.getElementById("img-fullscreen").src = "";
    };
}

export function addWarning(messageHTML) {
    const list = document.getElementById("warnings-list");
    const badge = document.getElementById("warnings-badge");
    const btn = document.getElementById("btn-show-warnings");
    document.getElementById("no-warnings-msg").classList.add("hidden");
    
    const li = document.createElement("li");
    li.innerHTML = messageHTML;
    list.appendChild(li);
    
    state.warningCount++;
    badge.textContent = state.warningCount;
    btn.classList.remove("hidden");
}

export function toggleWarningsModal(show) {
    const modal = document.getElementById("warnings-modal");
    if (show) modal.showModal();
    else modal.close();
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

    const close = () => modal.close();
    
    btnOpen.onclick = () => {
        if(state.appResults.length === 0) return;
        const current = state.appResults[state.currentIndex];
        const baseCalib = current.views[state.currentDirection].calibration || {
            pitch_offset: 0, roll_offset: 0, yaw_offset: 0,
            fov: 100, cam_height: 1.6, z_near: 1.2, z_far: 8.0, lane_width: 6.0
        };
        setCalibrationValues(baseCalib);
        previewImg.src = current.views[state.currentDirection].rect_url;
        modal.showModal();
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
        btnAutoVP.innerHTML = "COMPUTING...";
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
            btnAutoVP.innerHTML = `<svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> AUTO-VP`;
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
        btnApply.disabled = true; btnApply.innerHTML = "INFERRING AI...";
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
            btnApply.disabled = false; btnApply.innerHTML = "COMMIT TO PROJECT";
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
    const btnToggleMap = document.getElementById("btn-toggle-map");

    if (state.isMapVisible) {
        mapPanel.classList.remove("hidden");
        mainSplitter.classList.remove("hidden");
        mediaLayout.setAttribute("data-layout", "vertical");
        btnToggleMap.classList.remove("btn-primary");

        if (!state.layoutPrefs.mapOn.isManual) {
            autoFitSplitters(true);
        } else {
            imagePanel.style.width = state.layoutPrefs.mapOn.mainW;
            document.getElementById("perspective-container").style.flexBasis = state.layoutPrefs.mapOn.mediaBasis;
        }
    } else {
        mapPanel.classList.add("hidden");
        mainSplitter.classList.add("hidden");
        imagePanel.style.width = "100%";
        mediaLayout.setAttribute("data-layout", "horizontal");
        btnToggleMap.classList.add("btn-primary");

        if (!state.layoutPrefs.mapOff.isManual) {
            autoFitSplitters(false);
        } else {
            document.getElementById("perspective-container").style.flexBasis = state.layoutPrefs.mapOff.mediaBasis;
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
    
    const bevSection = document.getElementById("bev-section");
    const bevLayoutContainer = document.getElementById("bev-layout-container");
    if (bevSection && bevLayoutContainer) {
        const ro = new ResizeObserver(entries => {
            for (let entry of entries) {
                const rect = entry.contentRect;
                if (rect.width >= rect.height * 0.9) {
                    bevLayoutContainer.classList.remove("flex-col");
                    bevLayoutContainer.classList.add("flex-row");
                } else {
                    bevLayoutContainer.classList.remove("flex-row");
                    bevLayoutContainer.classList.add("flex-col");
                }
            }
        });
        ro.observe(bevSection);
    }
}

export function initDrawMode() {
    window.drawMode = false;
    window.drawAction = null;
    window.drawIndex = -1;
    window.drawPoints = [];

    const overlay = document.getElementById("draw-overlay");
    const imgDraw = document.getElementById("img-draw-preview");
    
    overlay.addEventListener("click", (e) => {
        if (!window.drawMode) return;
        
        const drawBox = imgDraw.getBoundingClientRect();
        const nw = imgDraw.naturalWidth;
        const nh = imgDraw.naturalHeight;
        if(!nw || !nh) return;
        
        const scale = Math.min(drawBox.width / nw, drawBox.height / nh);
        const wRendered = nw * scale;
        const hRendered = nh * scale;
        
        const xOffset = (drawBox.width - wRendered) / 2;
        const yOffset = (drawBox.height - hRendered) / 2;
        
        const clickX = e.clientX - drawBox.left - xOffset;
        const clickY = e.clientY - drawBox.top - yOffset;
        
        if (clickX < 0 || clickX > wRendered || clickY < 0 || clickY > hRendered) return;
        
        const px = clickX / wRendered;
        const py = clickY / hRendered;
        
        window.drawPoints.push([px, py]);
        renderDrawPoints();
    });

    window.deleteDefect = async (idx) => {
        await modifyDefects("delete", idx);
    };

    let changeClassIndex = -1;
    window.startEditDefect = (idx) => {
        changeClassIndex = idx;
        document.getElementById("change-class-modal").showModal();
    };

    document.getElementById("btn-change-class-cancel").onclick = () => {
        document.getElementById("change-class-modal").close();
    };

    document.getElementById("btn-change-class-save").onclick = async () => {
        const newClass = document.getElementById("change-class-select").value;
        document.getElementById("change-class-modal").close();
        const btn = document.getElementById("btn-change-class-save");
        btn.disabled = true;
        await modifyDefects("update", changeClassIndex, null, newClass);
        btn.disabled = false;
    };

    window.startReoutlineDefect = (idx) => {
        const current = state.appResults[state.currentIndex];
        window.drawMode = true;
        window.drawAction = "re-outline";
        window.drawIndex = idx;
        window.drawPoints = [];
        
        const editSrc = current.views[state.currentDirection].edit_bev_url || current.views[state.currentDirection].raw_bev_url;
        imgDraw.src = editSrc.split('?')[0] + '?t=' + Date.now();
        document.getElementById("draw-modal").showModal();
        document.getElementById("draw-class-container").classList.add("hidden");
    };

    document.getElementById("btn-add-defect").onclick = () => {
        const current = state.appResults[state.currentIndex];
        window.drawMode = true;
        window.drawAction = "add";
        window.drawPoints = [];
        
        const editSrc = current.views[state.currentDirection].edit_bev_url || current.views[state.currentDirection].raw_bev_url;
        imgDraw.src = editSrc.split('?')[0] + '?t=' + Date.now();
        document.getElementById("draw-modal").showModal();
        document.getElementById("draw-class-container").classList.remove("hidden");
    };

    document.getElementById("btn-draw-cancel").onclick = () => {
        window.drawMode = false;
        window.drawPoints = [];
        document.getElementById("draw-overlay").innerHTML = "";
        document.getElementById("draw-modal").close();
    };

    document.getElementById("btn-draw-undo").onclick = () => {
        if (window.drawPoints.length > 0) {
            window.drawPoints.pop();
            renderDrawPoints();
        }
    };

    document.getElementById("btn-draw-clear").onclick = () => {
        window.drawPoints = [];
        renderDrawPoints();
    };

    document.addEventListener("keydown", (e) => {
        if (window.drawMode && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (window.drawPoints.length > 0) {
                window.drawPoints.pop();
                renderDrawPoints();
            }
        }
    });

    const btnSam2 = document.getElementById("btn-draw-sam2");
    if(btnSam2) {
        btnSam2.onclick = async () => {
            if(window.drawPoints.length < 3) {
                alert("MINIMUM 3 POINTS REQUIRED FOR SAM2 PROMPT.");
                return;
            }
            
            const current = state.appResults[state.currentIndex];
            const view = state.currentDirection;
            const calib = current.views[view].calibration || {};
            
            const btnDone = document.getElementById("btn-draw-done");
            
            btnSam2.disabled = true;
            btnDone.disabled = true;
            btnSam2.innerHTML = "COMPUTING...";
            
            try {
                const newPoints = await fetchSam2Preview(current.filename, view, calib, window.drawPoints);
                window.drawPoints = newPoints;
                renderDrawPoints();
            } catch(err) {
                alert("SAM2 Error: " + err.message);
            } finally {
                btnSam2.disabled = false;
                btnDone.disabled = false;
                btnSam2.innerHTML = "APPLY SAM2";
            }
        };
    }

    document.getElementById("btn-draw-done").onclick = async () => {
        if(window.drawPoints.length < 3) {
            alert("MINIMUM 3 POINTS REQUIRED.");
            return;
        }
        
        const className = document.getElementById("draw-class-select").value;
        const btn = document.getElementById("btn-draw-done");
        
        btn.disabled = true; 
        if(btnSam2) btnSam2.disabled = true;
        btn.innerHTML = "SAVING...";
        
        try {
            await modifyDefects(window.drawAction, window.drawIndex, window.drawPoints, className);
        } finally {
            window.drawMode = false;
            window.drawPoints = [];
            document.getElementById("draw-overlay").innerHTML = "";
            document.getElementById("draw-modal").close();
            btn.disabled = false; 
            if(btnSam2) btnSam2.disabled = false;
            btn.innerHTML = "SAVE MASK";
        }
    };
}

function renderDrawPoints() {
    const overlay = document.getElementById("draw-overlay");
    overlay.innerHTML = "";
    
    const imgDraw = document.getElementById("img-draw-preview");
    const drawBox = imgDraw.getBoundingClientRect();
    
    const nw = imgDraw.naturalWidth;
    const nh = imgDraw.naturalHeight;
    const scale = Math.min(drawBox.width / nw, drawBox.height / nh);
    const wRendered = nw * scale;
    const hRendered = nh * scale;
    
    const xOffset = (drawBox.width - wRendered) / 2;
    const yOffset = (drawBox.height - hRendered) / 2;

    window.drawPoints.forEach((pt, i) => {
        const x = pt[0] * wRendered + xOffset;
        const y = pt[1] * hRendered + yOffset;
        
        const dot = document.createElement("div");
        dot.style.position = "absolute";
        dot.style.width = "4px";
        dot.style.height = "4px";
        dot.style.backgroundColor = "red";
        dot.style.transform = "translate(-2px, -2px)";
        dot.style.pointerEvents = "none";
        dot.style.left = x + "px";
        dot.style.top = y + "px";
        overlay.appendChild(dot);
        
        if (i > 0) {
            const prevPt = window.drawPoints[i-1];
            const px = prevPt[0] * wRendered + xOffset;
            const py = prevPt[1] * hRendered + yOffset;
            const dist = Math.hypot(x - px, y - py);
            const angle = Math.atan2(y - py, x - px);
            
            const line = document.createElement("div");
            line.style.position = "absolute";
            line.style.backgroundColor = "red";
            line.style.transformOrigin = "left";
            line.style.pointerEvents = "none";
            line.style.height = "1px";
            line.style.width = dist + "px";
            line.style.left = px + "px";
            line.style.top = py + "px";
            line.style.transform = `rotate(${angle}rad)`;
            overlay.appendChild(line);
        }
    });
    
    if (window.drawPoints.length > 2) {
        const lastPt = window.drawPoints[window.drawPoints.length - 1];
        const firstPt = window.drawPoints[0];
        
        const lx = lastPt[0] * wRendered + xOffset;
        const ly = lastPt[1] * hRendered + yOffset;
        const fx = firstPt[0] * wRendered + xOffset;
        const fy = firstPt[1] * hRendered + yOffset;
        
        const dist = Math.hypot(fx - lx, fy - ly);
        const angle = Math.atan2(fy - ly, fx - lx);
        
        const line = document.createElement("div");
        line.style.position = "absolute";
        line.style.backgroundColor = "rgba(255,0,0,0.5)";
        line.style.transformOrigin = "left";
        line.style.pointerEvents = "none";
        line.style.height = "1px";
        line.style.width = dist + "px";
        line.style.left = lx + "px";
        line.style.top = ly + "px";
        line.style.transform = `rotate(${angle}rad)`;
        overlay.appendChild(line);
    }
}

async function modifyDefects(action, index, points=null, className=null) {
    if (state.appResults.length === 0) return;
    const current = state.appResults[state.currentIndex];
    const view = state.currentDirection;
    const calib = current.views[view].calibration || {};
    
    try {
        const res = await fetch("/modify_defects", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                filename: current.filename,
                view: view,
                action: action,
                index: index,
                points: points,
                class_name: className,
                calibration: calib
                // Note: use_sam2 is not sent so backend defaults to False (saves points exactly as requested)
            })
        });
        const data = await res.json();
        if(data.success) {
            current.views[view].defects = data.defects;
            
            current.geojson = current.geojson.filter(f => f.properties.view !== view);
            current.geojson.push(...data.geojson);
            
            state.fullGeojson.features = state.fullGeojson.features.filter(f => !(f.properties.filename === current.original_name && f.properties.view === view));
            state.fullGeojson.features.push(...data.geojson);
            
            updateMapSource('defects-source', state.fullGeojson);
            updateCarousel(false);
        } else {
            alert(data.error || "Modification failed");
        }
    } catch(e) {
        console.error(e);
        alert("Failed to modify defects");
    }
}