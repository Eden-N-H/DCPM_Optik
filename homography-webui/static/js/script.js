document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let isModelLoaded = false;
    let imageFiles = [];
    let appIs360 = true;
    
    let map = null;
    let geoJsonLayer = null;
    let pathLayer = null;
    let mapMarkers = {}; 
    let currentMapOverlay = null; // Backend stitched map layer
    
    let fullResults = [];
    let fullGeojson = { type: "FeatureCollection", features: [] };
    let appResults = []; 
    let currentIndex = 0;
    let currentDirection = 'front'; 
    let activeMarkerFilename = null; 

    let stateLastLat = 0.0;
    let stateLastLon = 0.0;
    let stateLastLocId = 1;

    const btnProcess = document.getElementById("process-btn");
    const btnScan = document.getElementById("scan-btn");
    const selLocation = document.getElementById("sel-location");
    const uploadPanel = document.getElementById("upload-panel");
    const chkIs360 = document.getElementById("chk-is-360");
    const containerBevRear = document.getElementById("container-bev-rear");
    const btnLoadPhoto = document.getElementById("btn-load-photo");

    function stringToColor(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        let color = '#';
        for (let i = 0; i < 3; i++) {
            let value = (hash >> (i * 8)) & 0xFF;
            color += ('00' + value.toString(16)).substr(-2);
        }
        return color;
    }

    function initMap() {
        if (!map) {
            map = L.map('map').setView([-32.06, 151.90], 15);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 24, maxNativeZoom: 17 
            }).addTo(map);

            map.createPane('photoPane'); map.getPane('photoPane').style.zIndex = 250;
            map.createPane('trailPane'); map.getPane('trailPane').style.zIndex = 300;
            map.createPane('nodePane'); map.getPane('nodePane').style.zIndex = 450; 

            geoJsonLayer = L.geoJSON(null, {
                style: (feature) => {
                    const fColor = feature.properties.color || stringToColor(feature.properties.class || "Unknown");
                    const fOpacity = feature.properties.conf !== undefined ? feature.properties.conf : 0.4;
                    return { color: "#ffffff", weight: 2, opacity: 1, fillColor: fColor, fillOpacity: fOpacity };
                },
                onEachFeature: (feature, layer) => {
                    if (feature.geometry && feature.geometry.type === 'Polygon') {
                        layer.bindPopup(`<b>${feature.properties.class}</b><br>View: ${feature.properties.view}<br>Area: ${feature.properties.area_sqm} m²`);
                        
                        layer.on('click', () => {
                            const fname = feature.properties.filename;
                            if (fname) {
                                const target = fullResults.find(r => r.original_name === fname);
                                if (target) {
                                    if (selLocation.value !== target.location) {
                                        selLocation.value = target.location;
                                        appResults = fullResults.filter(r => r.location === target.location);
                                    }
                                    currentIndex = appResults.findIndex(r => r.original_name === target.original_name);
                                    updateCarousel(false);
                                }
                            }
                        });
                    }
                }
            }).addTo(map);

            pathLayer = L.geoJSON(null, { pane: 'trailPane', style: { color: "#94a3b8", weight: 2, dashArray: "4, 6" } }).addTo(map);
        }
    }

    document.getElementById("btn-toggle-upload").onclick = () => uploadPanel.classList.toggle("hidden");

    const setupDz = (dzId, inId, nameId, isMulti, callback) => {
        const dz = document.getElementById(dzId);
        const inp = document.getElementById(inId);
        const nm = document.getElementById(nameId);

        dz.onclick = () => inp.click();
        dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("border-blue-500"); };
        dz.ondragleave = () => dz.classList.remove("border-blue-500");
        dz.ondrop = (e) => {
            e.preventDefault(); dz.classList.remove("border-blue-500");
            handleFiles(e.dataTransfer.files, isMulti, callback, nm);
        };
        inp.onchange = (e) => handleFiles(e.target.files, isMulti, callback, nm);
    };

    const handleFiles = (files, isMulti, callback, nameElement) => {
        if (!files.length) return;
        if (isMulti) {
            callback(Array.from(files));
            nameElement.textContent = `${files.length} items queued`;
        } else {
            callback(files[0]);
            nameElement.textContent = files[0].name;
            document.getElementById("status-model").classList.remove("hidden");
            isModelLoaded = true;
        }
        nameElement.classList.remove("hidden");
        checkCanProcess();
    };

    const checkCanProcess = () => {
        const hasModel = (isModelLoaded || modelFile !== null);
        btnProcess.disabled = !(hasModel && imageFiles.length > 0);
        btnScan.disabled = !hasModel;
    };

    setupDz("dz-model", "in-model", "name-model", false, f => modelFile = f);
    setupDz("dz-image", "in-image", "name-image", true, f => imageFiles = f);

    async function executeJob(endpoint, useUploadedFiles) {
        const fd = new FormData();
        if (modelFile) fd.append("model", modelFile);
        
        fd.append("cam_height", document.getElementById("cam-height").value);
        fd.append("is_360", chkIs360.checked ? "true" : "false");
        fd.append("gps_snap", document.getElementById("chk-gps-snap").checked ? "true" : "false");
        fd.append("frame_skip", document.getElementById("frame-skip").value);
        
        fd.append("last_lat", stateLastLat);
        fd.append("last_lon", stateLastLon);
        fd.append("last_loc_id", stateLastLocId);
        
        if (useUploadedFiles) {
            imageFiles.forEach(f => fd.append("images", f));
        }

        appIs360 = chkIs360.checked;
        if (!appIs360) {
            containerBevRear.classList.add("hidden");
            setView('front');
        } else {
            containerBevRear.classList.remove("hidden");
        }

        uploadPanel.classList.add("hidden");
        document.getElementById("workspace").classList.remove("hidden");
        document.getElementById("btn-save-project").classList.remove("hidden");
        document.getElementById("btn-export-zip").classList.remove("hidden");
        document.getElementById("btn-export-flat-zip").classList.remove("hidden");
        document.getElementById("progress-container").classList.remove("hidden");
        
        btnProcess.disabled = true;
        btnScan.disabled = true;
        initMap();

        try {
            const res = await fetch(endpoint, { method: "POST", body: fd });
            const data = await res.json();
            
            if (!res.ok || data.error) throw new Error(data.error || "Unknown server error");

            stateLastLat = data.last_lat;
            stateLastLon = data.last_lon;
            stateLastLocId = data.last_loc_id;

            if (!appIs360 && data.has_video) {
                btnLoadPhoto.classList.remove("hidden");
            } else {
                btnLoadPhoto.classList.add("hidden");
            }

            setTimeout(() => {
                map.invalidateSize();
                if (data.initial_trail && data.initial_trail.features) {
                    pathLayer.addData(data.initial_trail);
                    if(pathLayer.getBounds().isValid()) map.fitBounds(pathLayer.getBounds(), { padding: [50, 50] });
                }

                data.initial_state.forEach(img => {
                    if(img.lat !== 0.0) {
                        const marker = L.circleMarker([img.lat, img.lon], {
                            radius: 2, fillColor: "#ef4444", color: "#ffffff", weight: 0.5, opacity: 1, fillOpacity: 1, pane: 'nodePane'
                        }).addTo(map);
                        marker.bindTooltip(`<b>Pending Process</b><br>${img.original_name}`, { direction: 'top', className: 'text-xs border-0 shadow-sm bg-white/90' });
                        mapMarkers[img.original_name] = marker;
                    }
                });

                startSSE(data.task_id, data.total_images);
            }, 200);

        } catch (e) {
            alert(e.message);
            uploadPanel.classList.remove("hidden");
            document.getElementById("progress-container").classList.add("hidden");
            checkCanProcess();
        }
    }

    btnProcess.onclick = () => executeJob("/process", true);
    btnScan.onclick = () => executeJob("/process_pipeline_folder", false);

    function startSSE(taskId, totalImages) {
        const source = new EventSource(`/stream/${taskId}`);
        let processedCount = 0;
        let startTime = Date.now();

        source.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === "error" || msg.type === "complete") {
                source.close();
                document.getElementById("progress-container").classList.add("hidden");
                checkCanProcess();
                
                imageFiles = [];
                document.getElementById("name-image").textContent = "Completed.";
                if (msg.type === "error") alert(`Background Task Error: ${msg.message}`);
                return;
            }

            if (msg.type === "update") {
                const r = msg.data;
                fullResults.push(r);
                
                let hasDefects = r.geojson && r.geojson.length > 0;
                if (hasDefects) {
                    fullGeojson.features.push(...r.geojson);
                    geoJsonLayer.addData(r.geojson);
                }

                if (mapMarkers[r.original_name]) {
                    const marker = mapMarkers[r.original_name];
                    
                    if (activeMarkerFilename !== r.original_name) {
                        marker.setStyle({ fillColor: "#3b82f6", radius: 2, color: "#ffffff", weight: 0.5 });
                    }
                    marker.bindTooltip(`<b>Photo Location</b><br>${r.original_name}`, { direction: 'top', className: 'text-xs border-0 shadow-sm bg-white/90' });
                    
                    marker.off('click'); 
                    marker.on('click', () => {
                        if (selLocation.value !== r.location) {
                            selLocation.value = r.location;
                            appResults = fullResults.filter(x => x.location === r.location);
                        }
                        currentIndex = appResults.findIndex(x => x.original_name === r.original_name);
                        updateCarousel(false);
                    });
                }

                processedCount++;
                const pct = totalImages > 0 ? (processedCount / totalImages) * 100 : 100;
                document.getElementById("progress-bar").style.width = `${pct}%`;
                document.getElementById("progress-text").textContent = `Segmenting ${processedCount} of ${totalImages}`;

                const elapsedSec = (Date.now() - startTime) / 1000;
                const avgSpeed = elapsedSec / processedCount;
                const remainSec = Math.ceil((totalImages - processedCount) * avgSpeed);
                const mins = Math.floor(remainSec / 60);
                const secs = remainSec % 60;
                document.getElementById("eta-text").textContent = `ETA: ${mins}m ${secs}s`;

                refreshLocationsUI();

                if (selLocation.value === r.location) {
                    appResults = fullResults.filter(x => x.location === r.location);
                    if (appResults.length > 0) {
                        document.getElementById("carousel-counter").textContent = `Item ${currentIndex + 1} of ${appResults.length}`;
                        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
                    }
                }

                if (fullResults.length === 1) updateCarousel(true);
            }
        };
    }

    function refreshLocationsUI() {
        const locations = [...new Set(fullResults.map(r => r.location))];
        const currentSelection = selLocation.value;
        selLocation.innerHTML = locations.map(loc => `<option value="${loc}">${loc}</option>`).join("");
        
        if (locations.includes(currentSelection)) selLocation.value = currentSelection;
        else if (locations.length > 0) selLocation.value = locations[0];

        selLocation.onchange = () => {
            appResults = fullResults.filter(r => r.location === selLocation.value);
            currentIndex = 0;
            updateCarousel(true);
        };

        appResults = fullResults.filter(r => r.location === selLocation.value);
        if(appResults.length > 0 && document.getElementById("img-rect").classList.contains("hidden")){
            updateCarousel(false);
        }
    }

    document.getElementById("btn-save-project").onclick = () => {
        if (fullResults.length === 0) return;
        const projectData = { is_360: appIs360, results: fullResults, geojson: fullGeojson };
        const blob = new Blob([JSON.stringify(projectData)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "dcpm_project.json";
        a.click();
        URL.revokeObjectURL(url);
    };

    document.getElementById("in-load-project").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const data = JSON.parse(ev.target.result);
                if (!data.results || !data.geojson) throw new Error("Invalid project format.");
                
                appIs360 = data.is_360 !== undefined ? data.is_360 : true;
                if (!appIs360) {
                    containerBevRear.classList.add("hidden");
                    setView('front');
                } else {
                    containerBevRear.classList.remove("hidden");
                }

                const hasVideo = data.results.some(r => r.original_name && (r.original_name.toLowerCase().includes('.mp4') || r.original_name.toLowerCase().includes('frame')));
                if (!appIs360 && hasVideo) {
                    btnLoadPhoto.classList.remove("hidden");
                } else {
                    btnLoadPhoto.classList.add("hidden");
                }

                fullResults = data.results;
                fullGeojson = data.geojson;
                
                if (fullResults.length > 0) {
                    const lastRec = fullResults[fullResults.length - 1];
                    stateLastLat = lastRec.lat;
                    stateLastLon = lastRec.lon;
                    stateLastLocId = parseInt(lastRec.location.replace("Location ", "")) || 1;
                }

                uploadPanel.classList.add("hidden");
                document.getElementById("workspace").classList.remove("hidden");
                document.getElementById("btn-save-project").classList.remove("hidden");
                document.getElementById("btn-export-zip").classList.remove("hidden");
                document.getElementById("btn-export-flat-zip").classList.remove("hidden");
                
                initMap();

                setTimeout(() => {
                    map.invalidateSize();
                    geoJsonLayer.addData(fullGeojson);
                    
                    let tCoords = fullResults.filter(r => r.lat !== 0.0).map(r => [r.lon, r.lat]);
                    if (tCoords.length > 1) {
                        pathLayer.addData({ type: "Feature", geometry: { type: "LineString", coordinates: tCoords } });
                    }

                    fullResults.forEach(img => {
                        if (!mapMarkers[img.original_name] && img.lat !== 0.0) {
                            const marker = L.circleMarker([img.lat, img.lon], {
                                radius: 2, fillColor: "#3b82f6", color: "#ffffff", weight: 0.5, opacity: 1, fillOpacity: 1, pane: 'nodePane'
                            }).addTo(map);
                            
                            marker.bindTooltip(`<b>Photo Location</b><br>${img.original_name}`, { direction: 'top', className: 'text-xs border-0 shadow-sm bg-white/90' });
                            mapMarkers[img.original_name] = marker;
                            
                            marker.on('click', () => {
                                if (selLocation.value !== img.location) {
                                    selLocation.value = img.location;
                                    appResults = fullResults.filter(r => r.location === img.location);
                                }
                                currentIndex = appResults.findIndex(r => r.original_name === img.original_name);
                                updateCarousel(false);
                            });
                        }
                    });
                    
                    if (geoJsonLayer.getBounds().isValid()) {
                        map.fitBounds(geoJsonLayer.getBounds(), { padding: [50, 50] });
                    }

                    refreshLocationsUI();
                    setView('front');
                    if (appResults.length > 0) updateCarousel(true);
                }, 200);

            } catch (err) {
                alert("Error loading project: " + err.message);
            }
        };
        reader.readAsText(file);
    });

    btnLoadPhoto.onclick = async function() {
        const hasFootprint = appResults.some(r => r.views && r.views['front'] && r.views['front'].footprint);
        if (!hasFootprint) {
            alert("This project is missing BEV footprint data.");
            return;
        }

        const originalText = this.innerHTML;
        this.innerHTML = "⏳ Initializing Photogrammetry...";
        this.disabled = true;
        
        document.getElementById("progress-bar").style.width = `0%`;
        document.getElementById("progress-text").textContent = `Warming Up SIFT Extractors...`;
        document.getElementById("eta-text").textContent = `ETA: Calculating...`;
        document.getElementById("progress-container").classList.remove("hidden");
        
        try {
            const res = await fetch("/generate-map", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    location: selLocation.value, 
                    results: appResults, 
                    view: currentDirection 
                })
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            startMapSSE(data.task_id, originalText);
            
        } catch (err) {
            alert("Map Generation Failed: " + err.message);
            resetMapUI(originalText);
        } 
    };

    function startMapSSE(taskId, originalText) {
        const source = new EventSource(`/stream/${taskId}`);
        let startTime = Date.now();

        source.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === "error") {
                source.close();
                alert(`Map Stitching Error: ${msg.message}`);
                resetMapUI(originalText);
                return;
            }

            if (msg.type === "map_progress") {
                const pct = msg.total > 0 ? (msg.current / msg.total) * 100 : 100;
                document.getElementById("progress-bar").style.width = `${pct}%`;
                document.getElementById("progress-text").textContent = msg.status_msg;

                if (msg.current > 0) {
                    const elapsedSec = (Date.now() - startTime) / 1000;
                    const avgSpeed = elapsedSec / msg.current;
                    const remainSec = Math.ceil((msg.total - msg.current) * avgSpeed);
                    const mins = Math.floor(remainSec / 60);
                    const secs = remainSec % 60;
                    document.getElementById("eta-text").textContent = `ETA: ${mins}m ${secs}s`;
                }
            }

            if (msg.type === "map_complete") {
                source.close();
                
                if (currentMapOverlay && map.hasLayer(currentMapOverlay)) {
                    map.removeLayer(currentMapOverlay);
                }
                
                currentMapOverlay = L.imageOverlay(msg.overlay_url, msg.bounds, { opacity: 0.9, pane: 'photoPane' }).addTo(map);
                map.fitBounds(msg.bounds);
                setupPhotoToggleUI(msg.overlay_url, msg.pure_url);
                
                resetMapUI(originalText);
            }
        };
    }

    function resetMapUI(originalText) {
        document.getElementById("progress-container").classList.add("hidden");
        btnLoadPhoto.innerHTML = originalText;
        btnLoadPhoto.disabled = false;
    }

    function setupPhotoToggleUI(overlayUrl, pureUrl) {
        if (map.photoToggleControl) map.removeControl(map.photoToggleControl);
        
        L.Control.PhotoToggle = L.Control.extend({
            options: { position: 'bottomleft' },
            onAdd: function() {
                var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom bg-white/95 px-3 py-2 shadow-sm text-sm flex items-center gap-3');
                
                var toggleBtn = L.DomUtil.create('div', 'font-bold cursor-pointer', container);
                toggleBtn.innerHTML = '🗺️ Map Layer: ON';
                
                var dlOverlayBtn = L.DomUtil.create('a', 'cursor-pointer text-blue-600 hover:text-blue-800 font-bold border-l border-gray-300 pl-3', container);
                dlOverlayBtn.innerHTML = '⬇️ DL Overlay';
                dlOverlayBtn.href = overlayUrl;
                dlOverlayBtn.download = "DCPM_Map_Overlay.png";
                
                var dlPureBtn = L.DomUtil.create('a', 'cursor-pointer text-emerald-600 hover:text-emerald-800 font-bold border-l border-gray-300 pl-3', container);
                dlPureBtn.innerHTML = '🖼️ DL Pure Ribbon';
                dlPureBtn.href = pureUrl;
                dlPureBtn.download = "DCPM_Pure_Ribbon.png";

                toggleBtn.onclick = function(e) {
                    e.stopPropagation();
                    if (currentMapOverlay && map.hasLayer(currentMapOverlay)) {
                        map.removeLayer(currentMapOverlay);
                        toggleBtn.innerHTML = '🗺️ Map Layer: OFF';
                        toggleBtn.style.color = '#6b7280';
                    } else if (currentMapOverlay) {
                        map.addLayer(currentMapOverlay);
                        toggleBtn.innerHTML = '🗺️ Map Layer: ON';
                        toggleBtn.style.color = '#000000';
                    }
                }
                return container;
            }
        });
        map.photoToggleControl = new L.Control.PhotoToggle();
        map.addControl(map.photoToggleControl);
    }

    const triggerZipExport = async (endpoint, btnId, loadingText, filename) => {
        if (fullResults.length === 0) return;
        const btn = document.getElementById(btnId);
        const originalText = btn.textContent;
        btn.textContent = loadingText;
        btn.disabled = true;

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ results: fullResults })
            });
            if (!res.ok) throw new Error("Failed to compile ZIP file");
            
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            alert(err.message);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    };

    document.getElementById("btn-export-zip").onclick = () => triggerZipExport("/export-zip", "btn-export-zip", "⏳ Compiling ZIP...", "DCPM_RAW_Export.zip");
    document.getElementById("btn-export-flat-zip").onclick = () => triggerZipExport("/export-flat-zip", "btn-export-flat-zip", "⏳ Compiling Flattened ZIP...", "DCPM_FLAT_Export.zip");

    function setView(dir) {
        currentDirection = dir;
        const contF = document.getElementById('container-bev-front');
        const contR = document.getElementById('container-bev-rear');
        const activeLabel = document.getElementById('label-active-view');
        
        if (dir === 'front') {
            contF.classList.add('border-blue-500', 'ring-2');
            contF.classList.remove('border-transparent');
            contR.classList.remove('border-blue-500', 'ring-2');
            contR.classList.add('border-transparent');
            activeLabel.textContent = 'Front View Active';
        } else {
            contR.classList.add('border-blue-500', 'ring-2');
            contR.classList.remove('border-transparent');
            contF.classList.remove('border-blue-500', 'ring-2');
            contF.classList.add('border-transparent');
            activeLabel.textContent = 'Rear View Active';
        }
        updateCarousel(false);
    }

    document.getElementById('container-bev-front').onclick = () => setView('front');
    document.getElementById('container-bev-rear').onclick = () => setView('rear');

    function updateCarousel(panMap = true) {
        if (appResults.length === 0) return;
        const current = appResults[currentIndex];
        
        document.getElementById("placeholder-rect").classList.add("hidden");
        const imgRect = document.getElementById("img-rect");
        const imgBevF = document.getElementById("img-bev-front");
        const imgBevR = document.getElementById("img-bev-rear");
        
        imgRect.classList.remove("hidden");
        imgBevF.classList.remove("hidden");
        if (appIs360) imgBevR.classList.remove("hidden");

        const activeViewData = current.views[currentDirection] || current.views['front'];

        document.getElementById("carousel-counter").textContent = `Item ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name;
        document.getElementById("carousel-telemetry").textContent = `Auto-Pitch: ${current.pitch}°`;

        imgBevF.src = current.views['front'].bev_url;
        if (appIs360 && current.views['rear']) imgBevR.src = current.views['rear'].bev_url;
        imgRect.src = activeViewData.rect_url;

        const tbody = document.getElementById("table-defects");
        tbody.innerHTML = activeViewData.defects.map(d => {
            const classColor = d.color || stringToColor(d.class);
            return `<tr>
                <td class="p-2"><span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: ${classColor}; border: 1px solid #ccc;"></span>${d.class}</td>
                <td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td>
                <td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td>
            </tr>`;
        }).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No detections</td></tr>`;

        if (activeMarkerFilename && mapMarkers[activeMarkerFilename]) {
            mapMarkers[activeMarkerFilename].setStyle({ fillColor: "#3b82f6", radius: 2, weight: 0.5, color: "#ffffff" });
        }
        
        activeMarkerFilename = current.original_name;
        const newActiveMarker = mapMarkers[activeMarkerFilename];
        if (newActiveMarker) {
            newActiveMarker.setStyle({ fillColor: "#fde047", radius: 6, weight: 2, color: "#000000" });
            newActiveMarker.bringToFront();
            if (panMap) map.setView(newActiveMarker.getLatLng(), 20, { animate: true });
        }

        document.getElementById("btn-prev").disabled = (currentIndex === 0);
        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
    }

    document.getElementById("btn-prev").onclick = () => { if (currentIndex > 0) { currentIndex--; updateCarousel(true); } };
    document.getElementById("btn-next").onclick = () => { if (currentIndex < appResults.length - 1) { currentIndex++; updateCarousel(true); } };
});