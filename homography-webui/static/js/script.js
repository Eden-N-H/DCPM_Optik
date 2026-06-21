document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let isModelLoaded = false;
    let imageFiles = [];
    let appIs360 = true;
    
    let map = null;
    let mapLoaded = false;
    let mapPopup = null;
    
    let fullGeojson = { type: "FeatureCollection", features: [] };
    let nodesGeoJson = { type: "FeatureCollection", features: [] };
    let trailGeoJson = { type: "FeatureCollection", features: [] };
    
    let fullResults = [];
    let appResults = []; 
    let currentIndex = 0;
    let currentDirection = 'front'; 
    let activeMarkerFilename = null; 

    let stateLastLat = null;
    let stateLastLon = null;
    let stateLastLocId = 1;

    const btnProcess = document.getElementById("process-btn");
    const btnScan = document.getElementById("scan-btn");
    const selLocation = document.getElementById("sel-location");
    const uploadPanel = document.getElementById("upload-panel");
    const chkIs360 = document.getElementById("chk-is-360");
    const chkDrawGrid = document.getElementById("chk-draw-grid");
    const containerBevRear = document.getElementById("container-bev-rear");
    
    const warningsContainer = document.getElementById("warnings-container");
    const warningsList = document.getElementById("warnings-list");
    const healthPanel = document.getElementById("health-panel");
    const healthContent = document.getElementById("health-content");
    const healthWarnings = document.getElementById("health-warnings");

    function stringToColor(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
        let color = '#';
        for (let i = 0; i < 3; i++) color += ('00' + ((hash >> (i * 8)) & 0xFF).toString(16)).substr(-2);
        return color;
    }

    function updateMapSource(sourceId, data) {
        if (mapLoaded && map) {
            const src = map.getSource(sourceId);
            if (src) src.setData(data);
        }
    }

    function fitMapToBounds(geoJsonData, maxZoom = 20) {
        if (!mapLoaded || !map || !geoJsonData || !geoJsonData.features || geoJsonData.features.length === 0) return;
        try {
            const bbox = turf.bbox(geoJsonData);
            if (bbox.some(val => isNaN(val) || !isFinite(val))) return;
            map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 50, maxZoom: maxZoom });
        } catch (e) {
            console.warn("Could not calculate bounds", e);
        }
    }

    function initMap() {
        if (!map) {
            map = new maplibregl.Map({
                container: 'map',
                style: {
                    version: 8,
                    sources: {
                        'google-satellite': {
                            type: 'raster',
                            tiles: ['https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'],
                            tileSize: 256,
                            maxzoom: 22
                        }
                    },
                    layers: [{ id: 'google-satellite-layer', type: 'raster', source: 'google-satellite' }]
                },
                center: [151.90, -32.06], 
                zoom: 15
            });

            mapPopup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

            map.on('load', () => {
                mapLoaded = true;

                map.addSource('trail-source', { type: 'geojson', data: trailGeoJson });
                map.addSource('defects-source', { type: 'geojson', data: fullGeojson });
                map.addSource('nodes-source', { type: 'geojson', data: nodesGeoJson });

                map.addLayer({
                    id: 'trail-layer', type: 'line', source: 'trail-source',
                    paint: { 'line-color': '#94a3b8', 'line-width': 2, 'line-dasharray': [2, 3] }
                });

                map.addLayer({
                    id: 'defects-layer', type: 'fill', source: 'defects-source',
                    paint: {
                        'fill-color': ['coalesce', ['get', 'color'], '#ffffff'],
                        'fill-opacity': ['coalesce', ['get', 'conf'], 0.4]
                    }
                });

                map.addLayer({
                    id: 'defects-outline-layer', type: 'line', source: 'defects-source',
                    paint: { 'line-color': '#ffffff', 'line-width': 1.5 }
                });

                map.addLayer({
                    id: 'nodes-layer', type: 'circle', source: 'nodes-source',
                    paint: {
                        'circle-radius': ['case', ['boolean', ['get', 'active'], false], 6, 3],
                        'circle-color': ['case',
                            ['boolean', ['get', 'active'], false], '#fde047',  
                            ['boolean', ['get', 'processed'], false], '#3b82f6', 
                            '#ef4444' 
                        ],
                        'circle-stroke-width': ['case', ['boolean', ['get', 'active'], false], 2, 0.5],
                        'circle-stroke-color': ['case', ['boolean', ['get', 'active'], false], '#000000', '#ffffff']
                    }
                });

                map.on('mouseenter', 'nodes-layer', (e) => {
                    map.getCanvas().style.cursor = 'pointer';
                    const f = e.features[0];
                    const status = f.properties.processed ? "Photo Location" : "Pending Process";
                    mapPopup.setLngLat(e.lngLat).setHTML(`<div class="text-xs"><b>${status}</b><br>${f.properties.original_name}</div>`).addTo(map);
                });
                
                map.on('mouseleave', 'nodes-layer', () => {
                    map.getCanvas().style.cursor = '';
                    mapPopup.remove();
                });
                
                map.on('click', 'nodes-layer', (e) => {
                    const f = e.features[0];
                    if (!f.properties.processed) return;
                    handleMapClick(f.properties.original_name, f.properties.location);
                });

                map.on('mouseenter', 'defects-layer', (e) => {
                    map.getCanvas().style.cursor = 'pointer';
                    const f = e.features[0];
                    mapPopup.setLngLat(e.lngLat).setHTML(`<div class="text-xs"><b>${f.properties.class}</b><br>View: ${f.properties.view}<br>Area: ${f.properties.area_sqm} m²</div>`).addTo(map);
                });
                
                map.on('mouseleave', 'defects-layer', () => {
                    map.getCanvas().style.cursor = '';
                    mapPopup.remove();
                });
                
                map.on('click', 'defects-layer', (e) => {
                    const f = e.features[0];
                    handleMapClick(f.properties.filename, null);
                });
                
                if (fullGeojson.features.length > 0) fitMapToBounds(fullGeojson);
                else if (trailGeoJson.features.length > 0) fitMapToBounds(trailGeoJson);
            });
        }
    }

    function handleMapClick(originalName, locationStr) {
        const target = fullResults.find(r => r.original_name === originalName);
        if (target) {
            if (selLocation.value !== target.location) {
                selLocation.value = target.location;
                appResults = fullResults.filter(r => r.location === target.location);
            }
            currentIndex = appResults.findIndex(r => r.original_name === target.original_name);
            updateCarousel(false);
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
        dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("border-blue-500"); handleFiles(e.dataTransfer.files, isMulti, callback, nm); };
        inp.onchange = (e) => handleFiles(e.target.files, isMulti, callback, nm);
    };

    const handleFiles = (files, isMulti, callback, nameElement) => {
        if (!files.length) return;
        if (isMulti) { callback(Array.from(files)); nameElement.textContent = `${files.length} items queued`; } 
        else { callback(files[0]); nameElement.textContent = files[0].name; document.getElementById("status-model").classList.remove("hidden"); isModelLoaded = true; }
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
        fd.append("draw_grid", chkDrawGrid.checked ? "true" : "false");
        fd.append("interval_m", document.getElementById("interval-m").value);
        
        if(stateLastLat !== null) fd.append("last_lat", stateLastLat);
        if(stateLastLon !== null) fd.append("last_lon", stateLastLon);
        fd.append("last_loc_id", stateLastLocId);
        
        if (useUploadedFiles) imageFiles.forEach(f => fd.append("images", f));

        appIs360 = chkIs360.checked;
        if (!appIs360) { containerBevRear.classList.add("hidden"); setView('front'); } 
        else containerBevRear.classList.remove("hidden");

        uploadPanel.classList.add("hidden");
        document.getElementById("workspace").classList.remove("hidden");
        document.getElementById("btn-save-project").classList.remove("hidden");
        document.getElementById("btn-export-zip").classList.remove("hidden");
        document.getElementById("btn-export-flat-zip").classList.remove("hidden");
        document.getElementById("progress-container").classList.remove("hidden");
        
        warningsList.innerHTML = "";
        warningsContainer.classList.add("hidden");
        healthContent.innerHTML = "";
        healthWarnings.innerHTML = "";
        healthPanel.classList.add("hidden");
        
        btnProcess.disabled = true;
        btnScan.disabled = true;
        
        fullGeojson = { type: "FeatureCollection", features: [] };
        nodesGeoJson = { type: "FeatureCollection", features: [] };
        trailGeoJson = { type: "FeatureCollection", features: [] };
        fullResults = [];
        appResults = [];

        initMap();

        try {
            const res = await fetch(endpoint, { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || "Unknown server error");

            stateLastLat = data.last_lat;
            stateLastLon = data.last_lon;
            stateLastLocId = data.last_loc_id;

            if (data.initial_trail && data.initial_trail.features) {
                trailGeoJson = data.initial_trail;
                updateMapSource('trail-source', trailGeoJson);
                fitMapToBounds(trailGeoJson);
            }

            data.initial_state.forEach(img => {
                if(img.lat !== null && img.lon !== null) {
                    nodesGeoJson.features.push({
                        type: "Feature",
                        geometry: { type: "Point", coordinates: [img.lon, img.lat] },
                        properties: { original_name: img.original_name, location: img.location, processed: false, active: false }
                    });
                }
            });
            updateMapSource('nodes-source', nodesGeoJson);

            setTimeout(() => { if (map) map.resize(); }, 300);

            startSSE(data.task_id, data.total_images);
        } catch (e) {
            alert(e.message); uploadPanel.classList.remove("hidden"); document.getElementById("progress-container").classList.add("hidden"); checkCanProcess();
        }
    }

    btnProcess.onclick = () => executeJob("/process", true);
    btnScan.onclick = () => executeJob("/process_pipeline_folder", false);

    function startSSE(taskId, totalImages) {
        const source = new EventSource(`/stream/${taskId}`);
        let processedCount = 0; let startTime = Date.now();

        source.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "error" || msg.type === "complete") {
                source.close();
                document.getElementById("progress-container").classList.add("hidden");
                checkCanProcess();
                imageFiles = [];
                document.getElementById("name-image").textContent = "Completed.";
                
                if (msg.type === "error") {
                    alert(`Background Task Error: ${msg.message}`);
                } else if (msg.type === "complete" && fullResults.length === 0) {
                    alert("Processing complete, but 0 frames were successfully extracted. Check the warnings panel for metadata rejection details.");
                    
                    document.getElementById("workspace").classList.add("hidden");
                    document.getElementById("upload-panel").classList.remove("hidden");
                    document.getElementById("btn-save-project").classList.add("hidden");
                    document.getElementById("btn-export-zip").classList.add("hidden");
                    document.getElementById("btn-export-flat-zip").classList.add("hidden");
                }
                return;
            }

            if (msg.type === "health_report") {
                healthPanel.classList.remove("hidden");
                const hr = msg.data;
                
                let gpsColor = hr.gps_score > 80 ? 'text-green-600' : 'text-orange-500';
                let imuColor = hr.imu_score > 90 ? 'text-green-600' : 'text-orange-500';

                // Displaying metric cards
                healthContent.innerHTML += `
                    <div class="p-3 bg-gray-50 rounded border shadow-sm flex flex-col justify-center">
                        <p class="text-[10px] text-gray-500 uppercase font-bold tracking-wider">${msg.original_name} - GPS Confidence</p>
                        <p class="text-2xl font-bold ${gpsColor}">${hr.gps_score.toFixed(1)}%</p>
                        <p class="text-xs text-gray-600 mt-1">Spatial Drift: ${hr.metrics.avg_gps_speed_error_ms.toFixed(2)} m/s</p>
                        <p class="text-xs text-gray-600">Poor Sat Fix Ratio: ${(hr.metrics.bad_fix_ratio * 100).toFixed(1)}%</p>
                        <p class="text-xs text-gray-600">Max Physical Jerk: ${hr.metrics.max_jerk_detected.toFixed(1)} m/s³</p>
                    </div>
                    <div class="p-3 bg-gray-50 rounded border shadow-sm flex flex-col justify-center">
                        <p class="text-[10px] text-gray-500 uppercase font-bold tracking-wider">${msg.original_name} - IMU Integrity</p>
                        <p class="text-2xl font-bold ${imuColor}">${hr.imu_score.toFixed(1)}%</p>
                        <p class="text-xs text-gray-600 mt-1">1G Deviation: ${hr.metrics.avg_grav_mag_error.toFixed(4)} G</p>
                    </div>
                `;
                
                if (hr.warnings.length > 0) {
                    hr.warnings.forEach(w => {
                        const li = document.createElement("li");
                        li.textContent = `[${msg.original_name}] ${w}`;
                        healthWarnings.appendChild(li);
                    });
                }
                return;
            }

            if (msg.type === "item_error") {
                console.warn(`[GPMF Skip] ${msg.original_name}: ${msg.message}`);
                
                warningsContainer.classList.remove("hidden");
                const li = document.createElement("li");
                li.textContent = `Skipped ${msg.original_name}: ${msg.message}`;
                warningsList.appendChild(li);
                
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
                fullResults.push(r);
                
                let hasDefects = r.geojson && r.geojson.length > 0;
                if (hasDefects) {
                    fullGeojson.features.push(...r.geojson);
                    updateMapSource('defects-source', fullGeojson);
                }

                const fIndex = nodesGeoJson.features.findIndex(f => f.properties.original_name === r.original_name);
                if (fIndex > -1) {
                    nodesGeoJson.features[fIndex].properties.processed = true;
                    nodesGeoJson.features[fIndex].properties.location = r.location;
                } else if (r.lat !== null && r.lon !== null) {
                    nodesGeoJson.features.push({
                        type: "Feature",
                        geometry: { type: "Point", coordinates: [r.lon, r.lat] },
                        properties: { original_name: r.original_name, location: r.location, processed: true, active: false }
                    });
                }
                updateMapSource('nodes-source', nodesGeoJson);

                processedCount++;
                const pct = totalImages > 0 ? (processedCount / totalImages) * 100 : 100;
                document.getElementById("progress-bar").style.width = `${pct}%`;
                document.getElementById("progress-text").textContent = `Segmenting ${processedCount} of ${totalImages}`;

                const elapsedSec = (Date.now() - startTime) / 1000;
                const remainSec = Math.ceil((totalImages - processedCount) * (elapsedSec / processedCount));
                document.getElementById("eta-text").textContent = `ETA: ${Math.floor(remainSec / 60)}m ${remainSec % 60}s`;

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

        selLocation.onchange = () => { appResults = fullResults.filter(r => r.location === selLocation.value); currentIndex = 0; updateCarousel(true); };
        appResults = fullResults.filter(r => r.location === selLocation.value);
        if(appResults.length > 0 && document.getElementById("img-rect").classList.contains("hidden")) updateCarousel(false);
    }

    document.getElementById("btn-save-project").onclick = () => {
        if (fullResults.length === 0) return;
        const blob = new Blob([JSON.stringify({ is_360: appIs360, results: fullResults, geojson: fullGeojson })], { type: "application/json" });
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
                
                appIs360 = data.is_360 !== undefined ? data.is_360 : true;
                if (!appIs360) { containerBevRear.classList.add("hidden"); setView('front'); } 
                else containerBevRear.classList.remove("hidden");

                fullResults = data.results;
                fullGeojson = data.geojson;
                
                if (fullResults.length > 0) {
                    const lastRec = fullResults[fullResults.length - 1];
                    stateLastLat = lastRec.lat; stateLastLon = lastRec.lon;
                    stateLastLocId = parseInt(lastRec.location.replace("Location ", "")) || 1;
                }

                uploadPanel.classList.add("hidden");
                document.getElementById("workspace").classList.remove("hidden");
                document.getElementById("btn-save-project").classList.remove("hidden");
                document.getElementById("btn-export-zip").classList.remove("hidden");
                document.getElementById("btn-export-flat-zip").classList.remove("hidden");
                
                nodesGeoJson = { type: "FeatureCollection", features: [] };
                let trailCoords = [];

                fullResults.forEach(img => {
                    if (img.lat !== null && img.lon !== null) {
                        trailCoords.push([img.lon, img.lat]);
                        nodesGeoJson.features.push({
                            type: "Feature",
                            geometry: { type: "Point", coordinates: [img.lon, img.lat] },
                            properties: { original_name: img.original_name, location: img.location, processed: true, active: false }
                        });
                    }
                });

                if (trailCoords.length > 1) {
                    trailGeoJson = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "LineString", coordinates: trailCoords }, properties: {} }] };
                } else {
                    trailGeoJson = { type: "FeatureCollection", features: [] };
                }

                initMap();

                setTimeout(() => {
                    if (map) map.resize();
                    updateMapSource('defects-source', fullGeojson);
                    updateMapSource('nodes-source', nodesGeoJson);
                    updateMapSource('trail-source', trailGeoJson);

                    if (fullGeojson.features.length > 0) fitMapToBounds(fullGeojson);
                    else if (trailGeoJson.features.length > 0) fitMapToBounds(trailGeoJson);

                    refreshLocationsUI(); 
                    setView('front'); 
                    if (appResults.length > 0) updateCarousel(true);
                }, 300);
            } catch (err) { alert("Error loading project: " + err.message); }
        };
        reader.readAsText(file);
    });

    const triggerZipExport = async (endpoint, btnId, loadingText, filename) => {
        if (fullResults.length === 0) return;
        const btn = document.getElementById(btnId); const originalText = btn.textContent;
        btn.textContent = loadingText; btn.disabled = true;
        try {
            const res = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ results: fullResults }) });
            if (!res.ok) throw new Error("Failed to compile ZIP file");
            const blob = await res.blob(); const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
        } catch (err) { alert(err.message); } finally { btn.textContent = originalText; btn.disabled = false; }
    };

    document.getElementById("btn-export-zip").onclick = () => triggerZipExport("/export-zip", "btn-export-zip", "⏳ Compiling ZIP...", "DCPM_RAW_Export.zip");
    document.getElementById("btn-export-flat-zip").onclick = () => triggerZipExport("/export-flat-zip", "btn-export-flat-zip", "⏳ Compiling Flattened ZIP...", "DCPM_FLAT_Export.zip");

    function setView(dir) {
        currentDirection = dir;
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

    document.getElementById('container-bev-front').onclick = () => setView('front');
    document.getElementById('container-bev-rear').onclick = () => setView('rear');

    function updateCarousel(panMap = true) {
        if (appResults.length === 0) return;
        const current = appResults[currentIndex];
        
        document.getElementById("placeholder-rect").classList.add("hidden");
        document.getElementById("img-rect").classList.remove("hidden");
        document.getElementById("img-bev-front").classList.remove("hidden");
        if (appIs360) document.getElementById("img-bev-rear").classList.remove("hidden");

        const activeViewData = current.views[currentDirection] || current.views['front'];
        document.getElementById("carousel-counter").textContent = `Item ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name;
        document.getElementById("carousel-telemetry").textContent = `Pitch: ${current.pitch}° | Roll: ${current.roll}°`;

        document.getElementById("img-bev-front").src = current.views['front'].bev_url;
        if (appIs360 && current.views['rear']) document.getElementById("img-bev-rear").src = current.views['rear'].bev_url;
        document.getElementById("img-rect").src = activeViewData.rect_url;

        document.getElementById("table-defects").innerHTML = activeViewData.defects.map(d => `<tr><td class="p-2"><span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: ${d.color || stringToColor(d.class)}; border: 1px solid #ccc;"></span>${d.class}</td><td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td><td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td></tr>`).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No detections</td></tr>`;

        activeMarkerFilename = current.original_name;
        nodesGeoJson.features.forEach(f => {
            f.properties.active = (f.properties.original_name === activeMarkerFilename);
        });
        updateMapSource('nodes-source', nodesGeoJson);

        if (panMap && mapLoaded && current.lat !== null) {
            map.flyTo({ center: [current.lon, current.lat], zoom: 20, speed: 1.5 });
        }

        document.getElementById("btn-prev").disabled = (currentIndex === 0);
        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
    }

    document.getElementById("btn-prev").onclick = () => { if (currentIndex > 0) { currentIndex--; updateCarousel(true); } };
    document.getElementById("btn-next").onclick = () => { if (currentIndex < appResults.length - 1) { currentIndex++; updateCarousel(true); } };
});