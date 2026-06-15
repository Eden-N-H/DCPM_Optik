document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let isModelLoaded = false;
    let imageFiles = [];
    
    let map = null;
    let geoJsonLayer = null;
    let pathLayer = null;
    let mapMarkers = {}; // { "filename.jpg": leaflet_marker }
    
    // Global Persistent State
    let fullResults = [];
    let fullGeojson = { type: "FeatureCollection", features: [] };
    
    // UI State
    let appResults = []; 
    let currentIndex = 0;
    let currentDirection = 'front'; 

    // Incremental GPS Clustering Tracking
    let stateLastLat = 0.0;
    let stateLastLon = 0.0;
    let stateLastLocId = 1;

    const btnProcess = document.getElementById("process-btn");
    const selLocation = document.getElementById("sel-location");
    const uploadPanel = document.getElementById("upload-panel");

    // ==========================================
    // Helper: Fallback Color Generator
    // ==========================================
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

    // ==========================================
    // Initialization & Map Logic
    // ==========================================
    function initMap() {
        if (!map) {
            map = L.map('map').setView([-32.06, 151.90], 15);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 24, maxNativeZoom: 17 
            }).addTo(map);

            geoJsonLayer = L.geoJSON(null, {
                style: (feature) => {
                    if (feature.geometry && feature.geometry.type === 'LineString') {
                        return { color: "#00e5ff", weight: 3, dashArray: "5, 10" }; 
                    }
                    
                    // Reads the backend python color, or uses fallback if loading old project JSON
                    const fColor = feature.properties.color || stringToColor(feature.properties.class || "Unknown");
                    
                    return { 
                        color: "#ffffff",     // Solid White Border/Stroke
                        weight: 2,            // Border thickness
                        opacity: 1,           // Border 100% opacity
                        fillColor: fColor,    // Exact YOLO overlay color
                        fillOpacity: 0.9      // 90% opacity
                    };
                },
                pointToLayer: (feature, latlng) => {
                    const marker = L.circleMarker(latlng, {
                        radius: 7, 
                        fillColor: "#2563eb", // Vivid Blue
                        color: "#ffffff",     // White border
                        weight: 2, 
                        opacity: 1,           
                        fillOpacity: 1        
                    });
                    
                    if (feature.properties && feature.properties.filename) {
                        mapMarkers[feature.properties.filename] = marker;
                        marker.on('click', () => {
                            const target = fullResults.find(r => r.original_name === feature.properties.filename);
                            if (target) {
                                if (selLocation.value !== target.location) {
                                    selLocation.value = target.location;
                                    appResults = fullResults.filter(r => r.location === target.location);
                                }
                                currentIndex = appResults.findIndex(r => r.original_name === target.original_name);
                                updateCarousel(false);
                            }
                        });
                    }
                    return marker;
                },
                onEachFeature: (feature, layer) => {
                    if (feature.geometry && feature.geometry.type === 'Polygon') {
                        layer.bindPopup(`<b>${feature.properties.class}</b><br>View: ${feature.properties.view}<br>Area: ${feature.properties.area_sqm} m²`);
                    } else if (feature.geometry && feature.geometry.type === 'Point') {
                        layer.bindPopup(`<b>Photo Location</b><br>${feature.properties.filename || 'Unknown'}`);
                    }
                }
            }).addTo(map);

            pathLayer = L.geoJSON(null, {
                style: { color: "#00e5ff", weight: 3, dashArray: "5, 10" }
            }).addTo(map);
        }
    }

    // ==========================================
    // File Upload Setup
    // ==========================================
    document.getElementById("btn-toggle-upload").onclick = () => {
        uploadPanel.classList.toggle("hidden");
    };

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
            nameElement.textContent = `${files.length} images queued`;
        } else {
            callback(files[0]);
            nameElement.textContent = files[0].name;
            document.getElementById("status-model").classList.remove("hidden");
            isModelLoaded = true;
        }
        nameElement.classList.remove("hidden");
        if ((isModelLoaded || modelFile) && imageFiles.length > 0) btnProcess.disabled = false;
    };

    setupDz("dz-model", "in-model", "name-model", false, f => modelFile = f);
    setupDz("dz-image", "in-image", "name-image", true, f => imageFiles = f);

    // ==========================================
    // Asynchronous Pipeline Logic & SSE
    // ==========================================
    btnProcess.onclick = async () => {
        const fd = new FormData();
        if (modelFile) fd.append("model", modelFile);
        imageFiles.forEach(f => fd.append("images", f));
        fd.append("cam_height", document.getElementById("cam-height").value);
        
        fd.append("last_lat", stateLastLat);
        fd.append("last_lon", stateLastLon);
        fd.append("last_loc_id", stateLastLocId);

        uploadPanel.classList.add("hidden");
        document.getElementById("workspace").classList.remove("hidden");
        document.getElementById("btn-save-project").classList.remove("hidden");
        document.getElementById("btn-export-zip").classList.remove("hidden");
        document.getElementById("btn-export-flat-zip").classList.remove("hidden");
        document.getElementById("progress-container").classList.remove("hidden");
        
        btnProcess.disabled = true;
        initMap();

        try {
            const res = await fetch("/process", { method: "POST", body: fd });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            stateLastLat = data.last_lat;
            stateLastLon = data.last_lon;
            stateLastLocId = data.last_loc_id;

            setTimeout(() => {
                map.invalidateSize();

                if (data.initial_trail && data.initial_trail.features) {
                    pathLayer.addData(data.initial_trail);
                    if(pathLayer.getBounds().isValid()) {
                        map.fitBounds(pathLayer.getBounds(), { padding: [50, 50] });
                    }
                }

                data.initial_state.forEach(img => {
                    if(img.lat !== 0.0) {
                        const marker = L.circleMarker([img.lat, img.lon], {
                            radius: 7, 
                            fillColor: "#ff0000", // Red
                            color: "#ffffff",     // White border
                            weight: 2, 
                            opacity: 1, 
                            fillOpacity: 1
                        }).addTo(map);
                        marker.bindPopup(`<b>Pending Process</b><br>${img.original_name}`);
                        mapMarkers[img.original_name] = marker;
                    }
                });

                startSSE(data.task_id, data.total_images);
            }, 200);

        } catch (e) {
            alert(e.message);
            uploadPanel.classList.remove("hidden");
            document.getElementById("progress-container").classList.add("hidden");
            btnProcess.disabled = false;
        }
    };

    function startSSE(taskId, totalImages) {
        const source = new EventSource(`/stream/${taskId}`);
        let processedCount = 0;
        let startTime = Date.now();

        source.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === "error" || msg.type === "complete") {
                source.close();
                document.getElementById("progress-container").classList.add("hidden");
                btnProcess.disabled = false;
                
                imageFiles = [];
                document.getElementById("name-image").textContent = "Folder completed.";
                
                if (msg.type === "error") alert(`Background Task Error: ${msg.message}`);
                return;
            }

            if (msg.type === "update") {
                const r = msg.data;
                fullResults.push(r);
                
                if (r.geojson && r.geojson.length > 0) {
                    fullGeojson.features.push(...r.geojson);
                    geoJsonLayer.addData(r.geojson);
                }

                if (mapMarkers[r.original_name]) {
                    const marker = mapMarkers[r.original_name];
                    marker.setStyle({ 
                        radius: 7, 
                        fillColor: "#2563eb", // Vivid Blue
                        color: "#ffffff", 
                        weight: 2, 
                        opacity: 1, 
                        fillOpacity: 1 
                    });
                    marker.setPopupContent(`<b>Photo Location</b><br>${r.original_name}`);
                    
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
                const pct = (processedCount / totalImages) * 100;
                document.getElementById("progress-bar").style.width = `${pct}%`;
                document.getElementById("progress-text").textContent = `Segmenting ${processedCount} of ${totalImages}`;

                const elapsedSec = (Date.now() - startTime) / 1000;
                const avgSpeed = elapsedSec / processedCount;
                const remainSec = Math.ceil((totalImages - processedCount) * avgSpeed);
                const mins = Math.floor(remainSec / 60);
                const secs = remainSec % 60;
                document.getElementById("eta-text").textContent = `ETA: ${mins}m ${secs}s`;

                refreshLocationsUI();
                
                if (fullResults.length === 1) {
                    updateCarousel(true);
                }
            }
        };
    }

    // ==========================================
    // UI Helpers & Project Load/Save
    // ==========================================
    function refreshLocationsUI() {
        const locations = [...new Set(fullResults.map(r => r.location))];
        const currentSelection = selLocation.value;
        
        selLocation.innerHTML = locations.map(loc => `<option value="${loc}">${loc}</option>`).join("");
        
        if (locations.includes(currentSelection)) {
            selLocation.value = currentSelection;
        } else if (locations.length > 0) {
            selLocation.value = locations[0];
        }

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
        const projectData = { results: fullResults, geojson: fullGeojson };
        const blob = new Blob([JSON.stringify(projectData, null, 2)], { type: "application/json" });
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
                    
                    fullResults.forEach(img => {
                        if (!mapMarkers[img.original_name] && img.lat !== 0.0) {
                            const marker = L.circleMarker([img.lat, img.lon], {
                                radius: 7, 
                                fillColor: "#2563eb", // Blue
                                color: "#ffffff", 
                                weight: 2, 
                                opacity: 1, 
                                fillOpacity: 1
                            }).addTo(map);
                            marker.bindPopup(`<b>Photo Location</b><br>${img.original_name}`);
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
                }, 200);

            } catch (err) {
                alert("Error loading project: " + err.message);
            }
        };
        reader.readAsText(file);
    });

    // ==========================================
    // Bulk Export ZIP Logic
    // ==========================================
    document.getElementById("btn-export-zip").onclick = async () => {
        if (fullResults.length === 0) return;
        
        const btn = document.getElementById("btn-export-zip");
        const originalText = btn.textContent;
        btn.textContent = "⏳ Compiling ZIP...";
        btn.disabled = true;

        try {
            const res = await fetch("/export-zip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ results: fullResults })
            });
            
            if (!res.ok) throw new Error("Failed to compile ZIP file");
            
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "DCPM_Export.zip";
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

    document.getElementById("btn-export-flat-zip").onclick = async () => {
        if (fullResults.length === 0) return;
        
        const btn = document.getElementById("btn-export-flat-zip");
        const originalText = btn.textContent;
        btn.textContent = "⏳ Compiling Flattened ZIP...";
        btn.disabled = true;

        try {
            const res = await fetch("/export-flat-zip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ results: fullResults })
            });
            
            if (!res.ok) throw new Error("Failed to compile ZIP file");
            
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "DCPM_Flattened_Export.zip";
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

    // ==========================================
    // Carousel Interaction
    // ==========================================
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
        imgBevR.classList.remove("hidden");

        const activeViewData = current.views[currentDirection];

        document.getElementById("carousel-counter").textContent = `Image ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name;
        document.getElementById("carousel-telemetry").textContent = `Pitch: ${current.pitch}°`;

        imgBevF.src = current.views['front'].bev_url;
        imgBevR.src = current.views['rear'].bev_url;
        imgRect.src = activeViewData.rect_url;

        const tbody = document.getElementById("table-defects");
        tbody.innerHTML = activeViewData.defects.map(d => {
            // Check for the backend color, fallback to JS color if loading an old project.
            const classColor = d.color || stringToColor(d.class);
            return `
            <tr>
                <td class="p-2"><span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: ${classColor}; border: 1px solid #ccc;"></span>${d.class}</td>
                <td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td>
                <td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td>
            </tr>
            `;
        }).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No defects found</td></tr>`;

        const activeMarker = mapMarkers[current.original_name];
        if (activeMarker) {
            if (panMap) map.setView(activeMarker.getLatLng(), 20, { animate: true });
            activeMarker.openPopup();
        }

        document.getElementById("btn-prev").disabled = (currentIndex === 0);
        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
    }

    document.getElementById("btn-prev").onclick = () => {
        if (currentIndex > 0) { currentIndex--; updateCarousel(true); }
    };
    document.getElementById("btn-next").onclick = () => {
        if (currentIndex < appResults.length - 1) { currentIndex++; updateCarousel(true); }
    };
});