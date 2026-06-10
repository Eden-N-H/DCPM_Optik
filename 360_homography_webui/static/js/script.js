document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let imageFiles = [];
    
    let map = null;
    let geoJsonLayer = null;
    let mapMarkers = {}; // Stores { "filename.jpg": leaflet_marker_object }
    
    // State Management Variables
    let fullResults = [];
    let fullGeojson = null;
    let appResults = []; // Filtered results for the current active location
    let currentIndex = 0;
    let currentDirection = 'front'; // Track toggle state ('front' or 'rear')

    const btnProcess = document.getElementById("process-btn");
    const selLocation = document.getElementById("sel-location");

    // ==========================================
    // Initialization & Map Logic
    // ==========================================
    function initMap() {
        if (!map) {
            map = L.map('map').setView([-32.06, 151.90], 15);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 24,          // Allow the map user to zoom in infinitely close
                maxNativeZoom: 17     // Stop requesting tiles past zoom 17
            }).addTo(map);
        }
    }

    // ==========================================
    // File Upload Setup
    // ==========================================
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
        }
        nameElement.classList.remove("hidden");
        if (modelFile && imageFiles.length > 0) btnProcess.disabled = false;
    };

    setupDz("dz-model", "in-model", "name-model", false, f => modelFile = f);
    setupDz("dz-image", "in-image", "name-image", true, f => imageFiles = f);

    // ==========================================
    // Process Pipeline Logic
    // ==========================================
    btnProcess.onclick = async () => {
        const fd = new FormData();
        fd.append("model", modelFile);
        imageFiles.forEach(f => fd.append("images", f));
        fd.append("cam_height", document.getElementById("cam-height").value);

        document.getElementById("loading").classList.remove("hidden");
        document.getElementById("upload-panel").classList.add("hidden");
        btnProcess.disabled = true;

        try {
            const res = await fetch("/process", { method: "POST", body: fd });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            loadWorkspace(data.results, data.geojson);

        } catch (e) {
            alert(e.message);
            document.getElementById("upload-panel").classList.remove("hidden");
        } finally {
            document.getElementById("loading").classList.add("hidden");
            btnProcess.disabled = false;
        }
    };

    // ==========================================
    // Save / Load Project JSON
    // ==========================================
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
                loadWorkspace(data.results, data.geojson);
            } catch (err) {
                alert("Error loading project: " + err.message);
            }
        };
        reader.readAsText(file);
    });

    // ==========================================
    // Workspace Management
    // ==========================================
    function loadWorkspace(resultsData, geojsonData) {
        fullResults = resultsData;
        fullGeojson = geojsonData;

        initMap();
        document.getElementById("upload-panel").classList.add("hidden");
        document.getElementById("workspace").classList.remove("hidden");
        document.getElementById("btn-save-project").classList.remove("hidden");
        
        setTimeout(() => { map.invalidateSize(); }, 200);

        renderMap(fullGeojson);
        populateLocations();
        setView('front'); // Initialize to front view
    }

    function populateLocations() {
        const locations = [...new Set(fullResults.map(r => r.location))];
        selLocation.innerHTML = locations.map(loc => `<option value="${loc}">${loc}</option>`).join("");
        
        selLocation.onchange = () => {
            appResults = fullResults.filter(r => r.location === selLocation.value);
            currentIndex = 0;
            updateCarousel(true);
        };
        
        if (locations.length > 0) {
            selLocation.value = locations[0];
            selLocation.dispatchEvent(new Event('change'));
        }
    }

    function renderMap(geoJsonData) {
        if (geoJsonLayer) map.removeLayer(geoJsonLayer);
        mapMarkers = {};

        geoJsonLayer = L.geoJSON(geoJsonData, {
            style: function(feature) {
                if (feature.geometry.type === 'Polygon') {
                    const fColor = feature.properties.view === 'rear' ? "#f59e0b" : "#ffaa00"; 
                    return { color: "#ff0000", weight: 2, fillColor: fColor, fillOpacity: 0.5 };
                }
                if (feature.geometry.type === 'LineString') {
                    return { color: "#00b4d8", weight: 3, dashArray: "5, 10" }; 
                }
            },
            pointToLayer: function(feature, latlng) {
                const marker = L.circleMarker(latlng, {
                    radius: 5, fillColor: "#3b82f6", color: "#fff", weight: 2, opacity: 1, fillOpacity: 0.9
                });
                
                if (feature.properties.type === 'camera') {
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
            onEachFeature: function(feature, layer) {
                if (feature.geometry.type === 'Polygon') {
                    layer.bindPopup(`<b>${feature.properties.class}</b><br>View: ${feature.properties.view}<br>Area: ${feature.properties.area_sqm} m²`);
                } else if (feature.geometry.type === 'Point') {
                    layer.bindPopup(`<b>Photo Location</b><br>${feature.properties.filename}`);
                }
            }
        }).addTo(map);

        if(geoJsonData.features.length > 0) {
            map.fitBounds(geoJsonLayer.getBounds(), { padding: [50, 50] });
        }
    }

    // ==========================================
    // View Click / Toggle Setup
    // ==========================================
    function setView(dir) {
        currentDirection = dir;
        const contF = document.getElementById('container-bev-front');
        const contR = document.getElementById('container-bev-rear');
        const activeLabel = document.getElementById('label-active-view');
        
        // Update UI styling for selected BEV
        if (dir === 'front') {
            contF.classList.add('border-blue-500', 'ring-2', 'ring-blue-100');
            contF.classList.remove('border-transparent');
            
            contR.classList.remove('border-blue-500', 'ring-2', 'ring-blue-100');
            contR.classList.add('border-transparent');
            
            activeLabel.textContent = 'Front View Active';
        } else {
            contR.classList.add('border-blue-500', 'ring-2', 'ring-blue-100');
            contR.classList.remove('border-transparent');
            
            contF.classList.remove('border-blue-500', 'ring-2', 'ring-blue-100');
            contF.classList.add('border-transparent');
            
            activeLabel.textContent = 'Rear View Active';
        }
        
        // Refresh annotations and table
        updateCarousel(false);
    }

    document.getElementById('container-bev-front').onclick = () => setView('front');
    document.getElementById('container-bev-rear').onclick = () => setView('rear');

    // ==========================================
    // UI Update Logic
    // ==========================================
    function updateCarousel(panMap = true) {
        if (appResults.length === 0) return;
        const current = appResults[currentIndex];
        
        // The table and perspective image belong to whichever view is selected
        const activeViewData = current.views[currentDirection];

        // Update Global Text
        document.getElementById("carousel-counter").textContent = `Image ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name;
        document.getElementById("carousel-telemetry").textContent = `Auto-Pitch: ${current.pitch}°`;

        // Update BOTH BEV Images constantly
        document.getElementById("img-bev-front").src = current.views['front'].bev_url;
        document.getElementById("img-bev-rear").src = current.views['rear'].bev_url;

        // Update the Single Perspective Image
        document.getElementById("img-rect").src = activeViewData.rect_url;

        // Update Table
        const tbody = document.getElementById("table-defects");
        tbody.innerHTML = activeViewData.defects.map(d => `
            <tr>
                <td class="p-2">${d.class}</td>
                <td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td>
                <td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td>
            </tr>
        `).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No defects in this view</td></tr>`;

        // Sync Map Marker
        const activeMarker = mapMarkers[current.original_name];
        if (activeMarker) {
            if (panMap) {
                map.setView(activeMarker.getLatLng(), 20, { animate: true });
            }
            activeMarker.openPopup();
        }

        // Handle button states
        document.getElementById("btn-prev").disabled = (currentIndex === 0);
        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
    }

    // ==========================================
    // Carousel Navigation
    // ==========================================
    document.getElementById("btn-prev").onclick = () => {
        if (currentIndex > 0) { currentIndex--; updateCarousel(true); }
    };
    document.getElementById("btn-next").onclick = () => {
        if (currentIndex < appResults.length - 1) { currentIndex++; updateCarousel(true); }
    };
});