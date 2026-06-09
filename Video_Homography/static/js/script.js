document.addEventListener("DOMContentLoaded", () => {
    let modelFile = null;
    let imageFiles = [];
    
    let map = null;
    let geoJsonLayer = null;
    let mapMarkers = {}; // Stores { "filename.jpg": leaflet_marker_object }
    
    let appResults = [];
    let currentIndex = 0;

    const btnProcess = document.getElementById("process-btn");

    function initMap() {
        if (!map) {
            map = L.map('map').setView([-32.06, 151.90], 15);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 19
            }).addTo(map);
        }
    }

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

    btnProcess.onclick = async () => {
        const fd = new FormData();
        fd.append("model", modelFile);
        imageFiles.forEach(f => fd.append("images", f));
        fd.append("cam_height", document.getElementById("cam-height").value);

        document.getElementById("loading").classList.remove("hidden");
        document.getElementById("workspace").classList.add("hidden");
        btnProcess.disabled = true;

        try {
            const res = await fetch("/process", { method: "POST", body: fd });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            appResults = data.results;
            currentIndex = 0;

            initMap();
            document.getElementById("workspace").classList.remove("hidden");
            
            // FIX: Force Leaflet to recalculate bounds after un-hiding the div
            setTimeout(() => { map.invalidateSize(); }, 200);

            renderMap(data.geojson);
            updateCarousel();

        } catch (e) {
            alert(e.message);
        } finally {
            document.getElementById("loading").classList.add("hidden");
            btnProcess.disabled = false;
        }
    };

    function renderMap(geoJsonData) {
        if (geoJsonLayer) map.removeLayer(geoJsonLayer);
        mapMarkers = {}; // Reset markers

        geoJsonLayer = L.geoJSON(geoJsonData, {
            style: function(feature) {
                if (feature.geometry.type === 'Polygon') {
                    return { color: "#ff0000", weight: 2, fillColor: "#ffaa00", fillOpacity: 0.5 };
                }
                if (feature.geometry.type === 'LineString') {
                    return { color: "#00b4d8", weight: 3, dashArray: "5, 10" }; // The driving trail
                }
            },
            pointToLayer: function(feature, latlng) {
                const marker = L.circleMarker(latlng, {
                    radius: 5, fillColor: "#3b82f6", color: "#fff", weight: 2, opacity: 1, fillOpacity: 0.9
                });
                if (feature.properties.type === 'camera') {
                    mapMarkers[feature.properties.filename] = marker;
                }
                return marker;
            },
            onEachFeature: function(feature, layer) {
                if (feature.geometry.type === 'Polygon') {
                    layer.bindPopup(`<b>${feature.properties.class}</b><br>Area: ${feature.properties.area_sqm} m²`);
                } else if (feature.geometry.type === 'Point') {
                    layer.bindPopup(`<b>Photo Location</b><br>${feature.properties.filename}`);
                }
            }
        }).addTo(map);

        if(geoJsonData.features.length > 0) {
            map.fitBounds(geoJsonLayer.getBounds(), { padding: [50, 50] });
        }
    }

    function updateCarousel() {
        if (appResults.length === 0) return;
        const current = appResults[currentIndex];

        // Update Text
        document.getElementById("carousel-counter").textContent = `Image ${currentIndex + 1} of ${appResults.length}`;
        document.getElementById("carousel-filename").textContent = current.original_name;
        document.getElementById("carousel-telemetry").textContent = `Auto-Pitch: ${current.pitch}°`;

        // Update Images
        document.getElementById("img-rect").src = current.rect_url;
        document.getElementById("img-bev").src = current.bev_url;

        // Update Table
        const tbody = document.getElementById("table-defects");
        tbody.innerHTML = current.defects.map(d => `
            <tr>
                <td class="p-2">${d.class}</td>
                <td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td>
                <td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td>
            </tr>
        `).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No defects detected</td></tr>`;

        // Sync Map! Pan to the camera dot and open popup
        const activeMarker = mapMarkers[current.original_name];
        if (activeMarker) {
            map.setView(activeMarker.getLatLng(), 18, { animate: true });
            activeMarker.openPopup();
        }

        // Handle button states
        document.getElementById("btn-prev").disabled = (currentIndex === 0);
        document.getElementById("btn-next").disabled = (currentIndex === appResults.length - 1);
    }

    // Carousel Navigation
    document.getElementById("btn-prev").onclick = () => {
        if (currentIndex > 0) { currentIndex--; updateCarousel(); }
    };
    document.getElementById("btn-next").onclick = () => {
        if (currentIndex < appResults.length - 1) { currentIndex++; updateCarousel(); }
    };
});