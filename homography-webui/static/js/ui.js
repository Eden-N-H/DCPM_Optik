import { state } from './state.js';
import { updateMapSource } from './map.js';
import { stringToColor } from './utils.js';

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
    if (!files.length) return;
    if (isMulti) { callback(Array.from(files)); nameElement.textContent = `${files.length} items queued`; } 
    else { callback(files[0]); nameElement.textContent = files[0].name; document.getElementById("status-model").classList.remove("hidden"); state.isModelLoaded = true; }
    nameElement.classList.remove("hidden");
    checkCanProcess();
};

export function setupDz(dzId, inId, nameId, isMulti, callback) {
    const dz = document.getElementById(dzId);
    const inp = document.getElementById(inId);
    const nm = document.getElementById(nameId);

    dz.onclick = () => inp.click();
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("border-blue-500"); };
    dz.ondragleave = () => dz.classList.remove("border-blue-500");
    dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("border-blue-500"); handleFiles(e.dataTransfer.files, isMulti, callback, nm); };
    inp.onchange = (e) => handleFiles(e.target.files, isMulti, callback, nm);
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
    
    document.getElementById("placeholder-rect").classList.add("hidden");
    document.getElementById("img-rect").classList.remove("hidden");
    document.getElementById("img-bev-front").classList.remove("hidden");
    if (state.appIs360) document.getElementById("img-bev-rear").classList.remove("hidden");

    const activeViewData = current.views[state.currentDirection] || current.views['front'];
    document.getElementById("carousel-counter").textContent = `Item ${state.currentIndex + 1} of ${state.appResults.length}`;
    document.getElementById("carousel-filename").textContent = current.original_name;
    document.getElementById("carousel-telemetry").textContent = `Pitch: ${current.pitch}° | Roll: ${current.roll}°`;

    document.getElementById("img-bev-front").src = current.views['front'].bev_url;
    if (state.appIs360 && current.views['rear']) document.getElementById("img-bev-rear").src = current.views['rear'].bev_url;
    document.getElementById("img-rect").src = activeViewData.rect_url;

    document.getElementById("table-defects").innerHTML = activeViewData.defects.map(d => `<tr><td class="p-2"><span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: ${d.color || stringToColor(d.class)}; border: 1px solid #ccc;"></span>${d.class}</td><td class="p-2 text-gray-500">${(d.conf*100).toFixed(0)}%</td><td class="p-2 font-bold text-red-600">${d.area_sqm} m²</td></tr>`).join('') || `<tr><td colspan="3" class="p-2 text-center text-gray-500">No detections</td></tr>`;

    state.activeMarkerFilename = current.original_name;
    state.nodesGeoJson.features.forEach(f => {
        f.properties.active = (f.properties.original_name === state.activeMarkerFilename);
    });
    updateMapSource('nodes-source', state.nodesGeoJson);

    if (panMap && state.mapLoaded && current.lat !== null) {
        state.map.flyTo({ center: [current.lon, current.lat], zoom: 20, speed: 1.5 });
    }

    document.getElementById("btn-prev").disabled = (state.currentIndex === 0);
    document.getElementById("btn-next").disabled = (state.currentIndex === state.appResults.length - 1);
}