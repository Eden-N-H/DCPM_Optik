import { state } from './state.js';

export function toggleMapLayerVisibility(view, isVisible) {
    if (!state.mapLoaded || !state.map) return;
    const visibility = isVisible ? 'visible' : 'none';
    state.orthoLayerIds.forEach(id => {
        if (id.endsWith(`-${view}`)) {
            if (state.map.getLayer(id)) {
                state.map.setLayoutProperty(id, 'visibility', visibility);
            }
        }
    });
}

export function updateMapSource(sourceId, data) {
    if (state.mapLoaded && state.map) {
        const src = state.map.getSource(sourceId);
        if (src) src.setData(data);
    }
}

export function fitMapToBounds(geoJsonData, maxZoom = 20) {
    if (!state.mapLoaded || !state.map || !geoJsonData || !geoJsonData.features || geoJsonData.features.length === 0) return;
    try {
        const bbox = turf.bbox(geoJsonData);
        if (bbox.some(val => isNaN(val) || !isFinite(val))) return;
        state.map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 50, maxZoom: maxZoom });
    } catch (e) {
        console.warn("Could not calculate bounds", e);
    }
}

export function initMap(onMapClickCallback) {
    if (!state.map) {
        state.map = new maplibregl.Map({
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

        state.mapPopup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

        state.map.on('load', () => {
            state.mapLoaded = true;

            state.map.addSource('trail-source', { type: 'geojson', data: state.trailGeoJson });
            state.map.addSource('defects-source', { type: 'geojson', data: state.fullGeojson });
            state.map.addSource('nodes-source', { type: 'geojson', data: state.nodesGeoJson });

            state.map.addLayer({
                id: 'trail-layer', type: 'line', source: 'trail-source',
                paint: { 'line-color': '#94a3b8', 'line-width': 2, 'line-dasharray': [2, 3] }
            });

            state.map.addLayer({
                id: 'defects-layer', type: 'fill', source: 'defects-source',
                paint: {
                    'fill-color': ['coalesce', ['get', 'color'], '#ffffff'],
                    'fill-opacity': ['coalesce', ['get', 'conf'], 0.4]
                }
            });

            state.map.addLayer({
                id: 'defects-outline-layer', type: 'line', source: 'defects-source',
                paint: { 'line-color': '#ffffff', 'line-width': 1.5 }
            });

            state.map.addLayer({
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

            state.map.on('mouseenter', 'nodes-layer', (e) => {
                state.map.getCanvas().style.cursor = 'pointer';
                const f = e.features[0];
                const status = f.properties.processed ? "Photo Location" : "Pending Process";
                state.mapPopup.setLngLat(e.lngLat).setHTML(`<div class="text-xs"><b>${status}</b><br>${f.properties.original_name}</div>`).addTo(state.map);
            });
            
            state.map.on('mouseleave', 'nodes-layer', () => {
                state.map.getCanvas().style.cursor = '';
                state.mapPopup.remove();
            });
            
            state.map.on('click', 'nodes-layer', (e) => {
                const f = e.features[0];
                if (!f.properties.processed) return;
                if (onMapClickCallback) onMapClickCallback(f.properties.original_name);
            });

            state.map.on('mouseenter', 'defects-layer', (e) => {
                state.map.getCanvas().style.cursor = 'pointer';
                const f = e.features[0];
                state.mapPopup.setLngLat(e.lngLat).setHTML(`<div class="text-xs"><b>${f.properties.class}</b><br>View: ${f.properties.view}<br>Area: ${f.properties.area_sqm} m²</div>`).addTo(state.map);
            });
            
            state.map.on('mouseleave', 'defects-layer', () => {
                state.map.getCanvas().style.cursor = '';
                state.mapPopup.remove();
            });
            
            state.map.on('click', 'defects-layer', (e) => {
                const f = e.features[0];
                if (onMapClickCallback) onMapClickCallback(f.properties.filename);
            });
            
            if (state.fullGeojson.features.length > 0) fitMapToBounds(state.fullGeojson);
            else if (state.trailGeoJson.features.length > 0) fitMapToBounds(state.trailGeoJson);
        });
    }
}

export function clearOrthomosaics() {
    if (!state.mapLoaded || !state.map) return;
    state.orthoLayerIds.forEach(id => {
        if (state.map.getLayer(id)) state.map.removeLayer(id);
        if (state.map.getSource(id)) state.map.removeSource(id);
    });
    state.orthoLayerIds = [];
    state.lowestRasterLayerId = 'defects-layer';
}

export function addOrthomosaicShingle(r, isFrontVisible, isRearVisible) {
    if (!state.mapLoaded || !state.map) return;
    ['front', 'rear'].forEach(view => {
        if (r.views[view] && r.views[view].footprint && r.views[view].footprint.corners) {
            const rawBevUrl = r.views[view].raw_bev_url;
            const corners = r.views[view].footprint.corners;
            const sourceId = 'ortho-' + r.filename + '-' + view;
            
            if (!state.map.getSource(sourceId)) {
                state.map.addSource(sourceId, { type: 'image', url: rawBevUrl, coordinates: corners });
                const isVisible = view === 'front' ? isFrontVisible : isRearVisible;
                
                state.map.addLayer({
                    id: sourceId,
                    type: 'raster',
                    source: sourceId,
                    layout: { 'visibility': isVisible ? 'visible' : 'none' },
                    paint: { 'raster-opacity': 1.0, 'raster-fade-duration': 0 }
                }, state.lowestRasterLayerId);
                
                state.orthoLayerIds.push(sourceId);
                state.lowestRasterLayerId = sourceId; 
            }
        }
    });
}