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
            state.map.addSource('pass-pairs-source', { type: 'geojson', data: state.passPairsGeoJson });

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

            // Repeat-pass diagnostic overlay: connects each auto-matched pair
            // of frames (same physical spot, different pass) with a bright
            // magenta dashed line, so lateral/longitudinal drift between
            // passes is visible directly on the map rather than only in a
            // table of numbers.
            state.map.addLayer({
                id: 'pass-pairs-layer', type: 'line', source: 'pass-pairs-source',
                layout: { 'visibility': 'none' },
                paint: { 'line-color': '#ff00ff', 'line-width': 2.5, 'line-dasharray': [1, 1.5] }
            });

            state.map.addLayer({
                id: 'pass-pairs-points-layer', type: 'circle', source: 'pass-pairs-source',
                filter: ['==', ['geometry-type'], 'Point'],
                layout: { 'visibility': 'none' },
                paint: { 'circle-radius': 4, 'circle-color': '#ff00ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#000000' }
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

            state.map.on('mouseenter', 'pass-pairs-layer', (e) => {
                state.map.getCanvas().style.cursor = 'pointer';
                const f = e.features[0];
                const p = f.properties;
                state.mapPopup.setLngLat(e.lngLat).setHTML(`<div class="text-xs"><b>Repeat-Pass Drift</b><br>Lateral: ${p.corrected_lateral_m} m<br>Longitudinal: ${p.corrected_longitudinal_m} m<br>ΔHeading: ${p.delta_heading}°</div>`).addTo(state.map);
            });
            state.map.on('mouseleave', 'pass-pairs-layer', () => {
                state.map.getCanvas().style.cursor = '';
                state.mapPopup.remove();
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
}

export function addOrthomosaicShingle(r, isFrontVisible, isRearVisible) {
    if (!state.mapLoaded || !state.map) return;
    ['front', 'rear'].forEach(view => {
        if (r.views[view] && r.views[view].footprint && r.views[view].footprint.corners) {
            const ts = Date.now();
            const rawBevUrl = r.views[view].raw_bev_url + `?t=${ts}`;
            const corners = r.views[view].footprint.corners;
            const sourceId = 'ortho-' + r.filename + '-' + view;
            
            if (!state.map.getSource(sourceId)) {
                state.map.addSource(sourceId, { type: 'image', url: rawBevUrl, coordinates: corners });
                const isVisible = view === 'front' ? isFrontVisible : isRearVisible;
                
                // Anchor every new frame's imagery layer directly beneath the
                // fixed 'defects-layer' id (rather than beneath the *previous*
                // ortho layer, as before). Frames are always added here in
                // sequential capture order, so inserting each new one just
                // under 'defects-layer' stacks it ON TOP of every previously
                // added frame (while still staying under defect overlays).
                //
                // Previously this used a rolling "lowestRasterLayerId" that
                // pointed at the last-added frame, so each new frame was
                // inserted BELOW it -- meaning the very first frame captured
                // in a sequence stayed on top of literally everything after
                // it, permanently. Because each frame's BEV footprint is
                // rotated to its own capture-time heading and heavily
                // overlaps its neighbours (footprint depth > capture
                // interval), that one stale, top-most rectangle's straight
                // edge would cut diagonally across newer, differently
                // oriented frames underneath -- especially through corners --
                // producing the visible staircase/"stepping" artifact along
                // the corridor. Newest-on-top ordering fixes this at the
                // source: the freshest, most spatially-correct frame is
                // always what's visible, with no smoothing/blurring needed.
                state.map.addLayer({
                    id: sourceId,
                    type: 'raster',
                    source: sourceId,
                    layout: { 'visibility': isVisible ? 'visible' : 'none' },
                    paint: { 'raster-opacity': 1.0, 'raster-fade-duration': 0 }
                }, 'defects-layer');
                
                state.orthoLayerIds.push(sourceId);
            }
        }
    });
}

export function setPassPairsVisible(visible) {
    if (!state.mapLoaded || !state.map) return;
    const vis = visible ? 'visible' : 'none';
    if (state.map.getLayer('pass-pairs-layer')) state.map.setLayoutProperty('pass-pairs-layer', 'visibility', vis);
    if (state.map.getLayer('pass-pairs-points-layer')) state.map.setLayoutProperty('pass-pairs-points-layer', 'visibility', vis);
}

export function setPassPairsData(pairs) {
    const features = [];
    pairs.forEach((p, i) => {
        features.push({
            type: "Feature",
            properties: {
                pair_index: i,
                corrected_lateral_m: p.corrected_lateral_m,
                corrected_longitudinal_m: p.corrected_longitudinal_m,
                delta_heading: p.delta_heading
            },
            geometry: { type: "LineString", coordinates: [[p.lon_a, p.lat_a], [p.lon_b, p.lat_b]] }
        });
    });
    state.passPairsGeoJson = { type: "FeatureCollection", features };
    updateMapSource('pass-pairs-source', state.passPairsGeoJson);
    setPassPairsVisible(true);
}

export function clearPassPairsData() {
    state.passPairsGeoJson = { type: "FeatureCollection", features: [] };
    updateMapSource('pass-pairs-source', state.passPairsGeoJson);
    setPassPairsVisible(false);
}
