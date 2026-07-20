export const state = {
    modelFile: null,
    isModelLoaded: false,
    imageFiles: [],
    appIs360: true,
    
    map: null,
    mapLoaded: false,
    mapPopup: null,
    isMapVisible: true,
    
    fullGeojson: { type: "FeatureCollection", features: [] },
    nodesGeoJson: { type: "FeatureCollection", features: [] },
    trailGeoJson: { type: "FeatureCollection", features: [] },
    passPairsGeoJson: { type: "FeatureCollection", features: [] },
    
    // Ortho (BEV) frame layers are always inserted directly beneath the
    // fixed 'defects-layer' id in map.js, so the newest-added frame is
    // always rendered on top of older ones. No rolling "lowest layer"
    // pointer is needed for this any more -- see addOrthomosaicShingle.
    orthoLayerIds: [], 

    fullResults: [],
    appResults: [], 
    currentIndex: 0,
    currentDirection: 'front', 
    activeMarkerFilename: null, 

    stateLastLat: null,
    stateLastLon: null,
    stateLastLocId: 1,
    
    currentTaskId: null,
    warningCount: 0,
    
    layoutPrefs: {
        mapOn: { mainW: "45%", mediaBasis: "55%", isManual: false },
        mapOff: { mediaBasis: "50%", isManual: false }
    }
};
