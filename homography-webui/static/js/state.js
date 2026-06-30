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
    
    orthoLayerIds: [], 
    lowestRasterLayerId: 'defects-layer', 

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