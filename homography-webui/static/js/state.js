export const state = {
    modelFile: null,
    isModelLoaded: false,
    imageFiles: [],
    appIs360: true,
    
    map: null,
    mapLoaded: false,
    mapPopup: null,
    
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
    stateLastLocId: 1
};