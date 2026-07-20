import os
import json
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.affinity import translate
from geo_math import haversine_distance, calculate_bearing

def project_to_local(lat, lon, base_lat, base_lon):
    dist = haversine_distance(base_lat, base_lon, lat, lon)
    bearing = calculate_bearing(base_lat, base_lon, lat, lon)
    angle = math.radians(bearing)
    return dist * math.sin(angle), dist * math.cos(angle)

def local_to_projected(x, y, base_lat, base_lon):
    R = 6378137.0
    d = math.hypot(x, y)
    bearing = math.atan2(x, y)
    lat_rad = math.radians(base_lat)
    lon_rad = math.radians(base_lon)
    out_lat = math.asin(math.sin(lat_rad)*math.cos(d/R) + math.cos(lat_rad)*math.sin(d/R)*math.cos(bearing))
    out_lon = lon_rad + math.atan2(math.sin(bearing)*math.sin(d/R)*math.cos(lat_rad), math.cos(d/R) - math.sin(lat_rad)*math.sin(out_lat))
    return math.degrees(out_lat), math.degrees(out_lon)

def shape_similarity(poly1, poly2):
    """
    Evaluates how similar two mask shapes are in dimensions and orientation,
    ignoring their absolute positions by centering them both at the origin.
    """
    try:
        c1 = poly1.centroid
        c2 = poly2.centroid
        p1_c = translate(poly1, xoff=-c1.x, yoff=-c1.y)
        p2_c = translate(poly2, xoff=-c2.x, yoff=-c2.y)
        union_area = p1_c.union(p2_c).area
        if union_area == 0:
            return 0
        return p1_c.intersection(p2_c).area / union_area
    except Exception:
        return 0

def group_defects(results, upload_folder):
    """
    Analyzes project-wide detections to find identical physical defects.
    Groups overlapping defects of the same instance (picking the best/closest frame)
    and stitches continuous longitudinal defects (unioning their polygons).
    """
    valid = [r for r in results if r.get('lat') is not None]
    if not valid: return results
    base_lat, base_lon = valid[0]['lat'], valid[0]['lon']
    
    detections = []
    for r_idx, r in enumerate(results):
        for feat in r.get('geojson', []):
            if feat['geometry']['type'] == 'Polygon':
                coords = feat['geometry']['coordinates'][0]
                local_coords = [project_to_local(lat, lon, base_lat, base_lon) for lon, lat in coords]
                poly = Polygon(local_coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                detections.append({
                    'r_idx': r_idx,
                    'filename': r['filename'],
                    'original_name': r['original_name'],
                    'view': feat['properties']['view'],
                    'det_idx': feat['properties'].get('det_idx', 0),
                    'class': feat['properties']['class'],
                    'color': feat['properties']['color'],
                    'conf': feat['properties']['conf'],
                    'poly': poly,
                    'feat': feat,
                    'lat': r['lat'],
                    'lon': r['lon']
                })
                
    by_class = {}
    for d in detections:
        by_class.setdefault(d['class'], []).append(d)
        
    for cls_name, items in by_class.items():
        n = len(items)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        # 1. Cluster nearby/related detections
        for i in range(n):
            for j in range(i+1, n):
                item1 = items[i]
                item2 = items[j]
                
                # Only consider defects in relatively close temporal proximity (same pass)
                if abs(item1['r_idx'] - item2['r_idx']) > 10:
                    continue
                    
                dist = item1['poly'].distance(item2['poly'])
                
                # Link if they are physically touching/continuous, OR if they 
                # are slightly offset due to drift but have a highly similar shape (duplicates).
                if dist < 0.2:
                    union(i, j)
                elif dist < 2.0 and shape_similarity(item1['poly'], item2['poly']) > 0.5:
                    union(i, j)
                    
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(items[i])
            
        for g_id, group in groups.items():
            if len(group) == 1:
                continue
                
            # Pre-calculate distance to camera (in projected local space) for quality resolution
            for item in group:
                c_x, c_y = project_to_local(item['lat'], item['lon'], base_lat, base_lon)
                item['dist_to_cam'] = item['poly'].centroid.distance(Point(c_x, c_y))
                
            # 2. Iteratively collapse duplicates inside the group
            active_items = list(group)
            while True:
                found_dup = False
                for i in range(len(active_items)):
                    for j in range(i+1, len(active_items)):
                        item1 = active_items[i]
                        item2 = active_items[j]
                        dist = item1['poly'].distance(item2['poly'])
                        
                        if dist < 2.0 and shape_similarity(item1['poly'], item2['poly']) > 0.5:
                            # We found a duplicate pair. Drop the one further from the camera.
                            if item1['dist_to_cam'] > item2['dist_to_cam']:
                                active_items.pop(i)
                            else:
                                active_items.pop(j)
                            found_dup = True
                            break
                    if found_dup:
                        break
                if not found_dup:
                    break
                    
            # 3. Resolve the finalized group
            if len(active_items) == 1:
                # Group collapsed down to a single physical defect
                best_item = active_items[0]
                
                for item in group:
                    if item != best_item:
                        _mark_detection_hidden(upload_folder, item['filename'], item['view'], item['det_idx'])
                        
                _update_detection_meta(upload_folder, best_item['filename'], best_item['view'], best_item['det_idx'], {
                    "is_grouped": True,
                    "spanned_frames": [x['original_name'] for x in group]
                })
                
            else:
                # Multiple distinct connected segments remain (Multi-frame spanning defect)
                # Stitch them exactly as they are.
                polys_to_merge = [x['poly'].buffer(1e-5) for x in active_items]
                merged_poly = unary_union(polys_to_merge).buffer(-1e-5)
                
                # Resolve isolated artifacts if union produces disconnected chunks
                if merged_poly.geom_type == 'GeometryCollection':
                    polys = [geom for geom in merged_poly.geoms if geom.geom_type in ('Polygon', 'MultiPolygon')]
                    if polys:
                        merged_poly = max(polys, key=lambda a: a.area)
                        
                if merged_poly.geom_type == 'MultiPolygon':
                    merged_poly = max(merged_poly.geoms, key=lambda a: a.area)
                    
                # Bind the stitched polygon to the earliest seen frame in the segment
                best_item = min(active_items, key=lambda x: x['r_idx'])
                
                for item in group:
                    if item != best_item:
                        _mark_detection_hidden(upload_folder, item['filename'], item['view'], item['det_idx'])
                        
                if hasattr(merged_poly, 'exterior'):
                    exterior = merged_poly.exterior.coords
                else:
                    exterior = active_items[0]['poly'].exterior.coords
                    
                geo_coords = [[local_to_projected(x, y, base_lat, base_lon)[1], local_to_projected(x, y, base_lat, base_lon)[0]] for x, y in exterior]
                
                _update_detection_meta(upload_folder, best_item['filename'], best_item['view'], best_item['det_idx'], {
                    "is_stitched": True,
                    "spanned_frames": [x['original_name'] for x in group],
                    "world_polygon": geo_coords,
                    "area_sqm": merged_poly.area
                })

def _mark_detection_hidden(upload_folder, filename, view, det_idx):
    meta_path = os.path.join(upload_folder, f"process_meta_{filename}.json")
    if not os.path.exists(meta_path): return
    with open(meta_path, 'r') as f: meta = json.load(f)
    if det_idx < len(meta['view_meta'][view]['detections']):
        meta['view_meta'][view]['detections'][det_idx]['hidden'] = True
    with open(meta_path, 'w') as f: json.dump(meta, f)

def _update_detection_meta(upload_folder, filename, view, det_idx, updates):
    meta_path = os.path.join(upload_folder, f"process_meta_{filename}.json")
    if not os.path.exists(meta_path): return
    with open(meta_path, 'r') as f: meta = json.load(f)
    if det_idx < len(meta['view_meta'][view]['detections']):
        meta['view_meta'][view]['detections'][det_idx].update(updates)
    with open(meta_path, 'w') as f: json.dump(meta, f)
