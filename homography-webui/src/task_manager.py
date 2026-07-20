import os
import json
import uuid
import threading
import queue

from constants import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT
from geo_math import calculate_bearing, haversine_distance, apply_camera_offset
from pipeline_image import process_single_image
from pipeline_video import process_video_frames_async, get_video_frame_metadata

active_tasks = {}
cancel_flags = {}

def start_processing_job(image_data, options, last_lat, last_lon, loc_id, upload_folder, global_model, model_lock, sam2_predictor=None):
    image_data = sorted(image_data, key=lambda x: x['filename'])
    trail_coordinates = []
    initial_ui_state = []
    has_video = any(a['ext'] in ALLOWED_VIDEO_EXT for a in image_data)

    cam_off_fwd = options.get('cam_offset_forward_m', 0.0) or 0.0
    cam_off_right = options.get('cam_offset_right_m', 0.0) or 0.0

    for i in range(len(image_data)):
        if image_data[i]['ext'] in ALLOWED_IMAGE_EXT:
            lat, lon = image_data[i]['lat'], image_data[i]['lon']

            # Centered-difference heading: previously this was a pure forward
            # difference (bearing from i to i+1 only), which systematically
            # "looks ahead" through corners and produces a heading bias whose
            # magnitude depends on the spacing between captured photos --
            # spacing that isn't repeatable between sessions. Using the
            # midpoint bearing (i-1 -> i+1) removes that directional bias for
            # all interior points; only the first/last photo in a sequence
            # fall back to a one-sided difference.
            prev_valid = i > 0 and image_data[i-1].get('lat') is not None and lat is not None
            next_valid = i < len(image_data) - 1 and image_data[i+1].get('lat') is not None and lat is not None

            if prev_valid and next_valid:
                heading = calculate_bearing(image_data[i-1]['lat'], image_data[i-1]['lon'], image_data[i+1]['lat'], image_data[i+1]['lon'])
            elif next_valid:
                heading = calculate_bearing(lat, lon, image_data[i+1]['lat'], image_data[i+1]['lon'])
            elif prev_valid:
                heading = calculate_bearing(image_data[i-1]['lat'], image_data[i-1]['lon'], lat, lon)
            else:
                heading = image_data[i-1].get('heading', 0.0) if i > 0 else 0.0

            image_data[i]['heading'] = heading

            if lat is not None and lon is not None:
                # Shift the raw GPS antenna fix to the true camera position
                # before it's used for trail plotting, location clustering,
                # or (downstream, in pipeline_image) defect world coordinates.
                if cam_off_fwd or cam_off_right:
                    image_data[i]['raw_lat'], image_data[i]['raw_lon'] = lat, lon
                    lat, lon = apply_camera_offset(lat, lon, heading, cam_off_right, cam_off_fwd)
                    image_data[i]['lat'], image_data[i]['lon'] = lat, lon

                trail_coordinates.append([lon, lat])
                if last_lat is not None and last_lon is not None:
                    dist = haversine_distance(last_lat, last_lon, lat, lon)
                    if dist > 50.0: loc_id += 1
                last_lat, last_lon = lat, lon
            initial_ui_state.append(image_data[i])
        elif image_data[i]['ext'] in ALLOWED_VIDEO_EXT:
            video_frames = get_video_frame_metadata(image_data[i]['path'], options, image_data[i]['original_name'])
            for vf in video_frames:
                initial_ui_state.append(vf)
                if vf.get('lat') is not None and vf.get('lon') is not None:
                    trail_coordinates.append([vf['lon'], vf['lat']])
                    last_lat, last_lon = vf['lat'], vf['lon']
        image_data[i]['location'] = f"Location {loc_id}"

    task_id = str(uuid.uuid4())
    active_tasks[task_id] = queue.Queue()
    cancel_flags[task_id] = False
    total_est_frames = len(initial_ui_state)

    def process_worker(assets, t_id, worker_options):
        try:
            def is_cancelled():
                return cancel_flags.get(t_id, False)

            for asset in assets:
                if is_cancelled():
                    active_tasks[t_id].put({"type": "cancelled", "message": "Job cancelled by user."})
                    break

                def on_frame_processed(payload):
                    if "error" in payload:
                        active_tasks[t_id].put({"type": "item_error", "original_name": payload.get("original_name", asset['original_name']), "message": payload["error"], "is_video": payload.get("is_video", False)})
                    elif payload.get("type") == "health_report":
                        active_tasks[t_id].put({"type": "health_report", "original_name": payload.get("original_name"), "data": payload["data"]})
                    elif payload.get("type") == "cancelled":
                        active_tasks[t_id].put({"type": "cancelled", "message": "Job cancelled by user during video processing."})
                    else:
                        active_tasks[t_id].put({"type": "update", "data": payload})
                    
                if asset['ext'] in ALLOWED_VIDEO_EXT:
                    process_video_frames_async(
                        asset['path'], global_model, upload_folder, 
                        asset['filename'], asset['original_name'], asset['location'], 
                        worker_options, model_lock, on_frame_processed, is_cancelled,
                        sam2_predictor=sam2_predictor
                    )
                    if is_cancelled(): break
                else:
                    try:
                        telemetry = {
                            "lat": asset['lat'],
                            "lon": asset['lon'],
                            "raw_lat": asset.get('raw_lat', asset['lat']),
                            "raw_lon": asset.get('raw_lon', asset['lon']),
                            "heading": asset.get('heading', 0.0),
                            "grav_vec": asset.get('grav_vec'),
                            "klns": asset.get('klns'),
                            "xfov": asset.get('xfov'),
                            "yfov": asset.get('yfov'),
                            "pitch": asset.get('pitch'),
                            "roll": asset.get('roll'),
                            "cam_offset_forward_m": cam_off_fwd,
                            "cam_offset_right_m": cam_off_right
                        }

                        defects, geo_feats, gen_files, footprints, view_meta, calibrations = process_single_image(
                            asset['path'], global_model, asset['filename'], upload_folder, 
                            telemetry, worker_options, model_lock, asset['original_name'],
                            sam2_predictor=sam2_predictor
                        )
                        
                        process_meta_data = {
                            "telemetry": telemetry,
                            "options": worker_options,
                            "view_meta": view_meta,
                            "original_name": asset['original_name']
                        }
                        with open(os.path.join(upload_folder, f"process_meta_{asset['filename']}.json"), 'w') as f:
                            json.dump(process_meta_data, f)
                        
                        # Preserve full precision floats to eliminate map rendering quantisation steps
                        result_payload = {
                            "original_name": asset['original_name'], "filename": asset['filename'],
                            "lat": asset['lat'] if asset['lat'] is not None else None, 
                            "lon": asset['lon'] if asset['lon'] is not None else None,
                            "pitch": asset.get('pitch') if asset.get('pitch') is not None else None,
                            "roll": asset.get('roll') if asset.get('roll') is not None else None,
                            "location": asset['location'], "geojson": geo_feats, "views": {}
                        }
                        
                        for view in (['front', 'rear'] if worker_options.get('is_360', True) else ['front']):
                            gf = gen_files[view]
                            result_payload["views"][view] = {
                                "calibration": calibrations[view],
                                "raw_filename": gf["raw_rect"], "raw_bev_filename": gf["raw_bev"],
                                "raw_bev_url": f"/static/uploads/{gf['raw_bev']}", "rect_url": f"/static/uploads/{gf['rect']}",
                                "bev_url": f"/static/uploads/{gf['bev']}",
                                "edit_bev_url": f"/static/uploads/{gf.get('edit_bev', gf['raw_bev'])}",
                                "defects": defects[view], "footprint": footprints[view]
                            }
                        active_tasks[t_id].put({"type": "update", "data": result_payload})
                    except Exception as e:
                        active_tasks[t_id].put({"type": "item_error", "original_name": asset['original_name'], "message": str(e), "is_video": False})
            
            if not is_cancelled():
                active_tasks[t_id].put({"type": "complete"})
        except Exception as e:
            active_tasks[t_id].put({"type": "error", "message": str(e)})

    threading.Thread(target=process_worker, args=(image_data, task_id, options)).start()

    initial_geojson = []
    if len(trail_coordinates) > 1:
        initial_geojson.append({
            "type": "Feature", "properties": {"type": "trail"},
            "geometry": {"type": "LineString", "coordinates": trail_coordinates}
        })

    return {
        "success": True, "task_id": task_id, "total_images": total_est_frames,
        "has_video": has_video, "initial_state": initial_ui_state,
        "initial_trail": {"type": "FeatureCollection", "features": initial_geojson},
        "last_lat": last_lat, "last_lon": last_lon, "last_loc_id": loc_id
    }
