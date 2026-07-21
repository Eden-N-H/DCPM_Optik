import os
import json
import zipfile
from io import BytesIO
from werkzeug.utils import secure_filename
from constants import ALLOWED_IMAGE_EXT

def create_raw_zip(project_data, upload_folder):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            safe_orig = secure_filename(r['original_name'])
            if not safe_orig.lower().endswith(tuple(ALLOWED_IMAGE_EXT)): safe_orig += ".jpg"
            base_orig = os.path.splitext(safe_orig)[0]
            for view in r['views'].keys():
                file_path = os.path.join(upload_folder, r['views'][view]['raw_filename'])
                if os.path.exists(file_path): zf.write(file_path, f"{loc}/{view}/RAW_{safe_orig}")
                meta_path = os.path.join(upload_folder, f"meta_{r['filename']}.json")
                if os.path.exists(meta_path): zf.write(meta_path, f"{loc}/{view}/RAW_{base_orig}.json")
    memory_file.seek(0)
    return memory_file

def create_flat_zip(project_data, upload_folder):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            safe_orig = secure_filename(r['original_name'])
            if not safe_orig.lower().endswith(tuple(ALLOWED_IMAGE_EXT)): safe_orig += ".jpg"
            base_orig = os.path.splitext(safe_orig)[0]
            for view in r['views'].keys():
                file_path = os.path.join(upload_folder, r['views'][view].get('raw_bev_filename', ''))
                if os.path.exists(file_path): zf.write(file_path, f"{loc}/{view}/FLAT_{safe_orig}")
                meta_path = os.path.join(upload_folder, f"meta_{r['filename']}.json")
                if os.path.exists(meta_path): zf.write(meta_path, f"{loc}/{view}/FLAT_{base_orig}.json")
    memory_file.seek(0)
    return memory_file

def create_project_zip(project_state, upload_folder):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Write the overarching state JSON to the root of the archive
        zf.writestr('project_state.json', json.dumps(project_state))
        
        # 2. Extract base names for substring matching
        base_names = {os.path.splitext(r['filename'])[0] for r in project_state.get('results', [])}
        
        # 3. Scan the working directory and pull in any file that belongs to these frames
        # (Catches source_, meta_, process_meta_, rect_, bev_, raw_bev_, corridor_, etc.)
        for f in os.listdir(upload_folder):
            filepath = os.path.join(upload_folder, f)
            if not os.path.isfile(filepath): 
                continue
                
            for b in base_names:
                if b in f:
                    zf.write(filepath, f"data/{f}")
                    break
                    
    memory_file.seek(0)
    return memory_file

