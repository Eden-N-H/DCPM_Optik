import math
import os
import json

def atomic_write_json(filepath, data, **kwargs):
    """
    Safely writes JSON data to disk by writing to a temporary file 
    first and performing an atomic rename, preventing file corruption 
    if a concurrent read or crash occurs mid-write.
    """
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, **kwargs)
    os.replace(tmp_path, filepath)

def safe_float(value, default=None):
    try:
        if value is None or str(value).strip() == "": return default
        return float(value)
    except (TypeError, ValueError): return default

def sanitize_meta(obj):
    if isinstance(obj, dict): return {str(k): sanitize_meta(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [sanitize_meta(v) for v in obj]
    elif isinstance(obj, tuple): return [sanitize_meta(v) for v in obj]
    elif isinstance(obj, bytes):
        if len(obj) > 1024: return f"<binary data: {len(obj)} bytes>"
        try: return obj.decode('utf-8', errors='ignore')
        except: return f"<binary data: {len(obj)} bytes>"
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
        return obj
    elif hasattr(obj, 'tolist') and callable(obj.tolist): return sanitize_meta(obj.tolist())
    elif hasattr(obj, 'item') and callable(obj.item): return sanitize_meta(obj.item())
    elif hasattr(obj, 'printable'): return str(obj.printable)
    return obj
