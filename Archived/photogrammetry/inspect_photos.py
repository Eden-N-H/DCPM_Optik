import os
import pathlib
import exifread

def check_photo_properties():
    base_dir = pathlib.Path(__file__).parent.resolve()
    image_dir = base_dir / "input_images"
    
    if not image_dir.exists() or not os.listdir(image_dir):
        print("❌ Error: 'clean_images' folder is empty or missing.")
        return

    # Grab the first image file to inspect
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))])
    
    if not image_files:
        print("❌ Error: No compatible images found.")
        return
        
    target_image = image_dir / image_files[0]
    print(f"🔍 Inspecting metadata properties for: {target_image.name}\n" + "="*50)

    with open(target_image, 'rb') as f:
        tags = exifread.process_file(f)
        
    if not tags:
        print("⚠️ Warning: No EXIF metadata tags found in this image file at all.")
        print("The image might have been stripped of its properties (common when sharing via WhatsApp/Slack).")
        return

    # --- 1. Check for Basic Hardware Context ---
    make = tags.get('Image Make', 'Unknown')
    model = tags.get('Image Model', 'Unknown')
    print(f"📱 Camera Hardware: {make} {model}")

    # --- 2. Check for GPS Spatial Priors (For Geographic Position) ---
    has_gps = any('GPS' in key for key in tags.keys())
    if has_gps:
        print("✅ GPS Data: FOUND (Your model can be absolute-positioned on Earth)")
        # Print a few example GPS tags if they exist
        for key in ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPS GPSAltitude']:
            if key in tags:
                print(f"   ↳ {key}: {tags[key]}")
    else:
        print("❌ GPS Data: MISSING")

    # --- 3. Check for Lens Intrinsics (For Focal Length & Distortions) ---
    focal_length = tags.get('EXIF FocalLength', None)
    focal_35mm = tags.get('EXIF FocalLengthIn35mmFilm', None)
    if focal_length:
        print(f"✅ Lens Intrinsics: FOUND (Physical Focal Length: {focal_length}mm)")
        if focal_35mm:
            print(f"   ↳ Equivalent 35mm Focal Length: {focal_35mm}mm")
    else:
        print("❌ Lens Intrinsics: MISSING (The engine will have to guess the lens type)")

    # --- 4. Check for Apple-Specific Gravity/Motion Priors ---
    # Apple encodes pitch/roll and gravity vectors into MakerNotes.
    # Exifread will dump these as raw hex blocks, but we can check if they exist.
    has_maker_notes = 'EXIF MakerNote' in tags
    if has_maker_notes and "apple" in str(make).lower():
        print("✅ Apple MakerNotes: FOUND (Likely contains Gravity Vector/CoreMotion metadata)")
    else:
        print("ℹ️ Apple MakerNotes: NOT FOUND (Standard EXIF headers only)")

    print("="*50)
    print("💡 Quick Tip: If you want to see absolutely every raw metadata tag inside the file,")
    print("uncomment the print loop at the bottom of this script!")
    
    # Optional: Uncomment the lines below to see everything hidden inside the photo:
    # for tag in tags.keys():
    #     print(f"{tag}: {tags[tag]}")

if __name__ == "__main__":
    check_photo_properties()