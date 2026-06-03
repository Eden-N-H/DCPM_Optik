import exiftool

def extract_gopro_metadata(image_path=r"C:\OPTIK\ExtractEXIF\Images\Chichester Dam Road - G0052068.JPG"):
    """
    Extracts explicit EXIF metadata and utilizes Composite calculated tags
    as fallbacks to successfully read GoPro spatial telemetry streams.
    """
    # Tags required to fetch or compute the exact composite data layout
    target_tags = [
        "EXIF:ApertureValue",
        "EXIF:FNumber",
        "EXIF:ExifImageWidth",
        "EXIF:ExifImageHeight",
        "Composite:Megapixels",
        "Composite:ScaleFactor35efl",
        "EXIF:ExposureTime",
        "EXIF:CreateDate",
        "EXIF:DateTimeOriginal",
        "EXIF:ModifyDate",
        "GPS:GPSAltitude",
        "GPS:GPSDateStamp",
        "GPS:GPSTimeStamp",
        "GPS:GPSLatitude",
        "GPS:GPSLongitude",
        "Composite:CircleOfConfusion",
        "Composite:FOV",
        "EXIF:FocalLength",
        "Composite:GPSPosition",
        "Composite:HyperfocalDistance",
        "Composite:LightValue",
        # --- GOPRO SPECIFIC TELEMETRY STREAM FALLBACKS ---
        "Composite:GPSAltitude",
        "Composite:GPSDateTime"
    ]

    # Absolute path to your local ExifTool executable
    exe_path = r"C:\OPTIK\ExtractEXIF\exiftool-13.59_32\exiftool.exe"

    with exiftool.ExifToolHelper(executable=exe_path) as et:
        try:
            # Fetch metadata dictionary for the image
            metadata_list = et.get_tags(image_path, tags=target_tags)
            metadata = metadata_list[0] if metadata_list else {}
            
            print("---- Composite & Extracted Metadata ----\n")
            
            # 1. Aperture
            aperture = metadata.get("EXIF:FNumber") or metadata.get("EXIF:ApertureValue")
            print(f"Aperture                        : {aperture}")
            
            # 2. Image Size
            w = metadata.get("EXIF:ExifImageWidth")
            h = metadata.get("EXIF:ExifImageHeight")
            print(f"Image Size                      : {w} {h}")
            
            # 3. Megapixels
            mp = metadata.get("Composite:Megapixels")
            print(f"Megapixels                      : {mp}")
            
            # 4. Scale Factor To 35 mm Equivalent
            scale = metadata.get("Composite:ScaleFactor35efl")
            print(f"Scale Factor To 35 mm Equivalent: {scale}")
            
            # 5. Shutter Speed
            shutter = metadata.get("EXIF:ExposureTime")
            print(f"Shutter Speed                   : {shutter}")
            
            # 6. Timestamps
            print(f"Create Date                     : {metadata.get('EXIF:CreateDate')}")
            print(f"Date/Time Original              : {metadata.get('EXIF:DateTimeOriginal')}")
            print(f"Modify Date                     : {metadata.get('EXIF:ModifyDate')}")
            
            # 7. GPS Data Extraction with Composite Fallbacks
            lat = metadata.get("GPS:GPSLatitude")
            lon = metadata.get("GPS:GPSLongitude")
            
            # Coordinate Fallback: Parse Composite:GPSPosition if standalone tags are empty
            if lat is None or lon is None:
                composite_pos = metadata.get("Composite:GPSPosition")
                if composite_pos:
                    try:
                        coords = str(composite_pos).split()
                        if len(coords) >= 2:
                            lat = coords[0]
                            lon = coords[1]
                    except Exception:
                        pass

            # Altitude Fallback: Extract calculated composite values if explicit tag returns None
            alt = metadata.get("GPS:GPSAltitude") or metadata.get("Composite:GPSAltitude") or "N/A"
            print(f"GPS Altitude                    : {alt}")
            
            # Date/Time Fallback: Extract from composite stream if explicit stamps return None
            gps_date = metadata.get("GPS:GPSDateStamp")
            gps_time = metadata.get("GPS:GPSTimeStamp")
            
            if gps_date and gps_time:
                gps_datetime = f"{gps_date} {gps_time}"
            else:
                gps_datetime = metadata.get("Composite:GPSDateTime") or "N/A"
                
            print(f"GPS Date/Time                   : {gps_datetime}")
            print(f"GPS Latitude                    : {lat}")
            print(f"GPS Longitude                   : {lon}")
            
            # 8. Optics and Distance Calculations
            print(f"Circle Of Confusion             : {metadata.get('Composite:CircleOfConfusion')}")
            print(f"Field Of View                   : {metadata.get('Composite:FOV')}")
            print(f"Focal Length                    : {metadata.get('EXIF:FocalLength')}")
            print(f"GPS Position                    : {metadata.get('Composite:GPSPosition')}")
            print(f"Hyperfocal Distance             : {metadata.get('Composite:HyperfocalDistance')}")
            print(f"Light Value                     : {metadata.get('Composite:LightValue')}")

        except Exception as e:
            print(f"An error occurred while reading metadata: {e}")

# Example Usage
if __name__ == "__main__":
    sample_image = r"C:\OPTIK\ExtractEXIF\Images\Allyn River Road - G0160129.JPG"
    extract_gopro_metadata(sample_image)