import os
from PIL import Image, ImageDraw
import pathlib

def main():
    base_dir = pathlib.Path(__file__).parent.resolve()
    input_dir = base_dir / "input_images"
    output_dir = base_dir / "clean_images"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"❌ Error: Cannot find the 'input_images' folder.")
        return

    # Find all common image formats
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.JPG"))
    
    if not images:
        print(f"❌ Error: No images found inside: {input_dir}")
        return

    print(f"🧹 Masking {len(images)} images (keeping original resolution)...")

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Create a drawing tool
                draw = ImageDraw.Draw(img)
                
                # Draw a pure black box over the top 15% (hides the orange text)
                draw.rectangle([(0, 0), (width, int(height * 0.15))], fill="black")
                
                # Draw a pure black box over the bottom 15% (hides the car hood)
                draw.rectangle([(0, int(height * 0.85)), (width, height)], fill="black")
                
                # Save the file (Resolution stays exactly the same!)
                img.save(output_dir / img_path.name)
        except Exception as e:
            print(f"⚠️ Could not process {img_path.name}: {e}")

    print(f"✅ Masked images saved in the 'clean_images' folder.")

if __name__ == "__main__":
    main()
