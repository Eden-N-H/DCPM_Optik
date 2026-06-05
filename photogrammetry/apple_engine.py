import os
import subprocess
import pathlib
import sys
import shutil

def run_apple_gopro_engine():
    print("🍎 Initializing Apple Object Capture Engine for GoPro Hardware...")
    
    base_dir = pathlib.Path(__file__).parent.resolve()
    image_dir = base_dir / "input_images"
    output_dir = base_dir / "mac_solid_model"
    swift_script_path = base_dir / "temp_mac_engine.swift"
    
    output_model_file = output_dir / "road_model.usdz"

    if not image_dir.exists() or not os.listdir(image_dir):
        print("❌ Error: 'clean_images' folder is empty or missing.")
        sys.exit(1)

    print(f"📸 Dataset detected from GoPro HERO13 Black. Optimizing pipeline...")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Advanced Swift Engine Setup
    # Notice the optimization changes below: 
    # 1. featureSensitivity = .high (essential for fine asphalt/road surfaces)
    # 2. sampleOrdering = .sequential (tells the math you walked/drove down a path)
    swift_code = """
import Foundation
import RealityKit

let args = CommandLine.arguments
let inputFolder = URL(fileURLWithPath: args[1], isDirectory: true)
let outputFile = URL(fileURLWithPath: args[2], isDirectory: false)

guard PhotogrammetrySession.isSupported else {
    print("❌ Error: Photogrammetry is not supported on this Mac hardware configuration.")
    exit(1)
}

Task {
    do {
        var configuration = PhotogrammetrySession.Configuration()
        
        // 🛠️ CRITICAL GOPRO & ROAD OPTIMIZATIONS
        configuration.featureSensitivity = .high        // Maximize tracking on fine asphalt details
        configuration.sampleOrdering = .sequential      // Optimize for continuous camera movement down the road
        configuration.isObjectMaskingEnabled = false    // Force the engine to keep the entire environment/road surface
        
        print("🔮 Launching multi-view spatial reconstruction engine...")
        let session = try PhotogrammetrySession(input: inputFolder, configuration: configuration)
        
        // Request a medium detail model (good balance of performance and centimeter accuracy)
        let request = PhotogrammetrySession.Request.modelFile(url: outputFile, detail: .medium)

        try session.process(requests: [request])

        for try await output in session.outputs {
            switch output {
            case .processingComplete:
                print("\\n✅ SUCCESS! 3D Survey Mesh Generated Successfully!")
                exit(0)
            case .requestProgress(_, let fractionComplete):
                let progress = String(format: "%.0f", fractionComplete * 100)
                print("⏳ Triangulating spatial geometry: \\(progress)%", terminator: "\\r")
                fflush(stdout)
            case .requestError(_, let error):
                print("\\n❌ Reconstructive Engine Error: \\(error)")
                exit(1)
            default:
                break
            }
        }
    } catch {
        print("\\n❌ Fatal System Error: \\(error)")
        exit(1)
    }
}
RunLoop.main.run()
"""

    with open(swift_script_path, "w") as f:
        f.write(swift_code)

    print("🚀 Firing up Apple Silicon GPU Core Engine...")
    
    try:
        command = ["swift", str(swift_script_path), str(image_dir), str(output_model_file)]
        subprocess.run(command, check=True)
        print(f"💾 Absolute Survey Asset saved to: {output_model_file.relative_to(base_dir)}")
        print("🎉 View your model instantly by double-clicking it in Finder.")
        
    except subprocess.CalledProcessError:
        print("\n❌ Apple Engine failed to process your GoPro dataset.")
    finally:
        if swift_script_path.exists():
            os.remove(swift_script_path)

if __name__ == "__main__":
    run_apple_gopro_engine()