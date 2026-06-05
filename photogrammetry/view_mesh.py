import pathlib
import sys
import pyvista as pv

def view_ply_point_cloud():
    print("🌌 Starting VS Code Point Cloud Inspector (PyVista)...")
    
    base_dir = pathlib.Path(__file__).parent.resolve()
    
    # We are looking for your AI point cloud file
    ai_point_cloud_path = base_dir / "road_model.ply"

    if not ai_point_cloud_path.exists():
        print(f"❌ Error: Could not find 'road_model.ply' inside {base_dir}")
        sys.exit(1)

    print(f"📥 Loading dense point cloud: {ai_point_cloud_path.name}")
    
    try:
        # Load the point cloud using PyVista
        cloud = pv.read(str(ai_point_cloud_path))
        
        if cloud.n_points == 0:
            print("❌ Error: The point cloud file is empty or corrupted.")
            sys.exit(1)
            
        print(f"📊 Points loaded: {cloud.n_points:,}")
        
        # Create a plotter window
        plotter = pv.Plotter(title="AI Monocular Road Point Cloud Inspector")
        
        # Add the point cloud to the scene, rendering the RGB colors natively
        plotter.add_mesh(cloud, rgb=True, point_size=3.0)
        
        print("\n💡 INTERACTIVE CONTROLS:")
        print("   • Click + Drag        -> Rotate camera")
        print("   • Shift + Click + Drag -> Pan (Move sideways)")
        print("   • Scroll Wheel        -> Zoom in/out")
        print("   • Press 'Q'           -> Close window and exit script\n")
        
        # Show the interactive window
        plotter.show()
        
    except Exception as e:
        print(f"❌ An error occurred while opening the viewer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    view_ply_point_cloud()