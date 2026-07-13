"""Standalone Blender worker script for rendering synthetic scenes.
Executes headlessly via subprocess from dataset_builder.py.
"""
import sys
import json
import math
import tempfile
import glob
import shutil
import os
import traceback
import random

try:
    import bpy
except ImportError:
    pass  # We only expect this to run inside a Blender executable


def main():
    try:
        # Extract job file argument
        argv = sys.argv
        if "--" not in argv:
            sys.exit("No arguments passed to Blender worker.")
        
        args = argv[argv.index("--") + 1:]
        job_file = args[args.index("--job-file") + 1]

        with open(job_file, 'r') as f:
            job = json.load(f)

        # 1. Clean Scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # 2. Setup Render Engine
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32
        
        # Disable denoising to prevent crashes on builds lacking OpenImageDenoise
        bpy.context.scene.cycles.use_denoising = False 
        
        bpy.context.scene.render.resolution_x = job['render_size']
        bpy.context.scene.render.resolution_y = job['render_size']
        
        # Attempt to enable GPU explicitly with logging
        try:
            # We must explicitly enable the cycles addon in headless mode
            bpy.ops.preferences.addon_enable(module="cycles")
            prefs = bpy.context.preferences.addons['cycles'].preferences
            
            # Fetch devices (Initializes the internal device list)
            prefs.get_devices()
            
            # Try to set OPTIX (Significantly faster for RTX/T4 GPUs on Colab)
            try:
                prefs.compute_device_type = 'OPTIX'
                print("Blender Worker: Compute device type set to OPTIX")
            except TypeError:
                prefs.compute_device_type = 'CUDA'
                print("Blender Worker: Compute device type set to CUDA")
                
            has_gpu = False
            for d in prefs.devices:
                if d.type != 'CPU':
                    d.use = True
                    has_gpu = True
                    print(f"Blender Worker: Enabled GPU device: {d.name} ({d.type})")
                else:
                    # Disable CPU to prevent hybrid rendering bottlenecks
                    d.use = False
            
            if has_gpu:
                bpy.context.scene.cycles.device = 'GPU'
                print("Blender Worker: Cycles rendering set to GPU.")
            else:
                bpy.context.scene.cycles.device = 'CPU'
                print("Blender Worker: No GPU devices found, falling back to CPU.")
                
        except Exception as e:
            print(f"Blender Worker: GPU initialization failed: {e}. Falling back to CPU.", file=sys.stderr)
            bpy.context.scene.cycles.device = 'CPU'

        # 3. Build Road
        bpy.ops.mesh.primitive_plane_add(size=1)
        road = bpy.context.active_object
        road.name = "Road"
        road.scale = (job['road']['width'], job['road']['length'], 1)
        road.location = (job['road']['width'] / 2.0, job['road']['length'] / 2.0, 0)
        road.pass_index = 1
        
        mat_road = bpy.data.materials.new("RoadMat")
        mat_road.use_nodes = True
        nodes = mat_road.node_tree.nodes
        
        # Safely get BSDF
        bsdf = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break
        
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs['Scale'].default_value = 50.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs['Distance'].default_value = 0.1
        
        mat_road.node_tree.links.new(noise.outputs['Color'], bump.inputs['Height'])
        if bsdf:
            mat_road.node_tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
            bsdf.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.8
            
        road.data.materials.append(mat_road)

        # 4. Build Defects
        pass_indices = {"crack": 2, "pothole": 3, "puddle": 4, "patch": 5, "manhole": 6}
        
        for i, d in enumerate(job['defects']):
            dtype = d['type']
            scale = d['scale']
            
            # Blender 4.0+ primitive_cylinder_add uses radius/depth
            if dtype in ["pothole", "puddle", "manhole"]:
                bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=scale[0]/2.0, depth=0.01)
            else:
                bpy.ops.mesh.primitive_plane_add(size=1)
                
            defect = bpy.context.active_object
            defect.name = f"Defect_{i}_{dtype}"
            
            s_x = scale[0]
            s_y = scale[1] if len(scale) > 1 else scale[0]
            if dtype not in ["pothole", "puddle", "manhole"]:
                defect.scale = (s_x, s_y, 1)
                
            z_offset = 0.002 + (i * 0.0001)  # Stagger to prevent Z-fighting
            defect.location = (d['position'][0], d['position'][1], z_offset)
            defect.rotation_euler = (0, 0, math.radians(d['orientation']))
            defect.pass_index = pass_indices.get(dtype, 2)
            
            mat = bpy.data.materials.new(f"Mat_{i}_{dtype}")
            mat.use_nodes = True
            n = mat.node_tree.nodes
            
            # Safely get BSDF
            b = None
            for node in n:
                if node.type == 'BSDF_PRINCIPLED':
                    b = node
                    break
            
            # Create Severity AOV
            aov = n.new("ShaderNodeOutputAOV")
            aov.name = "Severity" # IMPORTANT: Target the specific AOV pass by name
            aov.inputs['Value'].default_value = d.get('severity', 0.5)
            
            if b:
                if dtype == "crack":
                    b.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1)
                elif dtype == "pothole":
                    b.inputs['Base Color'].default_value = (0.05, 0.04, 0.03, 1)
                elif dtype == "puddle":
                    b.inputs['Base Color'].default_value = (0.01, 0.01, 0.01, 1)
                    b.inputs['Roughness'].default_value = 0.0
                    b.inputs['Metallic'].default_value = 0.8
                elif dtype == "patch":
                    b.inputs['Base Color'].default_value = (0.2, 0.2, 0.2, 1)
                elif dtype == "manhole":
                    b.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1)
                    b.inputs['Metallic'].default_value = 1.0
                
            defect.data.materials.append(mat)

        # 5. Build Vehicles
        for i, v in enumerate(job['vehicles']):
            bpy.ops.mesh.primitive_cube_add(size=1)
            veh = bpy.context.active_object
            veh.name = f"Vehicle_{i}"
            veh.scale = (1.8, 4.5, 1.4)
            veh.location = (v[0], v[1], 0.7)
            veh.rotation_euler = (0, 0, math.radians(v[2]))
            veh.pass_index = 7
            
            # FIX: Assign a basic material to vehicles so they don't glow white and wash out
            mat = bpy.data.materials.new(f"Mat_Vehicle_{i}")
            mat.use_nodes = True
            n = mat.node_tree.nodes
            b = None
            for node in n:
                if node.type == 'BSDF_PRINCIPLED':
                    b = node
                    break
                    
            if b:
                color = (random.uniform(0.05, 0.3), random.uniform(0.05, 0.3), random.uniform(0.05, 0.3), 1.0)
                b.inputs['Base Color'].default_value = color
                b.inputs['Roughness'].default_value = 0.3
                b.inputs['Metallic'].default_value = 0.5
                
            veh.data.materials.append(mat)

        # 6. Build Camera
        cam_data = job['camera']
        bpy.ops.object.camera_add(
            location=(job['road']['width']/2.0, job['road']['length']/2.0, cam_data['height']),
            rotation=(math.radians(90 + cam_data['pitch']), 0, 0)
        )
        cam = bpy.context.active_object
        cam.data.angle = math.radians(cam_data['fov'])
        cam.data.clip_end = 1000.0
        bpy.context.scene.camera = cam

        # 7. World (HDRI / Weather)
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        wn = world.node_tree.nodes
        wn.clear()
        
        sky = wn.new("ShaderNodeTexSky")
        sky.sky_type = 'NISHITA'
        bg = wn.new("ShaderNodeBackground")
        # FIX: Lower the strength heavily to prevent Nishita sky from blowing out standard view transforms
        bg.inputs['Strength'].default_value = 0.05 
        out = wn.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(sky.outputs['Color'], bg.inputs['Color'])
        world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
        
        weather = job['env']['weather']
        dust_val = 1.0
        if weather == "clear":
            sky.sun_intensity = 1.0
            dust_val = 1.0
        elif weather == "overcast":
            sky.sun_intensity = 0.2
            dust_val = 5.0
        elif weather == "rain":
            sky.sun_intensity = 0.05
            dust_val = 10.0
            
        # Robustly handle the API change from dust_intensity to dust_density
        if hasattr(sky, 'dust_density'):
            sky.dust_density = dust_val
        elif hasattr(sky, 'dust_intensity'):
            sky.dust_intensity = dust_val
            
        hdri = job['env']['hdri']
        if "noon" in hdri:
            sky.sun_elevation = math.radians(75)
        elif "sunset" in hdri:
            sky.sun_elevation = math.radians(10)
        else:
            sky.sun_elevation = math.radians(45)

        # 8. Compositor Setup
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        tree.nodes.clear()
        
        vl = bpy.context.scene.view_layers["ViewLayer"]
        vl.use_pass_z = True
        vl.use_pass_object_index = True
        if "Severity" not in vl.aovs:
            aov = vl.aovs.add()
            aov.name = "Severity"
            
        rlayers = tree.nodes.new('CompositorNodeRLayers')
        tmp_dir = tempfile.mkdtemp()
        
        # RGB
        out_rgb = tree.nodes.new('CompositorNodeOutputFile')
        out_rgb.format.file_format = 'PNG'
        out_rgb.format.color_mode = 'RGB'
        out_rgb.base_path = tmp_dir
        out_rgb.file_slots[0].path = "rgb_"
        
        # Avoid AgX issues on some Blender versions
        try:
            bpy.context.scene.view_settings.view_transform = 'Standard'
        except Exception:
            pass
            
        if 'Image' in rlayers.outputs:
            tree.links.new(rlayers.outputs['Image'], out_rgb.inputs[0])
        
        # Depth (m to mm -> 16-bit BW)
        math_node = tree.nodes.new('CompositorNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = 1000.0
        out_depth = tree.nodes.new('CompositorNodeOutputFile')
        out_depth.format.file_format = 'PNG'
        out_depth.format.color_mode = 'BW'
        out_depth.format.color_depth = '16'
        out_depth.base_path = tmp_dir
        out_depth.file_slots[0].path = "depth_"
        if 'Depth' in rlayers.outputs:
            tree.links.new(rlayers.outputs['Depth'], math_node.inputs[0])
            tree.links.new(math_node.outputs['Value'], out_depth.inputs[0])
        
        # Segmentation (8-bit BW)
        out_seg = tree.nodes.new('CompositorNodeOutputFile')
        out_seg.format.file_format = 'PNG'
        out_seg.format.color_mode = 'BW'
        out_seg.format.color_depth = '8'
        out_seg.base_path = tmp_dir
        out_seg.file_slots[0].path = "seg_"
        if 'IndexOB' in rlayers.outputs:
            tree.links.new(rlayers.outputs['IndexOB'], out_seg.inputs[0])
        
        # Severity (16-bit BW)
        out_sev = tree.nodes.new('CompositorNodeOutputFile')
        out_sev.format.file_format = 'PNG'
        out_sev.format.color_mode = 'BW'
        out_sev.format.color_depth = '16'
        out_sev.base_path = tmp_dir
        out_sev.file_slots[0].path = "sev_"
        
        sev_socket = rlayers.outputs.get('Severity')
        if sev_socket:
            tree.links.new(sev_socket, out_sev.inputs[0])
        
        # 9. Render & Move Files
        bpy.ops.render.render(write_still=False)
        
        paths = job['paths']
        os.makedirs(os.path.dirname(paths['rgb']), exist_ok=True)
        os.makedirs(os.path.dirname(paths['depth']), exist_ok=True)
        os.makedirs(os.path.dirname(paths['seg']), exist_ok=True)
        os.makedirs(os.path.dirname(paths['sev']), exist_ok=True)
        
        def move_output(prefix, target_path):
            files = glob.glob(os.path.join(tmp_dir, f"{prefix}*.png"))
            if not files:
                raise FileNotFoundError(f"Missing render pass: {prefix} in {tmp_dir}")
            shutil.move(files[0], target_path)

        move_output("rgb_", paths['rgb'])
        move_output("depth_", paths['depth'])
        move_output("seg_", paths['seg'])
        move_output("sev_", paths['sev'])
        
        shutil.rmtree(tmp_dir)

    except Exception as e:
        print("BLENDER WORKER FATAL ERROR:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
