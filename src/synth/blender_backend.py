"""Real Blender-based mesh and render backends for procedural scene generation.

Utilizes the `bpy` module to construct 3D scenes, PBR materials, and extract
aligned ground-truth labels using Blender's Material Override pipeline.
"""

import math
import tempfile
import glob
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import bpy
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False


class BlenderMeshBackend:
    """Procedurally generates road meshes and defects using Blender API."""

    def __init__(self):
        if not BPY_AVAILABLE:
            raise ImportError("The 'bpy' module is required for the Blender backend.")
        self._clear_scene()
        self._init_materials()
        
    def _clear_scene(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images, 
                           bpy.data.cameras, bpy.data.lights]:
            for block in collection:
                collection.remove(block)

    def _init_materials(self):
        mat_road = bpy.data.materials.new(name="Material_Road")
        mat_road.use_nodes = True
        nodes = mat_road.node_tree.nodes
        links = mat_road.node_tree.links
        
        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            
        noise = nodes.new("ShaderNodeTexNoise")
        bump = nodes.new("ShaderNodeBump")
        
        noise.inputs['Scale'].default_value = 50.0
        noise.inputs['Detail'].default_value = 15.0
        
        links.new(noise.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        
        bsdf.inputs['Roughness'].default_value = 0.8
        self.mat_road = mat_road

    def create_road_mesh(self, lanes: int, lane_width: float, length: float) -> Any:
        width = lanes * lane_width
        
        bpy.ops.mesh.primitive_plane_add(size=1)
        road = bpy.context.active_object
        road.name = "Road"
        road.scale = (width, length, 1)
        road.location = (width / 2.0, length / 2.0, 0)
        
        mod_sub = road.modifiers.new("Subsurf", 'SUBSURF')
        mod_sub.subdivision_type = 'SIMPLE'
        mod_sub.levels = 4
        mod_sub.render_levels = 4
        
        tex = bpy.data.textures.new("Tex_Road", type='NOISE')
        tex.noise_scale = 5.0
        mod_disp = road.modifiers.new(name="Displace", type='DISPLACE')
        mod_disp.texture = tex
        mod_disp.strength = 0.02
        
        if road.data.materials:
            road.data.materials[0] = self.mat_road
        else:
            road.data.materials.append(self.mat_road)
            
        road.pass_index = 1
        road.color = (0.0, 0.0, 0.0, 1.0)
        return road

    def create_defect_mesh(self, defect_type: str, scale: Tuple[float, ...], 
                           position: Tuple[float, float], orientation: float) -> Any:
        if defect_type in ["pothole", "puddle"]:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=scale[0]/2.0, depth=0.01)
        elif defect_type == "manhole":
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=scale[0]/2.0, depth=0.02)
        else:
            bpy.ops.mesh.primitive_plane_add(size=1)
            
        defect = bpy.context.active_object
        defect.name = f"Defect_{defect_type}"
        
        z_offset = 0.005 + np.random.uniform(0.001, 0.005)
        defect.location = (position[0], position[1], z_offset)
        defect.rotation_euler = (0, 0, math.radians(orientation))
        
        if defect_type == "crack":
            defect.scale = (scale[0], scale[1], 1)
            defect.pass_index = 2
            severity_val = 0.8
            
        elif defect_type == "pothole":
            defect.pass_index = 3
            severity_val = min(1.0, scale[1] / 0.15)
            mod_sub = defect.modifiers.new("Subsurf", 'SUBSURF')
            mod_sub.subdivision_type = 'SIMPLE'
            mod_sub.levels = 3
            mod_sub.render_levels = 3
            tex = bpy.data.textures.new("Tex_Pothole", type='CLOUDS')
            tex.noise_scale = 0.5
            mod_disp = defect.modifiers.new(name="Displace", type='DISPLACE')
            mod_disp.texture = tex
            mod_disp.strength = -scale[1]
            
        elif defect_type == "puddle":
            defect.pass_index = 4
            severity_val = 0.5
            
        elif defect_type == "patch":
            defect.scale = (scale[0], scale[1], 1)
            defect.pass_index = 5
            severity_val = 0.2
            
        elif defect_type == "manhole":
            defect.pass_index = 6
            severity_val = 1.0
            
        defect.color = (severity_val, 0.0, 0.0, 1.0)
        
        mat = bpy.data.materials.new(name=f"Mat_{defect_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            out = nodes.get("Material Output")
            mat.node_tree.links.new(bsdf.outputs[0], out.inputs[0])
            
        if defect_type == "crack":
            bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1)
        elif defect_type == "pothole":
            bsdf.inputs['Base Color'].default_value = (0.1, 0.08, 0.05, 1)
        elif defect_type == "puddle":
            bsdf.inputs['Base Color'].default_value = (0.01, 0.01, 0.01, 1)
            bsdf.inputs['Roughness'].default_value = 0.0
            bsdf.inputs['Metallic'].default_value = 0.8
        elif defect_type == "patch":
            bsdf.inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1)
        elif defect_type == "manhole":
            bsdf.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1)
            bsdf.inputs['Metallic'].default_value = 1.0
            
        if defect.data.materials:
            defect.data.materials[0] = mat
        else:
            defect.data.materials.append(mat)
            
        return defect

    def setup_camera_object(self, height: float, pitch: float, view_type: str) -> Any:
        road_length = 100.0
        if "Road" in bpy.data.objects:
            road_length = bpy.data.objects["Road"].scale.y
            
        bpy.ops.object.camera_add(
            location=(bpy.data.objects["Road"].scale.x / 2.0, road_length / 2.0, height),
            rotation=(math.radians(90 + pitch), 0, 0)
        )
        cam = bpy.context.active_object
        cam.name = "Camera"
        bpy.context.scene.camera = cam
        
        fov_deg = 60.0 if view_type == "dashcam" else 90.0
        cam.data.angle = math.radians(fov_deg)
        cam.data.clip_end = 1000.0
        return cam


class BlenderRenderBackend:
    """Manages HDRI, Weather, and multi-pass rendering via Material Overrides."""

    def __init__(self, seed: int = None):
        if not BPY_AVAILABLE:
            raise ImportError("The 'bpy' module is required.")
        self.tmp_dir = Path(tempfile.mkdtemp())
        
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32
        
        self._setup_override_materials()

    def _setup_override_materials(self):
        # 1. Depth Material
        self.mat_depth = bpy.data.materials.new("Override_Depth")
        self.mat_depth.use_nodes = True
        nodes = self.mat_depth.node_tree.nodes
        links = self.mat_depth.node_tree.links
        nodes.clear()
        cam_data = nodes.new("ShaderNodeCameraData")
        math_div = nodes.new("ShaderNodeMath")
        math_div.operation = 'DIVIDE'
        math_div.inputs[1].default_value = 65.535 
        emit = nodes.new("ShaderNodeEmission")
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(cam_data.outputs['View Z Depth'], math_div.inputs[0])
        links.new(math_div.outputs['Value'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], out.inputs['Surface'])
        
        # 2. Segmentation Material
        self.mat_seg = bpy.data.materials.new("Override_Seg")
        self.mat_seg.use_nodes = True
        nodes = self.mat_seg.node_tree.nodes
        links = self.mat_seg.node_tree.links
        nodes.clear()
        obj_info = nodes.new("ShaderNodeObjectInfo")
        math_div = nodes.new("ShaderNodeMath")
        math_div.operation = 'DIVIDE'
        math_div.inputs[1].default_value = 255.0
        emit = nodes.new("ShaderNodeEmission")
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(obj_info.outputs['Object Index'], math_div.inputs[0])
        links.new(math_div.outputs['Value'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], out.inputs['Surface'])
        
        # 3. Severity Material
        self.mat_sev = bpy.data.materials.new("Override_Sev")
        self.mat_sev.use_nodes = True
        nodes = self.mat_sev.node_tree.nodes
        links = self.mat_sev.node_tree.links
        nodes.clear()
        obj_info = nodes.new("ShaderNodeObjectInfo")
        emit = nodes.new("ShaderNodeEmission")
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(obj_info.outputs['Color'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], out.inputs['Surface'])

    def set_hdri_environment(self, hdri_name: str) -> None:
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
            
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = 'NISHITA'
        bg = nodes.new("ShaderNodeBackground")
        out = nodes.new("ShaderNodeOutputWorld")
        
        if "noon" in hdri_name:
            sky.sun_elevation = math.radians(75)
        elif "sunset" in hdri_name:
            sky.sun_elevation = math.radians(10)
            sky.dust_intensity = 5.0
        else:
            sky.sun_elevation = math.radians(45)
            
        sky.sun_rotation = np.random.uniform(0, math.pi * 2)
        
        links.new(sky.outputs['Color'], bg.inputs['Color'])
        links.new(bg.outputs['Background'], out.inputs['Surface'])

    def place_vehicle(self, position: Tuple[float, float], rotation: float) -> Any:
        bpy.ops.mesh.primitive_cube_add(size=1)
        vehicle = bpy.context.active_object
        vehicle.name = "Vehicle"
        vehicle.scale = (1.8, 4.5, 1.4)
        vehicle.location = (position[0], position[1], 0.7)
        vehicle.rotation_euler = (0, 0, math.radians(rotation))
        vehicle.pass_index = 7
        return vehicle

    def apply_weather_effect(self, weather: str) -> None:
        nodes = bpy.context.scene.world.node_tree.nodes
        if "Sky Texture" in nodes:
            sky = nodes["Sky Texture"]
            if weather == "clear":
                sky.sun_intensity = 1.0
                sky.dust_intensity = 1.0
            elif weather == "overcast":
                sky.sun_intensity = 0.1
                sky.dust_intensity = 10.0
            elif weather == "rain":
                sky.sun_intensity = 0.05
                sky.dust_intensity = 8.0
                
        if "Road" in bpy.data.objects:
            mat = bpy.data.objects["Road"].data.materials[0]
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Roughness'].default_value = 0.1 if weather == "rain" else 0.8

    def _render_pass(self, output_path: Path, size: int, material_override=None, 
                     color_mode='BW', color_depth='16', view_transform='Raw', transparent=True):
        bpy.context.scene.render.resolution_x = size
        bpy.context.scene.render.resolution_y = size
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = color_mode
        bpy.context.scene.render.image_settings.color_depth = color_depth
        bpy.context.scene.render.filepath = str(output_path)
        
        bpy.context.scene.view_settings.view_transform = view_transform
        bpy.context.scene.render.film_transparent = transparent
        
        bpy.context.scene.view_layers["ViewLayer"].material_override = material_override
        bpy.ops.render.render(write_still=True)

    def render_rgb(self, output_path: Path, size: int) -> None:
        # Prevent RAW colorspace from destroying the RGB output; apply AgX/Standard
        try:
            bpy.context.scene.view_settings.view_transform = 'AgX'
        except TypeError:
            bpy.context.scene.view_settings.view_transform = 'Standard'
            
        self._render_pass(output_path, size, material_override=None, 
                          color_mode='RGB', color_depth='8', 
                          view_transform=bpy.context.scene.view_settings.view_transform, 
                          transparent=False)

    def render_depth(self, output_path: Path, size: int) -> None:
        self._render_pass(output_path, size, material_override=self.mat_depth, 
                          color_mode='BW', color_depth='16', view_transform='Raw', transparent=True)

    def render_segmentation(self, output_path: Path, size: int) -> None:
        self._render_pass(output_path, size, material_override=self.mat_seg, 
                          color_mode='BW', color_depth='8', view_transform='Raw', transparent=True)

    def render_severity(self, output_path: Path, size: int) -> None:
        import cv2
        tmp_sev_path = self.tmp_dir / "sev_tmp.png"
        self._render_pass(tmp_sev_path, size, material_override=self.mat_sev, 
                          color_mode='BW', color_depth='16', view_transform='Raw', transparent=True)
        
        if tmp_sev_path.exists():
            img = cv2.imread(str(tmp_sev_path), cv2.IMREAD_UNCHANGED)
            severity = (img.astype(np.float32) / 65535.0)
            np.save(str(output_path), severity)
            tmp_sev_path.unlink()