"""Dataset builder for orchestrating full synthetic dataset generation.

Generates the complete dataset of images split into train/val/test sets with
balanced dashcam/drone viewpoints. Uses concurrent subprocesses to instruct the
system Blender executable to render scenes, vastly increasing generation speed
while avoiding memory leaks and library conflict issues.

Requirements: 1.7, 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2

from src.synth.scene_generator import SceneConfig, SceneGenerator
from src.synth.renderer import (
    DomainRandomizationConfig,
    RenderConfig,
    SceneRenderer,
)
from src.utils.data_types import DatasetManifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET_SIZE = 16036

DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

VIEW_TYPES: List[str] = ["dashcam", "drone"]

@dataclass
class DatasetConfig:
    total_samples: int = DEFAULT_DATASET_SIZE
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS)
    )
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    dr_config: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    render_config: RenderConfig = field(default_factory=RenderConfig)
    seed: Optional[int] = None


class DryRunRenderBackend:
    """A dummy backend to satisfy SceneRenderer so it just yields paths."""
    def set_hdri_environment(self, hdri_name: str) -> None: pass
    def place_vehicle(self, position: Tuple[float, float], rotation: float) -> Any: return None
    def apply_weather_effect(self, weather: str) -> None: pass
    def render_rgb(self, output_path: Path, size: int) -> None: pass
    def render_depth(self, output_path: Path, size: int) -> None: pass
    def render_segmentation(self, output_path: Path, size: int) -> None: pass
    def render_severity(self, output_path: Path, size: int) -> None: pass


# ---------------------------------------------------------------------------
# Split computation helpers
# ---------------------------------------------------------------------------

def compute_split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    exact_counts = {name: total * ratio for name, ratio in ratios.items()}
    floor_counts = {name: int(math.floor(count)) for name, count in exact_counts.items()}
    remainder = total - sum(floor_counts.values())
    remainders = {name: exact_counts[name] - floor_counts[name] for name in ratios}

    sorted_names = sorted(remainders.keys(), key=lambda n: (-remainders[n], n))
    for i in range(remainder):
        floor_counts[sorted_names[i]] += 1
    return floor_counts

def compute_view_counts(split_count: int) -> Dict[str, int]:
    dashcam_count = split_count // 2
    drone_count = split_count - dashcam_count
    return {"dashcam": dashcam_count, "drone": drone_count}


# ---------------------------------------------------------------------------
# DatasetBuilder class
# ---------------------------------------------------------------------------

class DatasetBuilder:
    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        scene_generator: Optional[SceneGenerator] = None,
        renderer: Optional[SceneRenderer] = None,
    ) -> None:
        self.config = config or DatasetConfig()
        seed = self.config.seed

        self.scene_generator = scene_generator or SceneGenerator(
            config=self.config.scene_config, seed=seed
        )
        self.renderer = renderer or SceneRenderer(
            dr_config=self.config.dr_config,
            render_config=self.config.render_config,
            seed=seed,
            render_backend=DryRunRenderBackend()
        )
        self.rng = random.Random(seed)

    def generate_dataset(
        self,
        output_root: Path,
        total_samples: Optional[int] = None,
    ) -> DatasetManifest:
        output_root = Path(output_root)
        total = total_samples if total_samples is not None else self.config.total_samples

        split_counts = compute_split_counts(total, self.config.split_ratios)
        
        tasks = []
        scene_counter = 0
        for split_name, split_count in split_counts.items():
            view_counts = compute_view_counts(split_count)
            for view_type, view_count in view_counts.items():
                for _ in range(view_count):
                    scene_id = f"scene_{scene_counter:06d}"
                    tasks.append((output_root, split_name, view_type, scene_id))
                    scene_counter += 1

        all_samples: List[Dict[str, Any]] = []
        
        # Parallel Execution
        max_workers = max(1, os.cpu_count() // 2)
        logger.info(f"Starting rendering queue with {max_workers} background Blender workers...")
        
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(self._generate_single_sample, *task): task
                for task in tasks
            }
            for future in as_completed(future_to_job):
                task_info = future_to_job[future]
                try:
                    sample_metadata = future.result()
                    all_samples.append(sample_metadata)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Generated {completed}/{total} scenes...")
                except Exception as exc:
                    logger.error(f"Scene {task_info[3]} failed in dataset builder: {exc}")

        # Build manifest
        manifest = DatasetManifest(
            root=output_root,
            total_samples=len(all_samples),
            splits=split_counts,
            samples=all_samples,
        )
        self._write_manifest(output_root, manifest)
        return manifest

    def _get_severity_val(self, dtype: str, scale: Tuple) -> float:
        if dtype == "crack": return 0.8
        if dtype == "pothole": return min(1.0, scale[1] / 0.15)
        if dtype == "puddle": return 0.5
        if dtype == "patch": return 0.2
        if dtype == "manhole": return 1.0
        return 0.5

    def _generate_single_sample(
        self,
        output_root: Path,
        split: str,
        view_type: str,
        scene_id: str,
    ) -> Dict[str, Any]:
        
        scene_dir = output_root / split / view_type / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        road_mesh, road_width, road_length = self.scene_generator.generate_road_mesh()
        defect_instances = self.scene_generator.place_defects(
            road_width=road_width, road_length=road_length,
        )
        camera_config = self.scene_generator.setup_camera(
            view_type=view_type, road_length=road_length,
        )
        dr_state = self.renderer.apply_domain_randomization(
            road_width=road_width, road_length=road_length,
        )
        
        # Saves camera.json
        render_outputs = self.renderer.render(
            output_dir=scene_dir, camera_config=camera_config, scene_id=scene_id,
        )

        # Assemble Defect specs for the worker
        defects_payload = []
        for inst in defect_instances:
            defects_payload.append({
                "type": inst.spec.defect_type,
                "scale": inst.spec.scale,
                "position": inst.spec.position,
                "orientation": inst.spec.orientation,
                "severity": self._get_severity_val(inst.spec.defect_type, inst.spec.scale)
            })

        # Assemble JSON Job
        sev_png = str(render_outputs.severity).replace('.npy', '.png')
        
        job = {
            "scene_id": scene_id,
            "render_size": self.config.render_config.render_size,
            "road": {"width": road_width, "length": road_length},
            "defects": defects_payload,
            "vehicles": dr_state.vehicle_positions,
            "camera": {
                "height": camera_config.height,
                "pitch": camera_config.pitch,
                "fov": 60.0 if view_type == "dashcam" else 90.0
            },
            "env": {
                "hdri": dr_state.hdri_map,
                "weather": dr_state.weather
            },
            "paths": {
                "rgb": str(render_outputs.rgb),
                "depth": str(render_outputs.depth),
                "seg": str(render_outputs.segmentation),
                "sev": sev_png
            }
        }

        # Write Job File
        job_file = scene_dir / f"{scene_id}_job.json"
        with open(job_file, "w") as f:
            json.dump(job, f)

        # Execute Blender subprocess with explicit error capturing
        worker_script = Path(__file__).parent / "blender_worker.py"
        try:
            subprocess.run(
                ["blender", "-b", "-P", str(worker_script), "--", "--job-file", str(job_file)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            err_msg = f"\n=== BLENDER CRASH ===\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}\n===================="
            logger.error(f"Scene {scene_id} failed with error: {err_msg}")
            raise RuntimeError(f"Blender worker failed for {scene_id}. See logs for Traceback.")

        # Post-Process Severity (PNG to NPY)
        if Path(sev_png).exists():
            img = cv2.imread(sev_png, cv2.IMREAD_UNCHANGED)
            severity = (img.astype(np.float32) / 65535.0)
            np.save(str(render_outputs.severity), severity)
            Path(sev_png).unlink()

        # Clean up job file
        job_file.unlink(missing_ok=True)

        defect_types_present = list(set(inst.spec.defect_type for inst in defect_instances))

        return {
            "scene_id": scene_id,
            "split": split,
            "view_type": view_type,
            "defect_types_present": sorted(defect_types_present),
            "num_defects": len(defect_instances),
            "camera_config": {
                "height": camera_config.height,
                "pitch": camera_config.pitch,
                "view_type": camera_config.view_type,
            },
            "paths": {
                "rgb": str(render_outputs.rgb),
                "depth": str(render_outputs.depth),
                "segmentation": str(render_outputs.segmentation),
                "severity": str(render_outputs.severity),
                "camera_params": str(render_outputs.camera_params),
            },
            "domain_randomization": {
                "hdri_map": dr_state.hdri_map,
                "vehicle_count": dr_state.vehicle_count,
                "weather": dr_state.weather,
            },
        }

    def validate_split_balance(self, manifest: DatasetManifest) -> Dict[str, Any]:
        # Remains unchanged...
        total = manifest.total_samples
        results: Dict[str, Any] = {"valid": True, "issues": []}

        target = self.config.total_samples
        lower = int(target * 0.99)
        upper = int(target * 1.01)
        if not (lower <= total <= upper):
            results["valid"] = False
            results["issues"].append(
                f"Total samples {total} outside ±1% of {target} (expected {lower}-{upper})"
            )

        view_counts: Dict[str, int] = {"dashcam": 0, "drone": 0}
        for sample in manifest.samples:
            vt = sample.get("view_type", "")
            if vt in view_counts:
                view_counts[vt] += 1

        for view_type, count in view_counts.items():
            proportion = count / total if total > 0 else 0
            if not (0.48 <= proportion <= 0.52):
                results["valid"] = False
                results["issues"].append(
                    f"View type '{view_type}' proportion {proportion:.3f} outside 50% ±2% range"
                )

        for split_name, expected_ratio in self.config.split_ratios.items():
            split_count = manifest.splits.get(split_name, 0)
            actual_ratio = split_count / total if total > 0 else 0
            tolerance = 0.01
            if not (expected_ratio - tolerance <= actual_ratio <= expected_ratio + tolerance):
                results["valid"] = False
                results["issues"].append(
                    f"Split '{split_name}' ratio {actual_ratio:.3f} outside {expected_ratio} ±{tolerance}"
                )

        results["total_samples"] = total
        results["view_counts"] = view_counts
        results["split_counts"] = dict(manifest.splits)

        return results

    def _write_manifest(self, output_root: Path, manifest: DatasetManifest) -> None:
        manifest_path = output_root / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_dict = {
            "root": str(manifest.root),
            "total_samples": manifest.total_samples,
            "splits": manifest.splits,
            "samples": manifest.samples,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)