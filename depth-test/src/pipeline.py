"""Automated Local Pipeline Runner."""

import subprocess
import sys
from pathlib import Path


class PipelineRunner:
    """Executes the full pipeline DAG sequentially on the local machine."""
    
    def __init__(self, config_path: str, real_data: str, output_dir: str):
        self.config = config_path
        self.real_data = real_data
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        
    def run_all(self):
        print("Starting Automated Local Pipeline...")
        
        data_dir = self.output / "road_quality"
        cg_ckpt = self.output / "checkpoints" / "cyclegan"
        trans_dir = self.output / "road_quality_translated"
        mt_ckpt = self.output / "checkpoints" / "multitask"
        
        cmds = [
            ["data", "--config", self.config, "--output-dir", str(data_dir)],
            ["train_cyclegan", "--config", self.config, f"--data.root={data_dir}", "--real-data", self.real_data, "--output-dir", str(cg_ckpt)],
            ["translate", "--config", self.config, "--input-dir", str(data_dir), "--output-dir", str(trans_dir), "--checkpoint", str(cg_ckpt)],
            ["train", "--config", self.config, f"--data.root={trans_dir}", "--output-dir", str(mt_ckpt)]
        ]
        
        for cmd in cmds:
            full_cmd = [sys.executable, "-m", "src.main"] + cmd
            print(f"\n======================================")
            print(f"RUNNING STEP: {cmd[0].upper()}")
            print(f"======================================\n")
            
            proc = subprocess.run(full_cmd)
            if proc.returncode != 0:
                print(f"Pipeline failed at step: {cmd[0]}")
                sys.exit(1)
                
        print("\nPipeline completed successfully!")
