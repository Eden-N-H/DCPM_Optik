"""Road Quality Pipeline - Main Entry Point.

Usage:
    python -m src.main data --config configs/default.yaml --output-dir ./data/synthetic
    python -m src.main train --config configs/default.yaml [--training.lr=1e-3] [--resume checkpoint.pt]
    python -m src.main evaluate --config configs/default.yaml --checkpoint best_model.pt
    python -m src.main reconstruct --config configs/default.yaml --checkpoint best_model.pt --input video.mp4 --output ./output
    python -m src.main visualize --config configs/default.yaml --cyclegan-ckpt cyclegan.pt --multitask-ckpt multitask.pt --samples 5
    python -m src.main quicktest --config configs/default.yaml --samples 5 --output-dir ./quicktest_out
    python -m src.main web --config configs/default.yaml
    python -m src.main worker --orchestrator-url http://... --shared-drive-path /content/drive/MyDrive
    python -m src.main colab --output Colab_Pipeline.ipynb
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import ConfigLoader
from src.utils.logging import ExperimentLogger
from src.model import MultiTaskModel
from src.cyclegan import ResNetGenerator
from src.training import (
    RoadQualityDataset,
    MultiTaskTrainer,
    MetricsComputer,
    set_seed,
)
from src.reconstruction import ReconstructionPipeline
from src.synth.dataset_builder import DatasetBuilder, DatasetConfig
from src.visualization import PipelineVisualizer
from src.generate_colab import generate_colab_notebook


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def data(args, config):
    logger.info("Starting synthetic dataset generation...")
    dataset_cfg = DatasetConfig(
        total_samples=config.get('scene_generation.dataset_size', 16036),
        split_ratios={
            'train': config.get('data.train_split', 0.8),
            'val': config.get('data.val_split', 0.1),
            'test': config.get('data.test_split', 0.1)
        },
        seed=config.get('seed', 42),
        blender_executable=config.get('scene_generation.blender_executable', 'blender')
    )
    builder = DatasetBuilder(config=dataset_cfg)
    output_path = Path(args.output_dir)
    manifest = builder.generate_dataset(output_root=output_path)
    logger.info(f"Dataset generation complete. Manifest saved to {output_path / 'manifest.json'}")
    logger.info(f"Total samples generated: {manifest.total_samples}")


def train(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    seed = config.get('seed', 42)
    set_seed(seed)

    data_root = config.get('data.root', './data/road_quality')
    batch_size = config.get('training.batch_size', 8)
    num_workers = config.get('data.num_workers', 4)

    train_dataset = RoadQualityDataset(data_root, split='train', crop_size=480)
    val_dataset = RoadQualityDataset(data_root, split='val', crop_size=512)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.get('data.pin_memory', True),
        prefetch_factor=config.get('data.prefetch_factor', 2),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.get('data.pin_memory', True),
        prefetch_factor=config.get('data.prefetch_factor', 2),
    )

    model = MultiTaskModel(
        pretrained=config.get('model.encoder.pretrained', True),
        num_classes=config.get('model.heads.segmentation.num_classes', 3),
        lambda_adv=config.get('domain_adaptation.lambda_adv', 0.1),
    )

    trainer = MultiTaskTrainer(
        config=config.config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        webhook_url=args.webhook_url,
        worker_id=args.worker_id,
    )

    metrics = trainer.train(resume_from=args.resume)
    logger.info(f"Training complete. Final metrics: {metrics}")


def evaluate(args, config):
    # Omitted for brevity in testing context
    pass

def reconstruct(args, config):
    # Omitted for brevity in testing context
    pass

def visualize(args, config):
    # Omitted for brevity in testing context
    pass

def quicktest(args, config):
    # Omitted for brevity in testing context
    pass


def main():
    parser = argparse.ArgumentParser(description='Road Quality Analysis Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Data command
    data_parser = subparsers.add_parser('data', help='Generate synthetic dataset')
    data_parser.add_argument('--config', type=str, default='configs/default.yaml')
    data_parser.add_argument('--output-dir', type=str, default='./data/road_quality')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml')
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--output-dir', type=str, default='./checkpoints')
    train_parser.add_argument('--webhook-url', type=str, default=None, help='URL for telemetry reporting')
    train_parser.add_argument('--worker-id', type=str, default=None, help='ID of the current worker')

    # Eval, Recon, Viz, Quicktest (omitted arguments to keep short, but assume they exist)
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml')
    eval_parser.add_argument('--checkpoint', type=str, required=True)

    recon_parser = subparsers.add_parser('reconstruct', help='Run 3D reconstruction')
    recon_parser.add_argument('--config', type=str, default='configs/default.yaml')
    recon_parser.add_argument('--checkpoint', type=str, required=True)
    recon_parser.add_argument('--input', type=str, required=True)
    recon_parser.add_argument('--output', type=str, default='./reconstruction')

    viz_parser = subparsers.add_parser('visualize', help='Visualize pipeline end-to-end')
    viz_parser.add_argument('--config', type=str, default='configs/default.yaml')
    viz_parser.add_argument('--cyclegan-ckpt', type=str, required=False)
    viz_parser.add_argument('--multitask-ckpt', type=str, required=True)
    viz_parser.add_argument('--samples', type=int, default=5)
    viz_parser.add_argument('--output-dir', type=str, default='./visualizations')
    
    qt_parser = subparsers.add_parser('quicktest', help='Run an end-to-end generation and visualization test')
    qt_parser.add_argument('--config', type=str, default='configs/default.yaml')
    qt_parser.add_argument('--cyclegan-ckpt', type=str, required=False)
    qt_parser.add_argument('--multitask-ckpt', type=str, required=False)
    qt_parser.add_argument('--samples', type=int, default=5)
    qt_parser.add_argument('--output-dir', type=str, default='./quicktest_out')
    qt_parser.add_argument('--keep-data', action='store_true')

    colab_parser = subparsers.add_parser('colab', help='Generate Google Colab execution notebook')
    colab_parser.add_argument('--output', type=str, default='Colab_Pipeline.ipynb')

    web_parser = subparsers.add_parser('web', help='Start the web UI dispatcher')
    web_parser.add_argument('--config', type=str, default='configs/default.yaml')
    web_parser.add_argument('--port', type=int, default=5000)
    web_parser.add_argument('--host', type=str, default='0.0.0.0')

    # Worker command (The Relay Client)
    worker_parser = subparsers.add_parser('worker', help='Start the Colab worker client')
    worker_parser.add_argument('--orchestrator-url', type=str, required=True)
    worker_parser.add_argument('--shared-drive-path', type=str, required=True)
    worker_parser.add_argument('--worker-id', type=str, required=True)
    worker_parser.add_argument('--local-data-path', type=str, default='data', help='Local directory name to extract dataset to')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    overrides = [a for a in sys.argv[1:] if '=' in a and a.startswith('--')]

    if args.command == 'colab':
        generate_colab_notebook(args.output)
        return
        
    if args.command == 'worker':
        from src.worker import start_worker
        start_worker(args.orchestrator_url, args.shared_drive_path, args.worker_id, args.local_data_path)
        return

    config = ConfigLoader(config_path=Path(args.config), overrides=overrides if overrides else None)

    if args.command == 'data':
        data(args, config)
    elif args.command == 'train':
        train(args, config)
    elif args.command == 'evaluate':
        evaluate(args, config)
    elif args.command == 'reconstruct':
        reconstruct(args, config)
    elif args.command == 'visualize':
        visualize(args, config)
    elif args.command == 'quicktest':
        quicktest(args, config)
    elif args.command == 'web':
        from src.web import start_server
        start_server(args)


if __name__ == "__main__":
    main()
