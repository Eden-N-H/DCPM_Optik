"""Road Quality Pipeline - Main Entry Point.

Usage:
    python -m src.main data --config configs/default.yaml --output-dir ./data/synthetic
    python -m src.main train --config configs/default.yaml [--training.lr=1e-3] [--resume checkpoint.pt]
    python -m src.main evaluate --config configs/default.yaml --checkpoint best_model.pt
    python -m src.main reconstruct --config configs/default.yaml --checkpoint best_model.pt --input video.mp4 --output ./output
    python -m src.main visualize --config configs/default.yaml --cyclegan-ckpt cyclegan.pt --multitask-ckpt multitask.pt --samples 5
    python -m src.main web --config configs/default.yaml --checkpoint best_model.pt
    python -m src.main colab --output Colab_Pipeline.ipynb
"""

import argparse
import logging
import sys
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
    """Run synthetic data generation."""
    logger.info("Starting synthetic dataset generation...")
    
    # Map ConfigLoader settings to DatasetConfig
    dataset_cfg = DatasetConfig(
        total_samples=config.get('scene_generation.dataset_size', 16036),
        split_ratios={
            'train': config.get('data.train_split', 0.8),
            'val': config.get('data.val_split', 0.1),
            'test': config.get('data.test_split', 0.1)
        },
        seed=config.get('seed', 42)
    )
    
    builder = DatasetBuilder(config=dataset_cfg)
    output_path = Path(args.output_dir)
    
    manifest = builder.generate_dataset(output_root=output_path)
    logger.info(f"Dataset generation complete. Manifest saved to {output_path / 'manifest.json'}")
    logger.info(f"Total samples generated: {manifest.total_samples}")


def train(args, config):
    """Run training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Create datasets
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

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create model
    model = MultiTaskModel(
        pretrained=config.get('model.encoder.pretrained', True),
        num_classes=config.get('model.heads.segmentation.num_classes', 3),
        lambda_adv=config.get('domain_adaptation.lambda_adv', 0.1),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params / 1e6:.2f}M")

    # Create trainer
    trainer = MultiTaskTrainer(
        config=config.config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
    )

    # Train
    metrics = trainer.train(resume_from=args.resume)
    logger.info(f"Training complete. Final metrics: {metrics}")


def evaluate(args, config):
    """Run evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Evaluating on device: {device}")

    # Create model and load checkpoint
    model = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 3),
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint_data = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create test dataset
    data_root = config.get('data.root', './data/road_quality')
    test_dataset = RoadQualityDataset(data_root, split='test', crop_size=512)
    test_loader = DataLoader(
        test_dataset, batch_size=config.get('training.batch_size', 8),
        shuffle=False, num_workers=config.get('data.num_workers', 4),
    )

    # Evaluate
    metrics_computer = MetricsComputer(
        num_classes=config.get('model.heads.segmentation.num_classes', 3)
    )

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)

            predictions = model(images)

            targets = {
                'segmentation': batch['segmentation'].to(device),
                'depth': batch['depth'].to(device),
                'severity': batch['severity'].to(device),
                'camera_intrinsics': batch['camera_intrinsics'].to(device),
                'camera_extrinsics': batch['camera_extrinsics'].to(device),
            }

            detached_preds = {k: v.detach() for k, v in predictions.items()
                           if isinstance(v, torch.Tensor)}
            metrics_computer.update(detached_preds, targets)

    metrics = metrics_computer.compute()
    logger.info("Evaluation Results:")
    for key, value in sorted(metrics.items()):
        logger.info(f"  {key}: {value:.4f}")


def reconstruct(args, config):
    """Run 3D reconstruction from video/images."""
    import cv2
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Reconstructing on device: {device}")

    # Load model
    model = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 3),
    )
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create reconstruction pipeline
    recon_config = config.get('reconstruction', {})
    pipeline = ReconstructionPipeline(recon_config)

    # Process input (video or directory of images)
    input_path = Path(args.input)

    if input_path.suffix in ('.mp4', '.avi', '.mov'):
        # Video input
        cap = cv2.VideoCapture(str(input_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _process_frame(model, pipeline, frame_rgb, device)
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")
        cap.release()
    else:
        # Directory of images
        image_paths = sorted(input_path.glob('*.png')) + sorted(input_path.glob('*.jpg'))
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _process_frame(model, pipeline, frame_rgb, device)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")

    # Finalize and export
    output_path = Path(args.output)
    result = pipeline.finalize(output_path)
    if result:
        logger.info(f"BEV map saved to: {result}")
        logger.info(f"PLY file saved to: {output_path / 'reconstruction.ply'}")
    else:
        logger.warning("No valid points after filtering. No output generated.")


def _process_frame(model, pipeline, frame_rgb, device):
    """Process a single frame through the model and add to reconstruction."""
    import cv2
    import numpy as np
    from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD

    # Preprocess
    frame_resized = cv2.resize(frame_rgb, (512, 512))
    img = frame_resized.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    img = (img - mean) / std

    # To tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract predictions
    seg_pred = outputs['segmentation'][0].argmax(dim=0).cpu().numpy()  # [512, 512]
    depth_pred = outputs['depth'][0, 0].cpu().numpy()  # [512, 512]
    severity_pred = outputs['severity'][0, 0].cpu().numpy()  # [512, 512]
    intrinsics_pred = outputs['intrinsics'][0].cpu().numpy()  # [4]
    extrinsics_pred = outputs['extrinsics'][0].cpu().numpy()  # [6]

    # Add to reconstruction pipeline
    predictions = {
        'depth': depth_pred,
        'segmentation': seg_pred,
        'severity': severity_pred,
        'intrinsics': intrinsics_pred,
        'extrinsics': extrinsics_pred,
    }
    pipeline.process_frame(predictions, rgb=frame_resized)


def visualize(args, config):
    """Run end-to-end pipeline visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Visualizing pipeline on device: {device}")

    # Load Dataset (Using val split for stability and proper GT labels)
    data_root = config.get('data.root', './data/road_quality')
    dataset = RoadQualityDataset(data_root, split='val', crop_size=512)
    dataset.is_train = False # Disable augmentation

    # Load CycleGAN Generator
    cyclegan = ResNetGenerator(
        input_channels=config.get('cyclegan.input_nc', 4),
        output_channels=config.get('cyclegan.output_nc', 3),
        ngf=config.get('cyclegan.ngf', 64),
        n_residual_blocks=config.get('cyclegan.n_blocks', 9)
    )
    if args.cyclegan_ckpt:
        cg_data = torch.load(args.cyclegan_ckpt, map_location=device, weights_only=False)
        # Handle full trainer checkpoints vs raw state dicts
        state_dict = cg_data.get('G_AB_state_dict', cg_data)
        cyclegan.load_state_dict(state_dict)

    # Load MultiTask Model
    multitask = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 3),
    )
    if args.multitask_ckpt:
        mt_data = torch.load(args.multitask_ckpt, map_location=device, weights_only=False)
        state_dict = mt_data.get('model_state_dict', mt_data)
        multitask.load_state_dict(state_dict)

    # Init and Run Visualizer
    visualizer = PipelineVisualizer(config.config, dataset, cyclegan, multitask, device)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Generating storyboard grids for {args.samples} samples...")
    visualizer.visualize_samples(args.samples, output_dir)
    logger.info(f"Visualization complete! Output saved to {output_dir}")


def main():
    """Main entry point for the road quality analysis pipeline."""
    parser = argparse.ArgumentParser(description='Road Quality Analysis Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Data command
    data_parser = subparsers.add_parser('data', help='Generate synthetic dataset')
    data_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config YAML file')
    data_parser.add_argument('--output-dir', type=str, default='./data/road_quality',
                             help='Output directory for synthetic data')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config YAML file')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--output-dir', type=str, default='./checkpoints',
                             help='Output directory for checkpoints')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')

    # Reconstruct command
    recon_parser = subparsers.add_parser('reconstruct', help='Run 3D reconstruction')
    recon_parser.add_argument('--config', type=str, default='configs/default.yaml')
    recon_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    recon_parser.add_argument('--input', type=str, required=True,
                             help='Input video file or directory of images')
    recon_parser.add_argument('--output', type=str, default='./reconstruction',
                             help='Output directory')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize pipeline end-to-end')
    viz_parser.add_argument('--config', type=str, default='configs/default.yaml')
    viz_parser.add_argument('--cyclegan-ckpt', type=str, required=False, help='Path to CycleGAN checkpoint')
    viz_parser.add_argument('--multitask-ckpt', type=str, required=True, help='Path to MultiTask checkpoint')
    viz_parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    viz_parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory')

    # Colab Notebook Generation
    colab_parser = subparsers.add_parser('colab', help='Generate Google Colab execution notebook')
    colab_parser.add_argument('--output', type=str, default='Colab_Pipeline.ipynb',
                              help='Output path for the .ipynb file')

    # Web UI command
    web_parser = subparsers.add_parser('web', help='Start the web UI dispatcher')
    web_parser.add_argument('--config', type=str, default='configs/default.yaml')
    web_parser.add_argument('--port', type=int, default=5000,
                            help='Port to run the web server on')
    web_parser.add_argument('--host', type=str, default='0.0.0.0',
                            help='Host interface to bind to')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Collect remaining args as overrides (e.g., --training.lr=1e-3)
    overrides = [a for a in sys.argv[1:] if '=' in a and a.startswith('--')]

    # Don't try to parse config if generating colab (doesn't need it)
    if args.command == 'colab':
        generate_colab_notebook(args.output)
        return

    # Load config
    config = ConfigLoader(config_path=Path(args.config), overrides=overrides if overrides else None)

    # Dispatch
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
    elif args.command == 'web':
        from src.web import start_server
        start_server(args)


if __name__ == "__main__":
    main()
