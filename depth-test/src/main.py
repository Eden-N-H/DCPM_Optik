"""Road Quality Pipeline - Main Entry Point."""

import argparse
import logging
import sys
import glob
from pathlib import Path
import threading
import requests

import torch
from torch.utils.data import DataLoader

from src.utils.config import ConfigLoader
from src.model import MultiTaskModel
from src.training import RoadQualityDataset, MultiTaskTrainer, set_seed
from src.generate_colab import generate_colab_notebook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def _send_telemetry(url, worker_id, status, epoch=0, train_loss=0.0):
    if not url: return
    payload = {"worker_id": worker_id, "status": status, "epoch": epoch, "train_loss": train_loss}
    def _post():
        try: requests.post(url, json=payload, timeout=5)
        except: pass
    threading.Thread(target=_post, daemon=True).start()


def data(args, config):
    from src.synth.dataset_builder import DatasetBuilder, DatasetConfig
    logger.info("Starting synthetic dataset generation...")
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
    manifest = builder.generate_dataset(output_root=Path(args.output_dir))
    logger.info(f"Dataset generation complete. Saved to {args.output_dir}")


def train_cyclegan(args, config):
    from src.cyclegan.dataset import UnpairedRoadDataset
    from src.cyclegan import CycleGANConfig, CycleGANTrainer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training CycleGAN on device: {device}")
    
    data_root = args.data_root if hasattr(args, 'data_root') and args.data_root else config.get('data.root', './data/road_quality')
    
    train_dataset = UnpairedRoadDataset(data_root, args.real_data, split='train', size=config.get('cyclegan.input_size', 256))
    train_loader = DataLoader(train_dataset, batch_size=config.get('training.batch_size', 4), shuffle=True, num_workers=config.get('data.num_workers', 4), drop_last=True)
    
    cg_cfg = CycleGANConfig(
        input_nc=config.get('cyclegan.input_nc', 4),
        output_nc=config.get('cyclegan.output_nc', 3),
        epochs=config.get('cyclegan.training.epochs', 200),
        lr=config.get('cyclegan.training.lr', 0.0002),
        checkpoint_dir=args.output_dir
    )
    
    trainer = CycleGANTrainer(cg_cfg, device=device)
    
    if args.resume:
        # Load weights logic here if needed
        pass

    _send_telemetry(args.webhook_url, args.worker_id, "started")
    try:
        for epoch in range(cg_cfg.epochs):
            for i, batch in enumerate(train_loader):
                losses = trainer.train_step(batch['A'], batch['B'], batch['mask_A'])
                if losses.get('diverged', False):
                    logger.error("CycleGAN diverged.")
                    _send_telemetry(args.webhook_url, args.worker_id, "failed_nan")
                    return
            
            trainer.step_schedulers()
            _send_telemetry(args.webhook_url, args.worker_id, "running", epoch, losses.get('loss_G', 0.0))
            
            if (epoch + 1) % 5 == 0:
                trainer._save_checkpoint(Path(args.output_dir), epoch, reason="")
                
        trainer._save_checkpoint(Path(args.output_dir), cg_cfg.epochs, reason="final")
        _send_telemetry(args.webhook_url, args.worker_id, "completed")
    except Exception as e:
        logger.error(f"CycleGAN Failed: {e}")
        _send_telemetry(args.webhook_url, args.worker_id, "failed")


def translate(args, config):
    from src.cyclegan.translator import DatasetTranslator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        ckpts = sorted(glob.glob(str(ckpt / "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt}")
        ckpt = Path(ckpts[-1])
        
    translator = DatasetTranslator(str(ckpt), config.config, device)
    translator.translate_dataset(args.input_dir, args.output_dir)


def train(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training MT Model on device: {device}")

    set_seed(config.get('seed', 42))

    data_root = args.data_root if hasattr(args, 'data_root') and args.data_root else config.get('data.root', './data/road_quality')
    batch_size = config.get('training.batch_size', 8)
    num_workers = config.get('data.num_workers', 4)

    train_dataset = RoadQualityDataset(data_root, split='train', crop_size=480)
    val_dataset = RoadQualityDataset(data_root, split='val', crop_size=512)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiTaskModel(
        pretrained=config.get('model.encoder.pretrained', True),
        num_classes=config.get('model.heads.segmentation.num_classes', 8),
        lambda_adv=config.get('domain_adaptation.lambda_adv', 0.1),
    )

    trainer = MultiTaskTrainer(
        config=config.config, model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, output_dir=args.output_dir, webhook_url=args.webhook_url, worker_id=args.worker_id,
    )
    trainer.train(resume_from=args.resume)


def evaluate(args, config):
    import json
    from src.training.metrics import MetricsComputer
    from src.training.checkpoint import load_checkpoint
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Evaluating MT Model on device: {device}")

    data_root = config.get('data.root', './data/road_quality')
    dataset = RoadQualityDataset(data_root, split='test', crop_size=512)
    loader = DataLoader(dataset, batch_size=config.get('training.batch_size', 8), shuffle=False)

    model = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 8)
    ).to(device)

    load_checkpoint(Path(args.checkpoint), model, device=device)
    model.eval()

    metrics_comp = MetricsComputer(num_classes=config.get('model.heads.segmentation.num_classes', 8))

    logger.info("Starting evaluation pass...")
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = {
                'segmentation': batch['segmentation'].to(device),
                'depth': batch['depth'].to(device),
                'severity': batch['severity'].to(device),
                'camera_intrinsics': batch['camera_intrinsics'].to(device),
                'camera_extrinsics': batch['camera_extrinsics'].to(device),
            }
            preds = model(images, use_domain_adapter=False)
            metrics_comp.update(preds, targets)

    metrics = metrics_comp.compute()
    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    out_path = Path(args.checkpoint).parent / f"eval_metrics_{Path(args.checkpoint).stem}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics successfully saved to {out_path}")


def reconstruct(args, config):
    import cv2
    import numpy as np
    from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD
    from src.training.checkpoint import load_checkpoint
    from src.reconstruction.pipeline import ReconstructionPipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Reconstructing on device: {device}")

    model = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 8)
    ).to(device)

    load_checkpoint(Path(args.checkpoint), model, device=device)
    model.eval()

    pipeline = ReconstructionPipeline(config.get('reconstruction', {}))
    input_path = Path(args.input)
    frames = []
    
    if input_path.is_dir():
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            frames.extend(sorted(input_path.glob(ext)))
    elif input_path.is_file():
        if input_path.suffix.lower() not in ['.mp4', '.avi', '.mov']:
            frames = [input_path]
    else:
        logger.error(f"Input path {input_path} does not exist.")
        return

    def process_image(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_norm = img_resized.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_norm - mean) / std

        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor, use_domain_adapter=False)
        
        preds_np = {
            'depth': preds['depth'][0, 0].cpu().numpy(),
            'segmentation': preds['segmentation'][0].argmax(dim=0).cpu().numpy(),
            'severity': preds['severity'][0, 0].cpu().numpy(),
            'intrinsics': preds['intrinsics'][0].cpu().numpy(),
            'extrinsics': preds['extrinsics'][0].cpu().numpy()
        }
        pipeline.process_frame(preds_np, rgb=img_resized)

    if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(str(input_path))
        while True:
            ret, frame = cap.read()
            if not ret: break
            process_image(frame)
        cap.release()
    else:
        for f_path in frames:
            frame = cv2.imread(str(f_path))
            if frame is not None:
                process_image(frame)

    out_dir = Path(args.output)
    res = pipeline.finalize(out_dir)
    if res:
        logger.info(f"Reconstruction successfully saved to {out_dir}")
    else:
        logger.warning("Reconstruction failed or no points generated (Empty point cloud).")


def visualize(args, config):
    from src.training.checkpoint import load_checkpoint
    from src.cyclegan.generator import ResNetGenerator
    from src.visualization.visualizer import PipelineVisualizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Visualizing on device: {device}")

    mt_model = MultiTaskModel(
        pretrained=False,
        num_classes=config.get('model.heads.segmentation.num_classes', 8)
    ).to(device)
    
    if getattr(args, 'multitask_ckpt', None) and Path(args.multitask_ckpt).exists():
        load_checkpoint(Path(args.multitask_ckpt), mt_model, device=device)
    else:
        logger.warning("No MultiTask checkpoint provided. Using untrained weights.")
    mt_model.eval()

    if getattr(args, 'cyclegan_ckpt', None) and Path(args.cyclegan_ckpt).exists():
        cg_model = ResNetGenerator(
            input_channels=config.get('cyclegan.input_nc', 4),
            output_channels=config.get('cyclegan.output_nc', 3),
            ngf=config.get('cyclegan.ngf', 64),
            n_residual_blocks=config.get('cyclegan.n_blocks', 9)
        ).to(device)
        ckpt = torch.load(args.cyclegan_ckpt, map_location=device, weights_only=False)
        cg_model.load_state_dict(ckpt['G_AB_state_dict'])
        cg_model.eval()
    else:
        class DummyCG(torch.nn.Module):
            def forward(self, x):
                return x[:, :3, :, :]
        cg_model = DummyCG().to(device).eval()

    data_root = config.get('data.root', './data/road_quality')
    dataset = RoadQualityDataset(data_root, split='val', crop_size=512)

    visualizer = PipelineVisualizer(config.config, dataset, cg_model, mt_model, device)
    visualizer.visualize_samples(args.samples, Path(args.output_dir))
    logger.info(f"Visualizations saved to {args.output_dir}")


def auto(args, config):
    from src.pipeline import PipelineRunner
    runner = PipelineRunner(args.config, args.real_data, args.output_dir)
    runner.run_all()


def main():
    parser = argparse.ArgumentParser(description='Road Quality Analysis Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Data
    data_parser = subparsers.add_parser('data', help='Generate synthetic dataset')
    data_parser.add_argument('--config', type=str, default='configs/default.yaml')
    data_parser.add_argument('--output-dir', type=str, default='./data/road_quality')

    # Train CycleGAN
    cg_parser = subparsers.add_parser('train_cyclegan', help='Train CycleGAN')
    cg_parser.add_argument('--config', type=str, default='configs/default.yaml')
    cg_parser.add_argument('--resume', type=str, default=None)
    cg_parser.add_argument('--real-data', type=str, required=True)
    cg_parser.add_argument('--data.root', type=str, dest='data_root')
    cg_parser.add_argument('--output-dir', type=str, default='./checkpoints/cyclegan')
    cg_parser.add_argument('--webhook-url', type=str, default=None)
    cg_parser.add_argument('--worker-id', type=str, default=None)

    # Translate
    tr_parser = subparsers.add_parser('translate', help='Translate synthetic data to real style')
    tr_parser.add_argument('--config', type=str, default='configs/default.yaml')
    tr_parser.add_argument('--checkpoint', type=str, required=True)
    tr_parser.add_argument('--input-dir', type=str, required=True)
    tr_parser.add_argument('--output-dir', type=str, required=True)

    # Train MultiTask
    train_parser = subparsers.add_parser('train', help='Train the MultiTask model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml')
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--data.root', type=str, dest='data_root')
    train_parser.add_argument('--output-dir', type=str, default='./checkpoints/multitask')
    train_parser.add_argument('--webhook-url', type=str, default=None)
    train_parser.add_argument('--worker-id', type=str, default=None)
    
    # Evaluate MultiTask
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the MultiTask model')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    
    # Reconstruct
    recon_parser = subparsers.add_parser('reconstruct', help='Run 3D Reconstruction and BEV export')
    recon_parser.add_argument('--config', type=str, default='configs/default.yaml')
    recon_parser.add_argument('--checkpoint', type=str, required=True)
    recon_parser.add_argument('--input', type=str, required=True)
    recon_parser.add_argument('--output', type=str, required=True)
    
    # Visualize
    vis_parser = subparsers.add_parser('visualize', help='Generate visual output comparisons')
    vis_parser.add_argument('--config', type=str, default='configs/default.yaml')
    vis_parser.add_argument('--cyclegan-ckpt', type=str, default=None)
    vis_parser.add_argument('--multitask-ckpt', type=str, default=None)
    vis_parser.add_argument('--samples', type=int, default=5)
    vis_parser.add_argument('--output-dir', type=str, required=True)

    # Quicktest (Alias mapped directly to visualize)
    qt_parser = subparsers.add_parser('quicktest', help='Run a quick visualization test')
    qt_parser.add_argument('--config', type=str, default='configs/default.yaml')
    qt_parser.add_argument('--cyclegan-ckpt', type=str, default=None)
    qt_parser.add_argument('--multitask-ckpt', type=str, default=None)
    qt_parser.add_argument('--samples', type=int, default=5)
    qt_parser.add_argument('--output-dir', type=str, required=True)

    # Auto Pipeline
    auto_parser = subparsers.add_parser('auto', help='Run full pipeline locally')
    auto_parser.add_argument('--config', type=str, default='configs/default.yaml')
    auto_parser.add_argument('--real-data', type=str, required=True)
    auto_parser.add_argument('--output-dir', type=str, default='./workspace')

    # Worker, Web, Colab
    colab_parser = subparsers.add_parser('colab', help='Generate Colab Notebook')
    colab_parser.add_argument('--output', type=str, default='Colab_Pipeline.ipynb')

    web_parser = subparsers.add_parser('web', help='Start web UI')
    web_parser.add_argument('--config', type=str, default='configs/default.yaml')
    web_parser.add_argument('--port', type=int, default=5000)
    web_parser.add_argument('--host', type=str, default='0.0.0.0')

    worker_parser = subparsers.add_parser('worker', help='Start Colab worker')
    worker_parser.add_argument('--orchestrator-url', type=str, required=True)
    worker_parser.add_argument('--shared-drive-path', type=str, required=True)
    worker_parser.add_argument('--worker-id', type=str, required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'colab':
        generate_colab_notebook(args.output)
        return
    elif args.command == 'worker':
        from src.worker import start_worker
        start_worker(args.orchestrator_url, args.shared_drive_path, args.worker_id)
        return

    overrides = [a for a in sys.argv[1:] if '=' in a and a.startswith('--')]
    config = ConfigLoader(config_path=Path(args.config), overrides=overrides if overrides else None)

    if args.command == 'data': data(args, config)
    elif args.command == 'train_cyclegan': train_cyclegan(args, config)
    elif args.command == 'translate': translate(args, config)
    elif args.command == 'train': train(args, config)
    elif args.command == 'evaluate': evaluate(args, config)
    elif args.command == 'reconstruct': reconstruct(args, config)
    elif args.command == 'visualize': visualize(args, config)
    elif args.command == 'quicktest': visualize(args, config)
    elif args.command == 'auto': auto(args, config)
    elif args.command == 'web':
        from src.web import start_server
        start_server(args)

if __name__ == "__main__":
    main()
