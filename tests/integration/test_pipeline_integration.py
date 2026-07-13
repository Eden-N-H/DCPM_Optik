"""Integration tests for the full road quality pipeline.

Tests TorchScript export/inference equivalence, end-to-end training step,
and reconstruction pipeline on mock predictions.

Validates: Requirements 15.2, 15.3
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import MultiTaskModel
from src.reconstruction import ReconstructionPipeline


class TestTorchScriptExport:
    """Test TorchScript export and inference equivalence.

    Validates: Requirement 15.3 - Support TorchScript export for deployment
    without Python runtime dependency.
    """

    @pytest.fixture
    def model(self):
        """Create a MultiTaskModel for testing."""
        model = MultiTaskModel(pretrained=False, num_classes=3, lambda_adv=0.1)
        model.eval()
        return model

    @pytest.fixture
    def sample_input(self):
        """Create a sample 512x512 input image."""
        image = torch.randn(1, 3, 512, 512)
        return image

    def test_torchscript_trace_export(self, model, sample_input):
        """Test that the model can be exported via torch.jit.trace."""
        image = sample_input

        # Trace the model (strict=False needed because model returns a dict)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (image,), strict=False)

        assert traced_model is not None

    def test_torchscript_inference_equivalence(self, model, sample_input):
        """Test that TorchScript model produces same outputs as eager model."""
        image = sample_input

        # Get eager mode output
        with torch.no_grad():
            eager_output = model(image)

        # Trace and get scripted output (strict=False for dict output)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (image,), strict=False)
            traced_output = traced_model(image)

        # Compare outputs
        for key in ['segmentation', 'severity', 'depth', 'intrinsics', 'extrinsics']:
            assert key in traced_output, f"Missing key '{key}' in traced output"
            torch.testing.assert_close(
                eager_output[key], traced_output[key],
                atol=1e-5, rtol=1e-5,
                msg=f"Output mismatch for '{key}'"
            )

    def test_torchscript_save_and_load(self, model, sample_input, tmp_path):
        """Test that TorchScript model can be saved and loaded from disk."""
        image = sample_input
        model_path = tmp_path / "model_scripted.pt"

        # Trace and save (strict=False for dict output)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (image,), strict=False)
            torch.jit.save(traced_model, str(model_path))

        assert model_path.exists()

        # Load and run inference
        loaded_model = torch.jit.load(str(model_path))
        with torch.no_grad():
            loaded_output = loaded_model(image)

        # Verify outputs match
        with torch.no_grad():
            original_output = model(image)

        for key in ['segmentation', 'severity', 'depth', 'intrinsics', 'extrinsics']:
            torch.testing.assert_close(
                original_output[key], loaded_output[key],
                atol=1e-5, rtol=1e-5,
                msg=f"Loaded model output mismatch for '{key}'"
            )

    def test_torchscript_different_batch_sizes(self, model, sample_input):
        """Test that traced model works with different batch sizes."""
        image = sample_input

        with torch.no_grad():
            traced_model = torch.jit.trace(model, (image,), strict=False)

        # Test batch size 2
        image_b2 = torch.randn(2, 3, 512, 512)

        with torch.no_grad():
            output_b2 = traced_model(image_b2)

        assert output_b2['segmentation'].shape == (2, 3, 512, 512)
        assert output_b2['depth'].shape == (2, 1, 512, 512)
        assert output_b2['severity'].shape == (2, 1, 512, 512)
        assert output_b2['intrinsics'].shape == (2, 4)
        assert output_b2['extrinsics'].shape == (2, 6)


class TestEndToEndTrainingStep:
    """Test end-to-end training step on minimal synthetic data.

    Validates: Requirement 15.2 - Running inference on single 512x512 image
    produces all four outputs.
    """

    @pytest.fixture
    def model(self):
        """Create a MultiTaskModel for training."""
        model = MultiTaskModel(pretrained=False, num_classes=3, lambda_adv=0.1)
        return model

    @pytest.fixture
    def synthetic_batch(self):
        """Create a minimal synthetic data batch mimicking real data."""
        batch_size = 2
        return {
            'image': torch.randn(batch_size, 3, 512, 512),
            'segmentation': torch.randint(0, 3, (batch_size, 512, 512)),
            'depth': torch.rand(batch_size, 1, 512, 512),
            'severity': torch.rand(batch_size, 1, 512, 512),
            'camera_intrinsics': torch.tensor(
                [[256.0, 256.0, 256.0, 256.0]] * batch_size
            ),
            'camera_extrinsics': torch.randn(batch_size, 6),
        }

    def test_single_training_step(self, model, synthetic_batch):
        """Test a complete forward-backward-update training step."""
        from src.training.losses import MultiTaskLoss

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = MultiTaskLoss(
            seg_weight=1.5, depth_weight=1.0,
            camera_weight=0.3, adv_weight=0.1
        )

        images = synthetic_batch['image']

        # Forward pass
        predictions = model(images)

        # Verify all outputs present (Requirement 15.2)
        assert 'segmentation' in predictions
        assert 'severity' in predictions
        assert 'depth' in predictions
        assert 'intrinsics' in predictions
        assert 'extrinsics' in predictions

        # Verify output shapes for 512x512 input
        assert predictions['segmentation'].shape == (2, 3, 512, 512)
        assert predictions['severity'].shape == (2, 1, 512, 512)
        assert predictions['depth'].shape == (2, 1, 512, 512)
        assert predictions['intrinsics'].shape == (2, 4)
        assert predictions['extrinsics'].shape == (2, 6)

        # Compute loss
        targets = {
            'segmentation': synthetic_batch['segmentation'],
            'depth': synthetic_batch['depth'],
            'severity': synthetic_batch['severity'],
            'camera_intrinsics': synthetic_batch['camera_intrinsics'],
            'camera_extrinsics': synthetic_batch['camera_extrinsics'],
        }
        losses = criterion(predictions, targets)

        assert 'total' in losses
        assert not torch.isnan(losses['total']), "Training loss is NaN"
        assert not torch.isinf(losses['total']), "Training loss is Inf"

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()

        # Check gradients flow
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        assert has_grads, "No gradients flowed through the model"

        # Optimizer step
        optimizer.step()

    def test_training_step_with_amp(self, model, synthetic_batch):
        """Test training step with Automatic Mixed Precision."""
        from torch.amp import GradScaler, autocast
        from src.training.losses import MultiTaskLoss

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = MultiTaskLoss()
        scaler = GradScaler('cpu', enabled=False)  # disabled for CPU testing

        images = synthetic_batch['image']

        optimizer.zero_grad()

        # Forward pass with AMP context
        with autocast('cpu', enabled=False):  # disabled for CPU
            predictions = model(images)
            targets = {
                'segmentation': synthetic_batch['segmentation'],
                'depth': synthetic_batch['depth'],
                'severity': synthetic_batch['severity'],
                'camera_intrinsics': synthetic_batch['camera_intrinsics'],
                'camera_extrinsics': synthetic_batch['camera_extrinsics'],
            }
            losses = criterion(predictions, targets)

        # Backward with scaler
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Should complete without error
        assert not torch.isnan(losses['total'])

    def test_inference_produces_all_four_outputs(self, model):
        """Explicitly test Requirement 15.2: single 512x512 image produces all outputs."""
        model.eval()
        image = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            outputs = model(image)

        # All four outputs must be present
        assert 'segmentation' in outputs, "Missing segmentation output"
        assert 'severity' in outputs, "Missing severity output"
        assert 'depth' in outputs, "Missing depth output"
        assert 'intrinsics' in outputs, "Missing intrinsics output"
        assert 'extrinsics' in outputs, "Missing extrinsics output"

        # Verify correct shapes
        assert outputs['segmentation'].shape == (1, 3, 512, 512)
        assert outputs['severity'].shape == (1, 1, 512, 512)
        assert outputs['depth'].shape == (1, 1, 512, 512)
        assert outputs['intrinsics'].shape == (1, 4)
        assert outputs['extrinsics'].shape == (1, 6)

        # Verify value ranges
        assert (outputs['severity'] >= 0).all() and (outputs['severity'] <= 1).all()
        assert (outputs['depth'] >= 0).all() and (outputs['depth'] <= 1).all()
        assert (outputs['intrinsics'] > 0).all(), "Intrinsics should be positive (softplus)"


class TestReconstructionPipeline:
    """Test reconstruction pipeline on mock predictions.

    Validates: Requirements 15.2, 15.3 (full pipeline integration).
    """

    @pytest.fixture
    def pipeline(self):
        """Create a ReconstructionPipeline with default config."""
        config = {
            'bev_resolution': 0.02,
            'depth_confidence_threshold': 0.5,
            'height_range': [-0.5, 0.5],
        }
        return ReconstructionPipeline(config)

    @pytest.fixture
    def mock_predictions(self):
        """Create mock model predictions for a single frame."""
        H, W = 64, 64  # Small for speed
        depth = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float64)
        segmentation = np.random.randint(0, 3, (H, W)).astype(np.int32)
        severity = np.random.uniform(0, 1, (H, W)).astype(np.float64)
        intrinsics = np.array([300.0, 300.0, 32.0, 32.0], dtype=np.float64)
        extrinsics = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)

        return {
            'depth': depth,
            'segmentation': segmentation,
            'severity': severity,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
        }

    def test_process_single_frame(self, pipeline, mock_predictions):
        """Test that a single frame can be processed successfully."""
        result = pipeline.process_frame(mock_predictions)
        assert result is True

    def test_process_multiple_frames(self, pipeline, mock_predictions):
        """Test that multiple frames accumulate correctly."""
        for _ in range(5):
            result = pipeline.process_frame(mock_predictions)
            assert result is True

    def test_finalize_produces_output(self, pipeline, mock_predictions, tmp_path):
        """Test that finalize produces BEV map and PLY file."""
        # Process several frames
        for _ in range(3):
            pipeline.process_frame(mock_predictions)

        # Finalize
        output_dir = tmp_path / "reconstruction_output"
        result = pipeline.finalize(output_dir)

        # Should produce output files
        assert result is not None
        assert output_dir.exists()
        ply_path = output_dir / "reconstruction.ply"
        assert ply_path.exists(), "PLY file was not created"

    def test_process_frame_with_rgb(self, pipeline, mock_predictions):
        """Test processing with optional RGB data."""
        H, W = 64, 64
        rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        result = pipeline.process_frame(mock_predictions, rgb=rgb)
        assert result is True

    def test_invalid_intrinsics_skips_frame(self, pipeline):
        """Test that degenerate intrinsics cause frame to be skipped."""
        H, W = 64, 64
        predictions = {
            'depth': np.random.uniform(1.0, 10.0, (H, W)).astype(np.float64),
            'segmentation': np.zeros((H, W), dtype=np.int32),
            'severity': np.zeros((H, W), dtype=np.float64),
            'intrinsics': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            'extrinsics': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        result = pipeline.process_frame(predictions)
        assert result is False

    def test_model_to_reconstruction_pipeline(self, tmp_path):
        """Test full model inference → reconstruction pipeline integration."""
        # Create model and run inference
        model = MultiTaskModel(pretrained=False, num_classes=3, lambda_adv=0.0)
        model.eval()

        image = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            outputs = model(image)

        # Convert to numpy for reconstruction pipeline
        seg_pred = outputs['segmentation'][0].argmax(dim=0).cpu().numpy()
        depth_pred = outputs['depth'][0, 0].cpu().numpy()
        severity_pred = outputs['severity'][0, 0].cpu().numpy()
        intrinsics_pred = outputs['intrinsics'][0].cpu().numpy()
        extrinsics_pred = outputs['extrinsics'][0].cpu().numpy()

        predictions = {
            'depth': depth_pred,
            'segmentation': seg_pred,
            'severity': severity_pred,
            'intrinsics': intrinsics_pred,
            'extrinsics': extrinsics_pred,
        }

        # Feed into reconstruction pipeline
        config = {
            'bev_resolution': 0.05,
            'depth_confidence_threshold': 0.1,
            'height_range': [-5.0, 5.0],
        }
        pipeline = ReconstructionPipeline(config)
        result = pipeline.process_frame(predictions)

        # The pipeline should accept real model outputs
        # (may or may not produce valid points depending on prediction quality)
        assert isinstance(result, bool)
