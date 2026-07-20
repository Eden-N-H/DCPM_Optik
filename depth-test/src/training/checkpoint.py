"""Checkpoint save/load and reproducibility utilities."""
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.data_types import Checkpoint


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states() -> Dict[str, Any]:
    """Capture current RNG states for all random generators.

    Returns:
        Dict with keys: 'python', 'numpy', 'torch', 'cuda'
    """
    states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states from a checkpoint.

    Args:
        states: Dict with keys: 'python', 'numpy', 'torch', optionally 'cuda'
    """
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.random.set_rng_state(states['torch'])
    if 'cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
) -> None:
    """Save a training checkpoint atomically.

    Writes to a temporary file first, then renames it. This prevents
    Google Drive sync from capturing a corrupted or partially written file
    if the process is killed midway through saving.

    Args:
        path: File path to save checkpoint
        model: Model to save state from
        optimizer: Optimizer to save state from
        scheduler: Learning rate scheduler to save state from
        epoch: Current epoch number
        best_metric: Best validation metric achieved so far
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'rng_states': get_rng_states(),
    }

    # Atomic save: write to a .tmp file first
    tmp_path = path.with_suffix('.tmp')
    torch.save(checkpoint_data, str(tmp_path))
    
    # Replace is atomic on POSIX systems
    os.replace(str(tmp_path), str(path))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Checkpoint:
    """Load a training checkpoint and restore states.

    Args:
        path: File path to load checkpoint from
        model: Model to load state into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map tensors to (default: current device)

    Returns:
        Checkpoint dataclass with all stored information
    """
    path = Path(path)
    map_location = device if device is not None else 'cpu'
    checkpoint_data = torch.load(str(path), map_location=map_location, weights_only=False)

    # Restore model
    model.load_state_dict(checkpoint_data['model_state_dict'])

    # Restore optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # Restore scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

    # Restore RNG states
    if 'rng_states' in checkpoint_data:
        set_rng_states(checkpoint_data['rng_states'])

    return Checkpoint(
        epoch=checkpoint_data['epoch'],
        model_state_dict=checkpoint_data['model_state_dict'],
        optimizer_state_dict=checkpoint_data['optimizer_state_dict'],
        scheduler_state_dict=checkpoint_data.get('scheduler_state_dict', {}),
        best_metric=checkpoint_data['best_metric'],
        rng_states=checkpoint_data.get('rng_states', {}),
    )
