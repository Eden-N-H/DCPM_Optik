# create_dummy.py
import torch
from pathlib import Path
from src.model import MultiTaskModel

def main():
    print("Initializing dummy model...")
    # Initialize the model structure without downloading ImageNet weights
    model = MultiTaskModel(pretrained=False, num_classes=8)
    
    # Create the dummy checkpoint dict matching our Checkpoint structure
    checkpoint_data = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'scheduler_state_dict': {},
        'best_metric': 0.0,
    }
    
    output_path = Path("../checkpoints/dummy_model.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint_data, str(output_path))
    print(f"Dummy model saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
