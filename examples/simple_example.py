"""
Simple usage example for MLPot framework.
This script demonstrates basic model creation and training.
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add mlpot to path
sys.path.append(str(Path(__file__).parent.parent))

from mlpot.models.equivariant_net import EquivariantNet
from mlpot.core.trainer import PotentialTrainer
from mlpot.data.dataset import MolecularDataset, create_dataloader
from mlpot.utils.helpers import get_device, set_random_seed
from mlpot.utils.metrics import ModelEvaluator


def create_dummy_data(num_samples=100, num_atoms_range=(5, 20)):
    """
    Create dummy molecular data for demonstration.
    """
    data_list = []
    
    for i in range(num_samples):
        # Random number of atoms
        num_atoms = np.random.randint(*num_atoms_range)
        
        # Random positions
        positions = torch.randn(num_atoms, 3) * 5.0
        
        # Random atomic numbers (1-18 for light elements)
        atomic_numbers = torch.randint(1, 19, (num_atoms,))
        
        # Dummy energy (sum of atomic numbers for simplicity)
        energy = torch.tensor(float(atomic_numbers.sum()), dtype=torch.float32)
        
        # Dummy forces (random)
        forces = torch.randn(num_atoms, 3) * 0.1
        
        data = {
            'pos': positions,
            'atomic_numbers': atomic_numbers,
            'energy': energy,
            'forces': forces
        }
        
        data_list.append(data)
    
    return data_list


def simple_training_example():
    """
    Simple training example with dummy data.
    """
    print("MLPot Simple Training Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dummy data
    print("Creating dummy data...")
    train_data = create_dummy_data(num_samples=200)
    val_data = create_dummy_data(num_samples=50)
    
    # Create model
    print("Creating model...")
    model = EquivariantNet(
        hidden_dim=128,
        num_layers=2,
        num_radial_basis=32,
        cutoff_radius=5.0,
        max_neighbors=10,
        use_periodic_boundary=False,
        online_graph_construction=True
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create data loaders (using dummy data directly)
    print("Creating data loaders...")
    
    def collate_dummy_data(batch):
        """Simple collate function for dummy data."""
        batch_data = {}
        
        # Concatenate positions and forces
        batch_data['pos'] = torch.cat([sample['pos'] for sample in batch], dim=0)
        batch_data['forces'] = torch.cat([sample['forces'] for sample in batch], dim=0)
        batch_data['atomic_numbers'] = torch.cat([sample['atomic_numbers'] for sample in batch], dim=0)
        
        # Stack energies
        batch_data['energy'] = torch.stack([sample['energy'] for sample in batch], dim=0)
        
        # Create batch index
        batch_idx = []
        for i, sample in enumerate(batch):
            num_atoms = sample['pos'].size(0)
            batch_idx.extend([i] * num_atoms)
        batch_data['batch'] = torch.tensor(batch_idx, dtype=torch.long)
        
        # Add natoms
        batch_data['natoms'] = torch.tensor([sample['pos'].size(0) for sample in batch], dtype=torch.long)
        
        return batch_data
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate_dummy_data
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=8, shuffle=False, collate_fn=collate_dummy_data
    )
    
    # Setup training
    print("Setting up training...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    
    trainer = PotentialTrainer(
        model=model,
        optimizer=optimizer,
        loss_config={
            'energy_weight': 1.0,
            'force_weight': 10.0,
            'loss_type': 'l1'
        },
        device=device,
        gradient_clip=10.0
    )
    
    # Train for a few epochs
    print("Starting training...")
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        print("-" * 20)
        
        epoch_metrics = trainer.train_epoch(
            train_loader, val_loader, log_interval=5
        )
        
        print(f"Train Loss: {epoch_metrics['train_total_loss']:.6f}")
        print(f"Val Loss: {epoch_metrics.get('val_total_loss', 'N/A')}")
        print(f"Learning Rate: {epoch_metrics['learning_rate']:.2e}")
    
    # Simple evaluation
    print("\nRunning evaluation...")
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on validation set
    model.eval()
    total_energy_error = 0
    total_force_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            pred_energy, pred_forces = model(batch)
            true_energy = batch['energy']
            true_forces = batch['forces']
            
            energy_error = torch.abs(pred_energy - true_energy).mean().item()
            force_error = torch.abs(pred_forces - true_forces).mean().item()
            
            total_energy_error += energy_error
            total_force_error += force_error
            num_batches += 1
    
    avg_energy_error = total_energy_error / num_batches
    avg_force_error = total_force_error / num_batches
    
    print(f"Average Energy MAE: {avg_energy_error:.6f}")
    print(f"Average Force MAE: {avg_force_error:.6f}")
    
    print("\nTraining completed successfully!")
    return model


def model_prediction_example():
    """
    Example of using a trained model for prediction.
    """
    print("\nModel Prediction Example")
    print("=" * 30)
    
    # Create a simple model
    device = get_device()
    model = EquivariantNet(
        hidden_dim=64,
        num_layers=2,
        cutoff_radius=5.0
    ).to(device)
    
    # Create dummy molecule
    num_atoms = 10
    positions = torch.randn(num_atoms, 3, device=device) * 3.0
    atomic_numbers = torch.randint(1, 7, (num_atoms,), device=device)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
    
    data = {
        'pos': positions,
        'atomic_numbers': atomic_numbers,
        'batch': batch
    }
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        energy, forces = model(data)
    
    print(f"Predicted energy: {energy.item():.6f}")
    print(f"Force magnitude (avg): {forces.norm(dim=-1).mean().item():.6f}")
    print(f"Force shape: {forces.shape}")
    
    # Test equivariance
    print("\nTesting equivariance...")
    
    # Random rotation matrix
    angle = torch.rand(1) * 2 * np.pi
    rotation_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Rotate positions
    rotated_positions = torch.matmul(positions, rotation_matrix.T)
    rotated_data = data.copy()
    rotated_data['pos'] = rotated_positions
    
    # Predict on rotated molecule
    with torch.no_grad():
        rotated_energy, rotated_forces = model(rotated_data)
    
    # Check energy invariance
    energy_diff = torch.abs(energy - rotated_energy).item()
    print(f"Energy difference after rotation: {energy_diff:.8f}")
    
    # Check force equivariance
    expected_rotated_forces = torch.matmul(forces, rotation_matrix.T)
    force_diff = torch.abs(rotated_forces - expected_rotated_forces).mean().item()
    print(f"Force equivariance error: {force_diff:.8f}")
    
    if energy_diff < 1e-5 and force_diff < 1e-3:
        print("✓ Model passes basic equivariance test!")
    else:
        print("⚠ Model may have equivariance issues")


if __name__ == '__main__':
    # Run simple training example
    trained_model = simple_training_example()
    
    # Run prediction example
    model_prediction_example()
    
    print("\n" + "=" * 50)
    print("MLPot examples completed successfully!")
    print("Check the documentation for more advanced usage.")
    print("=" * 50)
