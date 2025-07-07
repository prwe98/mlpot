"""
Training script for molecular potential models.
This example demonstrates how to train an equivariant neural network on molecular data.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import sys

# Add mlpot to path
sys.path.append(str(Path(__file__).parent.parent))

from mlpot.models.equivariant_net import EquivariantNet
from mlpot.core.trainer import PotentialTrainer
from mlpot.data.dataset import MolecularDataset, DataNormalizer, create_dataloader
from mlpot.utils.helpers import set_random_seed, get_device, ConfigManager, Logger
from mlpot.utils.metrics import EnergyForceMetrics, ModelEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Equivariant Neural Network for Molecular Potentials')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for data')
    parser.add_argument('--train_data', type=str, default='train_dataset', help='Training dataset path')
    parser.add_argument('--val_data', type=str, default='valid_dataset', help='Validation dataset path')
    parser.add_argument('--data_format', type=str, default='lmdb', choices=['lmdb', 'pickle', 'npz'], 
                       help='Data format')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_radial_basis', type=int, default=128, help='Number of radial basis functions')
    parser.add_argument('--cutoff_radius', type=float, default=6.0, help='Cutoff radius')
    parser.add_argument('--max_neighbors', type=int, default=20, help='Maximum neighbors')
    parser.add_argument('--use_pbc', action='store_true', help='Use periodic boundary conditions')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--energy_weight', type=float, default=1.0, help='Energy loss weight')
    parser.add_argument('--force_weight', type=float, default=50.0, help='Force loss weight')
    parser.add_argument('--gradient_clip', type=float, default=100.0, help='Gradient clipping value')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--save_frequency', type=int, default=50, help='Save checkpoint frequency')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logging frequency')
    
    return parser.parse_args()


def setup_data(args, logger):
    """Setup datasets and data loaders."""
    logger.info("Setting up datasets...")
    
    # Create dataset paths
    train_path = os.path.join(args.data_root, args.train_data)
    val_path = os.path.join(args.data_root, args.val_data)
    
    # Create datasets
    train_dataset = MolecularDataset(
        data_path=train_path,
        format_type=args.data_format,
        use_pbc=args.use_pbc,
        cutoff=args.cutoff_radius,
        max_neighbors=args.max_neighbors
    )
    
    val_dataset = MolecularDataset(
        data_path=val_path,
        format_type=args.data_format,
        use_pbc=args.use_pbc,
        cutoff=args.cutoff_radius,
        max_neighbors=args.max_neighbors
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Setup data normalizer
    normalizer = DataNormalizer(method='zscore')
    
    # Check for cached statistics
    stats_path = os.path.join(args.data_root, 'statistics.pkl')
    if os.path.exists(stats_path):
        logger.info("Loading cached dataset statistics...")
        import pickle
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        normalizer.statistics = stats
        normalizer.fitted = True
    else:
        logger.info("Computing dataset statistics...")
        normalizer.fit(train_dataset)
        # Cache statistics
        import pickle
        with open(stats_path, 'wb') as f:
            pickle.dump(normalizer.statistics, f)
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        online_graph_construction=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        online_graph_construction=True
    )
    
    return train_loader, val_loader, normalizer


def setup_model(args, device, logger):
    """Setup the model."""
    logger.info("Setting up model...")
    
    model = EquivariantNet(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_radial_basis=args.num_radial_basis,
        cutoff_radius=args.cutoff_radius,
        max_neighbors=args.max_neighbors,
        predict_forces=True,
        direct_force_prediction=True,
        use_periodic_boundary=args.use_pbc,
        online_graph_construction=True
    ).to(device)
    
    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {param_count:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_count:,}")
    
    return model


def setup_training(model, args, logger):
    """Setup optimizer, scheduler, and trainer."""
    logger.info("Setting up training components...")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )
    
    # Loss configuration
    loss_config = {
        'energy_weight': args.energy_weight,
        'force_weight': args.force_weight,
        'loss_type': 'l1'
    }
    
    # Trainer
    trainer = PotentialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_config=loss_config,
        device=args.device,
        gradient_clip=args.gradient_clip
    )
    
    return trainer, optimizer, scheduler


def run_evaluation(model, val_loader, normalizer, device, logger):
    """Run model evaluation."""
    logger.info("Running evaluation...")
    
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate_dataset(val_loader, normalizer=normalizer)
    
    # Log key metrics
    logger.info("Evaluation Results:")
    logger.info(f"  Energy MAE: {metrics.get('energy_mae', 'N/A'):.6f}")
    logger.info(f"  Force MAE: {metrics.get('force_mae', 'N/A'):.6f}")
    logger.info(f"  Energy RMSE: {metrics.get('energy_rmse', 'N/A'):.6f}")
    logger.info(f"  Force RMSE: {metrics.get('force_rmse', 'N/A'):.6f}")
    logger.info(f"  Energy R²: {metrics.get('energy_r2', 'N/A'):.6f}")
    logger.info(f"  Force R²: {metrics.get('force_r2', 'N/A'):.6f}")
    
    return metrics


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = Logger('train', log_file=str(output_dir / 'training.log'))
    logger.info("Starting training...")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_manager = ConfigManager(vars(args))
    config_manager.save_to_file(str(output_dir / 'config.yaml'))
    
    try:
        # Setup data
        train_loader, val_loader, normalizer = setup_data(args, logger)
        
        # Setup model
        model = setup_model(args, device, logger)
        
        # Setup training
        trainer, optimizer, scheduler = setup_training(model, args, logger)
        
        # Initial evaluation
        logger.info("Running initial evaluation...")
        initial_metrics = run_evaluation(model, val_loader, normalizer, device, logger)
        
        # Training loop
        logger.info("Starting training loop...")
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            logger.info("-" * 50)
            
            # Train for one epoch
            epoch_metrics = trainer.train_epoch(
                train_loader, val_loader, log_interval=args.log_frequency
            )
            
            # Log epoch results
            train_loss = epoch_metrics['train_total_loss']
            val_loss = epoch_metrics.get('val_total_loss', float('inf'))
            learning_rate = epoch_metrics['learning_rate']
            
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"Learning Rate: {learning_rate:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'best_model.pt'
                trainer.save_checkpoint(
                    str(best_path),
                    epoch=epoch,
                    metrics=epoch_metrics,
                    args=vars(args)
                )
                logger.info(f"New best model saved! Val Loss: {best_val_loss:.6f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % args.save_frequency == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
                trainer.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch,
                    metrics=epoch_metrics,
                    args=vars(args)
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        
        # Load best model
        best_checkpoint = trainer.load_checkpoint(str(output_dir / 'best_model.pt'))
        final_metrics = run_evaluation(model, val_loader, normalizer, device, logger)
        
        # Save final metrics
        import json
        with open(output_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
