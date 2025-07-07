"""
Evaluation script for trained molecular potential models.
This example demonstrates how to evaluate and benchmark trained models.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add mlpot to path
sys.path.append(str(Path(__file__).parent.parent))

from mlpot.models.equivariant_net import EquivariantNet
from mlpot.data.dataset import MolecularDataset, DataNormalizer, create_dataloader
from mlpot.utils.metrics import EnergyForceMetrics, ModelEvaluator, BenchmarkSuite
from mlpot.utils.helpers import Logger, get_device
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Trained Molecular Potential Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to model config file')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for data')
    parser.add_argument('--test_datasets', nargs='+', default=['test_dataset'], 
                       help='Test dataset names')
    parser.add_argument('--data_format', type=str, default='lmdb', 
                       choices=['lmdb', 'pickle', 'npz'], help='Data format')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Output directory')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save model predictions')
    parser.add_argument('--create_plots', action='store_true', 
                       help='Create evaluation plots')
    
    return parser.parse_args()


def load_model_and_config(model_path, config_path, device, logger):
    """Load trained model and configuration."""
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load config
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif 'args' in checkpoint:
        config = checkpoint['args']
    else:
        # Use default config
        logger.warning("No config found, using default parameters")
        config = {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_radial_basis': 128,
            'cutoff_radius': 6.0,
            'max_neighbors': 20,
            'use_pbc': False
        }
    
    # Create model
    model = EquivariantNet(
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 4),
        num_radial_basis=config.get('num_radial_basis', 128),
        cutoff_radius=config.get('cutoff_radius', 6.0),
        max_neighbors=config.get('max_neighbors', 20),
        use_periodic_boundary=config.get('use_pbc', False),
        online_graph_construction=True
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, config


def setup_datasets(args, config, logger):
    """Setup test datasets."""
    logger.info("Setting up test datasets...")
    
    datasets = {}
    dataloaders = {}
    
    for dataset_name in args.test_datasets:
        dataset_path = os.path.join(args.data_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        
        dataset = MolecularDataset(
            data_path=dataset_path,
            format_type=args.data_format,
            use_pbc=config.get('use_pbc', False),
            cutoff=config.get('cutoff_radius', 6.0),
            max_neighbors=config.get('max_neighbors', 20)
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            online_graph_construction=True
        )
        
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = dataloader
        
        logger.info(f"Loaded {dataset_name}: {len(dataset)} samples")
    
    return datasets, dataloaders


def setup_normalizer(args, logger):
    """Setup data normalizer."""
    normalizer = DataNormalizer(method='zscore')
    
    # Load cached statistics
    stats_path = os.path.join(args.data_root, 'statistics.pkl')
    if os.path.exists(stats_path):
        logger.info("Loading dataset statistics...")
        import pickle
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        normalizer.statistics = stats
        normalizer.fitted = True
    else:
        logger.warning("No cached statistics found, normalization may be incorrect")
        normalizer = None
    
    return normalizer


def evaluate_single_dataset(model, dataloader, normalizer, device, dataset_name, logger):
    """Evaluate model on a single dataset."""
    logger.info(f"Evaluating on {dataset_name}...")
    
    evaluator = ModelEvaluator(model, device)
    
    # Collect predictions for detailed analysis
    all_predictions = {
        'energy_pred': [],
        'energy_true': [],
        'force_pred': [],
        'force_true': [],
        'num_atoms': []
    }
    
    metrics = EnergyForceMetrics()
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                # Predict
                pred_energy, pred_forces = model(batch)
                true_energy = batch['energy']
                true_forces = batch['forces']
                
                # Denormalize if needed
                if normalizer is not None:
                    pred_energy = normalizer.denormalize_energy(pred_energy)
                    pred_forces = normalizer.denormalize_forces(pred_forces)
                
                # Update metrics
                metrics.update(
                    pred_energy, true_energy,
                    pred_forces, true_forces,
                    batch.get('natoms', None)
                )
                
                # Store predictions
                all_predictions['energy_pred'].extend(pred_energy.cpu().numpy())
                all_predictions['energy_true'].extend(true_energy.cpu().numpy())
                all_predictions['force_pred'].extend(pred_forces.cpu().numpy().flatten())
                all_predictions['force_true'].extend(true_forces.cpu().numpy().flatten())
                
                if 'natoms' in batch:
                    all_predictions['num_atoms'].extend(batch['natoms'].cpu().numpy())
                
                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx + 1} batches")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Compute final metrics
    final_metrics = metrics.compute_all_metrics()
    error_stats = metrics.compute_error_statistics()
    
    # Convert predictions to numpy arrays
    for key in all_predictions:
        all_predictions[key] = np.array(all_predictions[key])
    
    return final_metrics, error_stats, all_predictions


def create_evaluation_plots(predictions, metrics, output_dir, dataset_name):
    """Create evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Energy parity plot
    plt.figure(figsize=(8, 6))
    energy_pred = predictions['energy_pred']
    energy_true = predictions['energy_true']
    
    plt.scatter(energy_true, energy_pred, alpha=0.6, s=1)
    
    # Perfect prediction line
    min_val = min(energy_true.min(), energy_pred.min())
    max_val = max(energy_true.max(), energy_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.xlabel('True Energy (eV)')
    plt.ylabel('Predicted Energy (eV)')
    plt.title(f'Energy Parity Plot - {dataset_name}\nMAE: {metrics["energy_mae"]:.4f} eV, R²: {metrics["energy_r2"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'{dataset_name}_energy_parity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Force parity plot
    plt.figure(figsize=(8, 6))
    force_pred = predictions['force_pred']
    force_true = predictions['force_true']
    
    # Sample points for visualization if too many
    if len(force_pred) > 50000:
        indices = np.random.choice(len(force_pred), 50000, replace=False)
        force_pred_sample = force_pred[indices]
        force_true_sample = force_true[indices]
    else:
        force_pred_sample = force_pred
        force_true_sample = force_true
    
    plt.scatter(force_true_sample, force_pred_sample, alpha=0.6, s=0.5)
    
    # Perfect prediction line
    min_val = min(force_true.min(), force_pred.min())
    max_val = max(force_true.max(), force_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.xlabel('True Force (eV/Å)')
    plt.ylabel('Predicted Force (eV/Å)')
    plt.title(f'Force Parity Plot - {dataset_name}\nMAE: {metrics["force_mae"]:.4f} eV/Å, R²: {metrics["force_r2"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'{dataset_name}_force_parity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy error distribution
    energy_errors = energy_pred - energy_true
    ax1.hist(energy_errors, bins=50, alpha=0.7, density=True)
    ax1.axvline(0, color='r', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Energy Error (eV)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Energy Error Distribution\nMean: {energy_errors.mean():.4f}, Std: {energy_errors.std():.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Force error distribution
    force_errors = force_pred - force_true
    ax2.hist(force_errors, bins=50, alpha=0.7, density=True)
    ax2.axvline(0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Force Error (eV/Å)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Force Error Distribution\nMean: {force_errors.mean():.4f}, Std: {force_errors.std():.4f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = Logger('eval', log_file=str(output_dir / 'evaluation.log'))
    logger.info("Starting evaluation...")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        model, config = load_model_and_config(
            args.model_path, args.config_path, device, logger
        )
        
        # Setup datasets
        datasets, dataloaders = setup_datasets(args, config, logger)
        
        if not dataloaders:
            logger.error("No valid datasets found!")
            return
        
        # Setup normalizer
        normalizer = setup_normalizer(args, logger)
        
        # Run evaluation on each dataset
        all_results = {}
        all_predictions = {}
        
        for dataset_name, dataloader in dataloaders.items():
            metrics, error_stats, predictions = evaluate_single_dataset(
                model, dataloader, normalizer, device, dataset_name, logger
            )
            
            all_results[dataset_name] = {
                'metrics': metrics,
                'error_stats': error_stats
            }
            all_predictions[dataset_name] = predictions
            
            # Log results
            logger.info(f"\nResults for {dataset_name}:")
            logger.info(f"  Energy MAE: {metrics['energy_mae']:.6f}")
            logger.info(f"  Force MAE: {metrics['force_mae']:.6f}")
            logger.info(f"  Energy RMSE: {metrics['energy_rmse']:.6f}")
            logger.info(f"  Force RMSE: {metrics['force_rmse']:.6f}")
            logger.info(f"  Energy R²: {metrics['energy_r2']:.6f}")
            logger.info(f"  Force R²: {metrics['force_r2']:.6f}")
            
            # Create plots if requested
            if args.create_plots:
                create_evaluation_plots(
                    predictions, metrics, output_dir, dataset_name
                )
                logger.info(f"Plots saved for {dataset_name}")
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save predictions if requested
        if args.save_predictions:
            import pickle
            with open(output_dir / 'predictions.pkl', 'wb') as f:
                pickle.dump(all_predictions, f)
            logger.info("Predictions saved")
        
        # Create summary report
        summary_lines = ["# Evaluation Summary\n"]
        for dataset_name, results in all_results.items():
            metrics = results['metrics']
            summary_lines.extend([
                f"## {dataset_name}\n",
                f"- Energy MAE: {metrics['energy_mae']:.6f} eV\n",
                f"- Force MAE: {metrics['force_mae']:.6f} eV/Å\n",
                f"- Energy RMSE: {metrics['energy_rmse']:.6f} eV\n",
                f"- Force RMSE: {metrics['force_rmse']:.6f} eV/Å\n",
                f"- Energy R²: {metrics['energy_r2']:.6f}\n",
                f"- Force R²: {metrics['force_r2']:.6f}\n",
                f"- Samples: {metrics['num_energy_samples']}\n\n"
            ])
        
        with open(output_dir / 'summary.md', 'w') as f:
            f.writelines(summary_lines)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
