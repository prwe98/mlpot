"""
Evaluation metrics for molecular potential models.
This module provides comprehensive metrics for assessing model performance.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


class EnergyForceMetrics:
    """
    Comprehensive metrics for energy and force predictions.
    """
    
    def __init__(self, energy_units: str = 'eV', force_units: str = 'eV/Å'):
        self.energy_units = energy_units
        self.force_units = force_units
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.energy_predictions = []
        self.energy_targets = []
        self.force_predictions = []
        self.force_targets = []
        self.num_atoms_list = []
    
    def update(
        self,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        force_pred: torch.Tensor,
        force_target: torch.Tensor,
        num_atoms: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new predictions and targets.
        
        Args:
            energy_pred: Predicted energies [batch_size]
            energy_target: Target energies [batch_size]
            force_pred: Predicted forces [num_atoms, 3]
            force_target: Target forces [num_atoms, 3]
            num_atoms: Number of atoms per molecule [batch_size]
        """
        # Convert to numpy for metric computation
        energy_pred_np = energy_pred.detach().cpu().numpy()
        energy_target_np = energy_target.detach().cpu().numpy()
        force_pred_np = force_pred.detach().cpu().numpy()
        force_target_np = force_target.detach().cpu().numpy()
        
        # Store predictions and targets
        self.energy_predictions.extend(energy_pred_np.flatten())
        self.energy_targets.extend(energy_target_np.flatten())
        self.force_predictions.extend(force_pred_np.flatten())
        self.force_targets.extend(force_target_np.flatten())
        
        if num_atoms is not None:
            self.num_atoms_list.extend(num_atoms.detach().cpu().numpy())
    
    def compute_energy_metrics(self) -> Dict[str, float]:
        """Compute energy-specific metrics."""
        if not self.energy_predictions:
            return {}
        
        pred = np.array(self.energy_predictions)
        target = np.array(self.energy_targets)
        
        metrics = {
            'energy_mae': mean_absolute_error(target, pred),
            'energy_rmse': np.sqrt(mean_squared_error(target, pred)),
            'energy_mse': mean_squared_error(target, pred),
            'energy_r2': r2_score(target, pred),
            'energy_mean_error': np.mean(pred - target),
            'energy_std_error': np.std(pred - target),
            'energy_max_error': np.max(np.abs(pred - target))
        }
        
        return metrics
    
    def compute_force_metrics(self) -> Dict[str, float]:
        """Compute force-specific metrics."""
        if not self.force_predictions:
            return {}
        
        pred = np.array(self.force_predictions)
        target = np.array(self.force_targets)
        
        metrics = {
            'force_mae': mean_absolute_error(target, pred),
            'force_rmse': np.sqrt(mean_squared_error(target, pred)),
            'force_mse': mean_squared_error(target, pred),
            'force_r2': r2_score(target, pred),
            'force_mean_error': np.mean(pred - target),
            'force_std_error': np.std(pred - target),
            'force_max_error': np.max(np.abs(pred - target))
        }
        
        return metrics
    
    def compute_per_atom_energy_metrics(self) -> Dict[str, float]:
        """Compute per-atom energy metrics."""
        if not self.energy_predictions or not self.num_atoms_list:
            return {}
        
        energy_pred = np.array(self.energy_predictions)
        energy_target = np.array(self.energy_targets)
        num_atoms = np.array(self.num_atoms_list)
        
        # Compute per-atom energies
        per_atom_pred = energy_pred / num_atoms
        per_atom_target = energy_target / num_atoms
        
        metrics = {
            'per_atom_energy_mae': mean_absolute_error(per_atom_target, per_atom_pred),
            'per_atom_energy_rmse': np.sqrt(mean_squared_error(per_atom_target, per_atom_pred)),
            'per_atom_energy_r2': r2_score(per_atom_target, per_atom_pred)
        }
        
        return metrics
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {}
        
        metrics.update(self.compute_energy_metrics())
        metrics.update(self.compute_force_metrics())
        metrics.update(self.compute_per_atom_energy_metrics())
        
        # Add sample counts
        metrics['num_energy_samples'] = len(self.energy_predictions)
        metrics['num_force_samples'] = len(self.force_predictions)
        
        return metrics
    
    def compute_error_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute detailed error statistics."""
        stats = {}
        
        if self.energy_predictions:
            energy_errors = np.array(self.energy_predictions) - np.array(self.energy_targets)
            stats['energy_errors'] = {
                'mean': float(np.mean(energy_errors)),
                'std': float(np.std(energy_errors)),
                'min': float(np.min(energy_errors)),
                'max': float(np.max(energy_errors)),
                'q25': float(np.percentile(energy_errors, 25)),
                'q50': float(np.percentile(energy_errors, 50)),
                'q75': float(np.percentile(energy_errors, 75)),
                'q95': float(np.percentile(energy_errors, 95)),
                'q99': float(np.percentile(energy_errors, 99))
            }
        
        if self.force_predictions:
            force_errors = np.array(self.force_predictions) - np.array(self.force_targets)
            stats['force_errors'] = {
                'mean': float(np.mean(force_errors)),
                'std': float(np.std(force_errors)),
                'min': float(np.min(force_errors)),
                'max': float(np.max(force_errors)),
                'q25': float(np.percentile(force_errors, 25)),
                'q50': float(np.percentile(force_errors, 50)),
                'q75': float(np.percentile(force_errors, 75)),
                'q95': float(np.percentile(force_errors, 95)),
                'q99': float(np.percentile(force_errors, 99))
            }
        
        return stats


class ModelEvaluator:
    """
    High-level model evaluation interface.
    """
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataloader,
        metrics: Optional[EnergyForceMetrics] = None,
        normalizer = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            metrics: Metrics object to accumulate results
            normalizer: Data normalizer for denormalization
            
        Returns:
            Dictionary of computed metrics
        """
        if metrics is None:
            metrics = EnergyForceMetrics()
        else:
            metrics.reset()
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Model prediction
                    pred_energy, pred_forces = self.model(batch)
                    
                    # Get targets
                    target_energy = batch['energy']
                    target_forces = batch['forces']
                    
                    # Denormalize if normalizer provided
                    if normalizer is not None:
                        pred_energy = normalizer.denormalize_energy(pred_energy)
                        pred_forces = normalizer.denormalize_forces(pred_forces)
                    
                    # Update metrics
                    metrics.update(
                        pred_energy, target_energy,
                        pred_forces, target_forces,
                        batch.get('natoms', None)
                    )
                    
                except Exception as e:
                    warnings.warn(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        return metrics.compute_all_metrics()
    
    def predict_single(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction for a single molecular configuration.
        """
        self.model.eval()
        
        # Move data to device
        data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in data.items()}
        
        with torch.no_grad():
            energy, forces = self.model(data)
        
        return energy, forces
    
    def compute_energy_conservation(
        self,
        trajectory: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute energy conservation metrics for a trajectory.
        """
        energies = []
        
        for frame in trajectory:
            energy, _ = self.predict_single(frame)
            energies.append(energy.item())
        
        energies = np.array(energies)
        
        return {
            'initial_energy': energies[0],
            'final_energy': energies[-1],
            'energy_drift': energies[-1] - energies[0],
            'energy_variance': np.var(energies),
            'max_energy_deviation': np.max(np.abs(energies - energies[0])),
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies)
        }


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for molecular potential models.
    """
    
    def __init__(self, model, device: str = 'cuda'):
        self.evaluator = ModelEvaluator(model, device)
        self.results = {}
    
    def run_benchmark(
        self,
        datasets: Dict[str, any],  # DataLoader objects
        normalizers: Optional[Dict[str, any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive benchmark on multiple datasets.
        """
        normalizers = normalizers or {}
        
        for dataset_name, dataloader in datasets.items():
            print(f"Evaluating on {dataset_name}...")
            
            normalizer = normalizers.get(dataset_name, None)
            metrics = self.evaluator.evaluate_dataset(
                dataloader, normalizer=normalizer
            )
            
            self.results[dataset_name] = metrics
            
            # Print summary
            print(f"Results for {dataset_name}:")
            print(f"  Energy MAE: {metrics.get('energy_mae', 'N/A'):.6f}")
            print(f"  Force MAE: {metrics.get('force_mae', 'N/A'):.6f}")
            print(f"  Energy R²: {metrics.get('energy_r2', 'N/A'):.6f}")
            print(f"  Force R²: {metrics.get('force_r2', 'N/A'):.6f}")
            print()
        
        # Save results if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        return self.results
    
    def compare_models(
        self,
        other_results: Dict[str, Dict[str, float]],
        metrics_to_compare: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare current model results with other model results.
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['energy_mae', 'force_mae', 'energy_r2', 'force_r2']
        
        comparison = {}
        
        for dataset_name in self.results.keys():
            if dataset_name not in other_results:
                continue
            
            comparison[dataset_name] = {}
            
            for metric in metrics_to_compare:
                if metric in self.results[dataset_name] and metric in other_results[dataset_name]:
                    current_value = self.results[dataset_name][metric]
                    other_value = other_results[dataset_name][metric]
                    
                    # For error metrics (lower is better), compute relative improvement
                    if 'mae' in metric or 'mse' in metric or 'rmse' in metric:
                        improvement = (other_value - current_value) / other_value * 100
                    else:  # For R² and similar (higher is better)
                        improvement = (current_value - other_value) / other_value * 100
                    
                    comparison[dataset_name][f'{metric}_improvement'] = improvement
        
        return comparison
