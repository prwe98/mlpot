# MLPot Framework

A modular and extensible framework for building equivariant neural network potentials for molecular systems.

## Features

- **Modular Architecture**: Clean separation between models, training, data processing, and utilities
- **Equivariant Neural Networks**: Built-in support for rotation and translation equivariant models
- **Multiple Data Formats**: Support for LMDB, pickle, and NPZ data formats
- **Comprehensive Training**: Full training pipeline with checkpointing, early stopping, and monitoring
- **Evaluation Tools**: Extensive metrics and visualization for model assessment
- **Flexible Configuration**: YAML-based configuration system
- **Production Ready**: Optimized for both research and production environments

## Project Structure

```
mlpot/
├── __init__.py                    # Main package initialization
├── core/                          # Core interfaces and base classes
│   ├── base_model.py             # Abstract base classes for models
│   └── trainer.py                # Training interface and utilities
├── models/                        # Model implementations
│   └── equivariant_net.py        # Equivariant neural network model
├── layers/                        # Reusable layer implementations
│   ├── geometric_layers.py       # Geometric and equivariant layers
│   └── graph_ops.py              # Graph construction and operations
├── data/                          # Data processing and loading
│   └── dataset.py                # Dataset classes and utilities
├── utils/                         # Utilities and helper functions
│   ├── metrics.py                # Evaluation metrics
│   └── helpers.py                # General utilities
└── examples/                      # Example scripts and configurations
    ├── train_potential.py        # Training script
    ├── evaluate_model.py         # Evaluation script
    └── config.yaml               # Configuration file
```

## Core Components

### 1. Base Model Interface (`core/base_model.py`)

Defines abstract interfaces that all molecular potential models must implement:

- `BasePotential`: Main interface for potential models
- `MessagePassingInterface`: Interface for message passing operations
- `EquivarianceInterface`: Interface for equivariant operations

### 2. Equivariant Network (`models/equivariant_net.py`)

Implementation of an equivariant graph neural network based on advanced architectural patterns:

- **GlobalScalarProcessor**: Handles scalar feature aggregation
- **GlobalVectorProcessor**: Handles vector feature aggregation with equivariance
- **EquivariantMessageLayer**: Message passing with equivariant operations
- **EquivariantUpdateLayer**: Feature updates maintaining equivariance

### 3. Geometric Layers (`layers/geometric_layers.py`)

Fundamental building blocks for geometry-aware neural networks:

- `ScaledActivation`: Improved activation functions
- `AtomicEmbedding`: Learnable atomic representations
- `RadialBasisFunction`: Distance encoding with multiple basis types
- `EquivariantLinear`: Linear layers preserving equivariance

### 4. Training Framework (`core/trainer.py`)

Comprehensive training system with:

- Flexible loss computation
- Gradient clipping and normalization
- Learning rate scheduling
- Checkpoint management
- Training history tracking

### 5. Data Processing (`data/dataset.py`)

Robust data handling with:

- Multiple format support (LMDB, pickle, NPZ)
- Automatic graph construction
- Data normalization utilities
- Batch collation for variable-size molecules

### 6. Evaluation Tools (`utils/metrics.py`)

Extensive evaluation capabilities:

- Energy and force metrics
- Error distribution analysis
- Model comparison utilities
- Benchmark suites

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlpot

# Install dependencies
pip install torch torch-geometric torch-scatter
pip install numpy scipy scikit-learn matplotlib
pip install lmdb pyyaml
```

### Basic Usage

#### Training a Model

```python
from mlpot.models.equivariant_net import EquivariantNet
from mlpot.core.trainer import PotentialTrainer
from mlpot.data.dataset import MolecularDataset, create_dataloader

# Create model
model = EquivariantNet(
    hidden_dim=512,
    num_layers=4,
    cutoff_radius=6.0
)

# Setup data
dataset = MolecularDataset(
    data_path='path/to/data.lmdb',
    format_type='lmdb'
)
dataloader = create_dataloader(dataset, batch_size=32)

# Create trainer
trainer = PotentialTrainer(
    model=model,
    optimizer=optimizer,
    loss_config={'energy_weight': 1.0, 'force_weight': 50.0}
)

# Train
trainer.fit(train_loader, val_loader, epochs=100)
```

#### Evaluating a Model

```python
from mlpot.utils.metrics import ModelEvaluator

# Load trained model
evaluator = ModelEvaluator(model, device='cuda')

# Evaluate
metrics = evaluator.evaluate_dataset(test_loader)
print(f"Energy MAE: {metrics['energy_mae']:.6f}")
print(f"Force MAE: {metrics['force_mae']:.6f}")
```

### Command Line Usage

#### Training

```bash
python examples/train_potential.py \
    --data_root /path/to/data \
    --hidden_dim 512 \
    --num_layers 4 \
    --batch_size 64 \
    --epochs 1000 \
    --output_dir ./outputs
```

#### Evaluation

```bash
python examples/evaluate_model.py \
    --model_path ./outputs/best_model.pt \
    --data_root /path/to/data \
    --test_datasets test_dataset \
    --create_plots \
    --output_dir ./evaluation
```

## Model Architecture

### Equivariant Neural Network

The core model implements an equivariant graph neural network with the following key features:

1. **Equivariant Message Passing**: Messages between atoms preserve rotational equivariance
2. **Global Feature Aggregation**: Both scalar and vector features are aggregated globally
3. **Multi-scale Interactions**: Multiple layers capture interactions at different scales
4. **Direct Force Prediction**: Forces are predicted directly without gradient computation

### Key Architectural Components

- **Atomic Embeddings**: Learnable representations for different elements
- **Radial Basis Functions**: Encode interatomic distances
- **Message Layers**: Compute equivariant messages between neighboring atoms
- **Update Layers**: Update node features while maintaining equivariance
- **Global Processors**: Handle molecular-level feature aggregation

## Data Formats

The framework supports multiple data formats:

### LMDB Format
- High-performance key-value storage
- Suitable for large datasets
- Memory-efficient random access

### Pickle Format
- Python native serialization
- Easy to use for small to medium datasets
- Supports complex data structures

### NPZ Format
- NumPy compressed format
- Good for array-heavy data
- Cross-platform compatibility

## Configuration

The framework uses YAML configuration files for easy experimentation:

```yaml
model:
  hidden_dim: 512
  num_layers: 4
  cutoff_radius: 6.0

training:
  batch_size: 64
  learning_rate: 0.0002
  epochs: 1000

loss:
  energy_weight: 1.0
  force_weight: 50.0
```

## Advanced Features

### Periodic Boundary Conditions
```python
model = EquivariantNet(
    use_periodic_boundary=True,
    cutoff_radius=6.0
)
```

### Custom Loss Functions
```python
trainer = PotentialTrainer(
    model=model,
    loss_config={
        'energy_weight': 1.0,
        'force_weight': 50.0,
        'loss_type': 'l1'
    }
)
```

### Data Normalization
```python
normalizer = DataNormalizer(method='zscore')
normalizer.fit(dataset)
```

## Performance Tips

1. **Use LMDB for large datasets** - Better I/O performance
2. **Enable mixed precision** - Faster training on modern GPUs
3. **Tune batch size** - Balance memory usage and convergence
4. **Use multiple workers** - Parallel data loading
5. **Gradient clipping** - Stabilize training

## Extending the Framework

### Adding New Models

```python
from mlpot.core.base_model import BasePotential

class CustomModel(BasePotential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
    
    def forward(self, data):
        # Implement forward pass
        return energy, forces
    
    def get_energy_and_forces(self, data):
        return self.forward(data)
```

### Custom Layers

```python
class CustomLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Initialize layer
    
    def forward(self, x):
        # Implement layer logic
        return output
```

### Custom Metrics

```python
from mlpot.utils.metrics import EnergyForceMetrics

class CustomMetrics(EnergyForceMetrics):
    def compute_custom_metric(self):
        # Implement custom metric
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model size
2. **Slow data loading**: Increase num_workers or use LMDB format
3. **Training instability**: Enable gradient clipping or reduce learning rate
4. **Poor convergence**: Check data normalization and loss weights

### Performance Optimization

1. Use `torch.compile()` for PyTorch 2.0+
2. Enable `torch.backends.cudnn.benchmark = True`
3. Use appropriate data types (float16 for mixed precision)
4. Profile with PyTorch profiler

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mlpot2024,
  title={MLPot: A Modular Framework for Equivariant Neural Network Potentials},
  author={Wu Xiaoyu},
  year={2025},
  url={https://github.com/your-repo/mlpot}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Join our community discussions
