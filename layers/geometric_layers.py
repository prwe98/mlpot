"""
Geometric layers and operations for equivariant neural networks.
This module contains fundamental building blocks for geometry-aware deep learning.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch_geometric.nn.models.schnet import GaussianSmearing


class ScaledActivation(nn.Module):
    """
    Scaled SiLU activation function.
    Provides better gradient flow for deep networks.
    """
    
    def __init__(self):
        super(ScaledActivation, self).__init__()
        self.silu = nn.SiLU()
        self.scale = 1.0 / 0.6  # Approximate scale factor for SiLU
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(x) * self.scale


class AtomicEmbedding(nn.Module):
    """
    Learnable atomic embeddings for different elements.
    Maps atomic numbers to dense feature representations.
    """
    
    def __init__(self, embedding_dim: int, max_atomic_number: int = 83):
        super(AtomicEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_atomic_number, embedding_dim)
        
        # Initialize with small random values
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        return self.embedding(atomic_numbers)


class PolynomialEnvelope(nn.Module):
    """
    Polynomial envelope function for smooth cutoff.
    Ensures smooth decay of interactions at cutoff radius.
    """
    
    def __init__(self, exponent: int = 5):
        super(PolynomialEnvelope, self).__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
    
    def forward(self, distances_scaled: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial envelope to scaled distances (d/cutoff).
        """
        envelope_values = (
            1 + 
            self.a * distances_scaled**self.p +
            self.b * distances_scaled**(self.p + 1) +
            self.c * distances_scaled**(self.p + 2)
        )
        
        return torch.where(
            distances_scaled < 1,
            envelope_values,
            torch.zeros_like(distances_scaled)
        )


class ExponentialEnvelope(nn.Module):
    """
    Exponential envelope function for smooth cutoff.
    Alternative to polynomial envelope with different decay characteristics.
    """
    
    def __init__(self):
        super(ExponentialEnvelope, self).__init__()
    
    def forward(self, distances_scaled: torch.Tensor) -> torch.Tensor:
        """
        Apply exponential envelope to scaled distances.
        """
        envelope_values = torch.exp(
            -(distances_scaled**2) / ((1 - distances_scaled) * (1 + distances_scaled))
        )
        
        return torch.where(
            distances_scaled < 1,
            envelope_values,
            torch.zeros_like(distances_scaled)
        )


class SphericalBesselBasis(nn.Module):
    """
    Spherical Bessel basis functions for radial encoding.
    Provides orthogonal basis for distance representation.
    """
    
    def __init__(self, num_radial: int, cutoff: float):
        super(SphericalBesselBasis, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # Precompute Bessel function zeros
        # For l=0 spherical Bessel functions
        bessel_zeros = torch.arange(1, num_radial + 1) * math.pi
        self.register_buffer('bessel_zeros', bessel_zeros)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical Bessel basis functions.
        """
        # Normalize distances
        normalized_distances = distances / self.cutoff
        
        # Compute spherical Bessel functions
        # j_0(n*pi*r) = sin(n*pi*r) / (n*pi*r)
        arguments = self.bessel_zeros.unsqueeze(0) * normalized_distances.unsqueeze(1)
        
        # Handle r=0 case
        safe_arguments = torch.where(arguments != 0, arguments, torch.ones_like(arguments))
        bessel_values = torch.sin(safe_arguments) / safe_arguments
        
        # Fix r=0 case (limit is 1 for n=1, 0 for n>1)
        mask = (arguments == 0)
        bessel_values = torch.where(mask, torch.zeros_like(bessel_values), bessel_values)
        bessel_values[:, 0] = torch.where(
            normalized_distances == 0, 
            torch.ones_like(normalized_distances), 
            bessel_values[:, 0]
        )
        
        return bessel_values


class RadialBasisFunction(nn.Module):
    """
    Radial basis function layer combining basis functions with envelope.
    Encodes interatomic distances into learnable representations.
    """
    
    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf_config: Dict[str, Any] = None,
        envelope_config: Dict[str, Any] = None
    ):
        super(RadialBasisFunction, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # Initialize radial basis functions
        rbf_config = rbf_config or {"name": "gaussian"}
        rbf_type = rbf_config.get("name", "gaussian")
        
        if rbf_type == "gaussian":
            self.radial_basis = GaussianSmearing(
                start=0.0,
                stop=cutoff,
                num_gaussians=num_radial
            )
        elif rbf_type == "bessel":
            self.radial_basis = SphericalBesselBasis(num_radial, cutoff)
        else:
            raise ValueError(f"Unknown RBF type: {rbf_type}")
        
        # Initialize envelope function
        envelope_config = envelope_config or {"name": "polynomial", "exponent": 5}
        envelope_type = envelope_config.get("name", "polynomial")
        
        if envelope_type == "polynomial":
            exponent = envelope_config.get("exponent", 5)
            self.envelope = PolynomialEnvelope(exponent)
        elif envelope_type == "exponential":
            self.envelope = ExponentialEnvelope()
        else:
            raise ValueError(f"Unknown envelope type: {envelope_type}")
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis functions with envelope.
        """
        # Compute basis functions
        basis_values = self.radial_basis(distances)
        
        # Apply envelope
        scaled_distances = distances / self.cutoff
        envelope_values = self.envelope(scaled_distances)
        
        # Combine basis and envelope
        return basis_values * envelope_values.unsqueeze(-1)


class EquivariantLinear(nn.Module):
    """
    Equivariant linear layer for vector features.
    Maintains equivariance under rotations.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(EquivariantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply equivariant linear transformation.
        Input shape: (N, 3, in_features)
        Output shape: (N, 3, out_features)
        """
        # Apply linear transformation along feature dimension
        output = torch.einsum('ndi,oi->ndo', x, self.weight)
        
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(0)
        
        return output


class VectorNorm(nn.Module):
    """
    Compute norms of vector features.
    Converts vector features to scalar invariants.
    """
    
    def __init__(self, dim: int = -2, keepdim: bool = False, eps: float = 1e-8):
        super(VectorNorm, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vector norms.
        Input shape: (..., 3, features)
        Output shape: (..., features) or (..., 1, features) if keepdim=True
        """
        return torch.sqrt(torch.sum(x**2, dim=self.dim, keepdim=self.keepdim) + self.eps)


class GatedEquivariantBlock(nn.Module):
    """
    Gated equivariant block combining scalar and vector features.
    Allows controlled mixing of scalar and vector information.
    """
    
    def __init__(self, scalar_features: int, vector_features: int):
        super(GatedEquivariantBlock, self).__init__()
        self.scalar_features = scalar_features
        self.vector_features = vector_features
        
        # Scalar processing
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_features + vector_features, scalar_features),
            ScaledActivation(),
            nn.Linear(scalar_features, scalar_features * 2)
        )
        
        # Vector processing
        self.vector_net = EquivariantLinear(vector_features, vector_features, bias=False)
        
        # Norm computation
        self.vector_norm = VectorNorm(dim=-2)
    
    def forward(self, scalar_input: torch.Tensor, 
                vector_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process scalar and vector features with gating.
        """
        # Compute vector norms
        vector_norms = self.vector_norm(vector_input)
        
        # Process scalars with vector norm information
        combined_input = torch.cat([scalar_input, vector_norms], dim=-1)
        scalar_output = self.scalar_net(combined_input)
        
        # Split scalar output into update and gate
        scalar_update, gate = torch.split(scalar_output, self.scalar_features, dim=-1)
        gate = torch.sigmoid(gate)
        
        # Apply gating to scalar features
        gated_scalars = scalar_input + gate * scalar_update
        
        # Process vector features
        vector_output = self.vector_net(vector_input)
        
        return gated_scalars, vector_output
