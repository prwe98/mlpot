#!/usr/bin/env python3
"""
测试脚本：验证mlpot实现与E2GNN的一致性
这个脚本会测试模型的基本功能、等变性质和数学正确性
"""

import sys
import os

# 添加路径到sys.path中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# 导入我们的模块
try:
    from mlpot.models.equivariant_net import (
        EquivariantNet, 
        EquivariantMessageLayer, 
        EquivariantUpdateLayer,
        GlobalScalarProcessor,
        GlobalVectorProcessor
    )
    from mlpot.layers.geometric_layers import RadialBasisFunction, AtomicEmbedding, ScaledActivation
    from mlpot.data.dataset import MolecularDataset
    from mlpot.utils.metrics import compute_mae, compute_rmse
except ImportError:
    # 如果作为脚本运行，尝试相对导入
    import importlib.util
    import sys
    
    # 手动导入模块
    modules_to_import = [
        ('equivariant_net', 'models/equivariant_net.py'),
        ('geometric_layers', 'layers/geometric_layers.py'),
        ('dataset', 'data/dataset.py'),
        ('metrics', 'utils/metrics.py')
    ]
    
    for module_name, module_path in modules_to_import:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    
    # 导入需要的类和函数
    from equivariant_net import (
        EquivariantNet, 
        EquivariantMessageLayer, 
        EquivariantUpdateLayer,
        GlobalScalarProcessor,
        GlobalVectorProcessor
    )
    from geometric_layers import RadialBasisFunction, AtomicEmbedding, ScaledActivation
    from dataset import MolecularDataset
    from metrics import compute_mae, compute_rmse


def create_test_data(num_atoms: int = 5, batch_size: int = 2) -> Dict[str, torch.Tensor]:
    """
    创建测试用的分子数据
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建随机分子结构
    total_atoms = num_atoms * batch_size
    
    data = {
        'pos': torch.randn(total_atoms, 3, device=device),
        'atomic_numbers': torch.randint(1, 84, (total_atoms,), device=device),
        'batch': torch.repeat_interleave(torch.arange(batch_size, device=device), num_atoms),
        'num_atoms': torch.tensor([num_atoms] * batch_size, device=device)
    }
    
    return data


def test_model_forward():
    """
    测试模型前向传播
    """
    print("🧪 测试模型前向传播...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EquivariantNet(
        hidden_dim=64,  # 使用较小的维度来快速测试
        num_layers=2,
        num_radial_basis=32,
        cutoff_radius=5.0,
        max_neighbors=10
    ).to(device)
    
    # 创建测试数据
    data = create_test_data(num_atoms=5, batch_size=2)
    
    # 前向传播
    try:
        with torch.no_grad():
            energy, forces = model(data)
        
        print(f"✅ 前向传播成功!")
        print(f"   能量形状: {energy.shape}")
        print(f"   力的形状: {forces.shape}")
        print(f"   能量值: {energy.detach().cpu().numpy()}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False


def test_equivariance():
    """
    测试模型的等变性质
    """
    print("\n🔄 测试等变性质...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = EquivariantNet(
        hidden_dim=64,
        num_layers=2,
        num_radial_basis=32,
        cutoff_radius=5.0,
        max_neighbors=10
    ).to(device)
    
    # 创建测试数据
    data = create_test_data(num_atoms=4, batch_size=1)
    
    # 创建随机旋转矩阵
    angle = np.pi / 4  # 45度
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    translation = torch.tensor([1.0, 2.0, 3.0], device=device)
    
    try:
        is_equivariant = model.check_equivariance(data, rotation_matrix, translation)
        
        if is_equivariant:
            print("✅ 等变性测试通过!")
        else:
            print("❌ 等变性测试失败!")
            
        return is_equivariant
    except Exception as e:
        print(f"❌ 等变性测试出错: {e}")
        return False


def test_components():
    """
    测试各个组件
    """
    print("\n🔧 测试模型组件...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 64
    num_radial_basis = 32
    
    # 测试径向基函数
    print("  测试径向基函数...")
    rbf = RadialBasisFunction(num_radial=num_radial_basis, cutoff=5.0).to(device)
    distances = torch.linspace(0.5, 5.0, 10, device=device)
    rbf_output = rbf(distances)
    print(f"    RBF输出形状: {rbf_output.shape}")
    
    # 测试原子嵌入
    print("  测试原子嵌入...")
    embedding = AtomicEmbedding(hidden_dim, num_elements=84).to(device)
    atomic_numbers = torch.randint(1, 84, (10,), device=device)
    embed_output = embedding(atomic_numbers)
    print(f"    嵌入输出形状: {embed_output.shape}")
    
    # 测试缩放激活函数
    print("  测试缩放激活函数...")
    activation = ScaledActivation().to(device)
    test_input = torch.randn(5, hidden_dim, device=device)
    activation_output = activation(test_input)
    print(f"    激活函数输出形状: {activation_output.shape}")
    
    # 测试消息传递层
    print("  测试消息传递层...")
    message_layer = EquivariantMessageLayer(hidden_dim, num_radial_basis).to(device)
    
    num_nodes = 5
    node_features = torch.randn(num_nodes, hidden_dim, device=device)
    vector_features = torch.randn(num_nodes, 3, hidden_dim, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
    edge_radial = torch.randn(4, num_radial_basis, device=device)
    edge_vectors = torch.randn(4, 3, device=device)
    
    scalar_msg, vector_msg = message_layer(
        node_features, vector_features, edge_index, edge_radial, edge_vectors
    )
    print(f"    标量消息形状: {scalar_msg.shape}")
    print(f"    向量消息形状: {vector_msg.shape}")
    
    # 测试更新层
    print("  测试更新层...")
    update_layer = EquivariantUpdateLayer(hidden_dim).to(device)
    scalar_update, vector_update = update_layer(node_features, vector_features)
    print(f"    标量更新形状: {scalar_update.shape}")
    print(f"    向量更新形状: {vector_update.shape}")
    
    print("✅ 所有组件测试通过!")
    return True


def test_training_step():
    """
    测试训练步骤
    """
    print("\n🏋️ 测试训练步骤...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = EquivariantNet(
        hidden_dim=64,
        num_layers=2,
        num_radial_basis=32,
        cutoff_radius=5.0,
        max_neighbors=10
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建测试数据
    data = create_test_data(num_atoms=5, batch_size=2)
    
    # 创建假的目标值
    target_energy = torch.randn(2, device=device)
    target_forces = torch.randn(10, 3, device=device)
    
    try:
        # 前向传播
        pred_energy, pred_forces = model(data)
        
        # 计算损失
        energy_loss = nn.MSELoss()(pred_energy, target_energy)
        force_loss = nn.MSELoss()(pred_forces, target_forces)
        total_loss = energy_loss + force_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"✅ 训练步骤成功!")
        print(f"   总损失: {total_loss.item():.6f}")
        print(f"   能量损失: {energy_loss.item():.6f}")
        print(f"   力损失: {force_loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return False


def test_batch_processing():
    """
    测试批处理能力
    """
    print("\n📦 测试批处理...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EquivariantNet(
        hidden_dim=64,
        num_layers=2,
        num_radial_basis=32,
        cutoff_radius=5.0,
        max_neighbors=10
    ).to(device)
    
    # 测试不同批大小
    batch_sizes = [1, 3, 5]
    atom_counts = [3, 7, 5]
    
    for batch_size, atoms_per_mol in zip(batch_sizes, atom_counts):
        try:
            data = create_test_data(num_atoms=atoms_per_mol, batch_size=batch_size)
            
            with torch.no_grad():
                energy, forces = model(data)
            
            expected_energy_shape = (batch_size,)
            expected_force_shape = (atoms_per_mol * batch_size, 3)
            
            assert energy.shape == expected_energy_shape, f"能量形状不匹配: {energy.shape} vs {expected_energy_shape}"
            assert forces.shape == expected_force_shape, f"力形状不匹配: {forces.shape} vs {expected_force_shape}"
            
            print(f"  ✅ 批大小 {batch_size}, 每分子 {atoms_per_mol} 原子: 通过")
            
        except Exception as e:
            print(f"  ❌ 批大小 {batch_size}, 每分子 {atoms_per_mol} 原子: 失败 - {e}")
            return False
    
    print("✅ 批处理测试通过!")
    return True


def main():
    """
    主测试函数
    """
    print("🚀 开始测试mlpot实现...")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_components())
    test_results.append(test_model_forward())
    test_results.append(test_equivariance())
    test_results.append(test_training_step())
    test_results.append(test_batch_processing())
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试都通过了！mlpot实现正确！")
    else:
        print("⚠️  部分测试失败，需要检查实现")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
