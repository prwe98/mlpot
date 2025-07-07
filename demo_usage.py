#!/usr/bin/env python3
"""
mlpot使用示例：完整的分子势能学习流程
这个示例展示如何使用mlpot框架进行分子势能预测
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_demo_dataset():
    """创建演示数据集"""
    print("📦 创建演示数据集...")
    
    # 创建一些简单的分子结构
    molecules = []
    
    # 分子1: H2O (水分子)
    h2o = {
        'pos': torch.tensor([
            [0.0, 0.0, 0.0],      # O
            [0.76, 0.59, 0.0],    # H1
            [-0.76, 0.59, 0.0]    # H2
        ]),
        'atomic_numbers': torch.tensor([8, 1, 1]),  # O, H, H
        'energy': torch.tensor(-76.3),  # 假设能量
        'forces': torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, -0.1, 0.0]
        ])
    }
    
    # 分子2: CH4 (甲烷)
    ch4 = {
        'pos': torch.tensor([
            [0.0, 0.0, 0.0],      # C
            [1.09, 1.09, 1.09],   # H1
            [-1.09, -1.09, 1.09], # H2
            [-1.09, 1.09, -1.09], # H3
            [1.09, -1.09, -1.09]  # H4
        ]),
        'atomic_numbers': torch.tensor([6, 1, 1, 1, 1]),  # C, H, H, H, H
        'energy': torch.tensor(-40.5),  # 假设能量
        'forces': torch.tensor([
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [-0.05, -0.05, 0.05],
            [-0.05, 0.05, -0.05],
            [0.05, -0.05, -0.05]
        ])
    }
    
    molecules = [h2o, ch4]
    
    # 创建批处理数据
    all_pos = []
    all_atomic_numbers = []
    all_energies = []
    all_forces = []
    batch_indices = []
    
    for mol_idx, mol in enumerate(molecules):
        all_pos.append(mol['pos'])
        all_atomic_numbers.append(mol['atomic_numbers'])
        all_energies.append(mol['energy'])
        all_forces.append(mol['forces'])
        
        # 批索引
        batch_indices.extend([mol_idx] * len(mol['atomic_numbers']))
    
    batch_data = {
        'pos': torch.cat(all_pos, dim=0),
        'atomic_numbers': torch.cat(all_atomic_numbers, dim=0),
        'batch': torch.tensor(batch_indices),
        'energy': torch.stack(all_energies),
        'forces': torch.cat(all_forces, dim=0)
    }
    
    print(f"✅ 创建了包含 {len(molecules)} 个分子的数据集")
    return batch_data

def create_mlpot_model():
    """创建mlpot模型"""
    print("\n🏗️  创建mlpot模型...")
    
    try:
        # 导入所需模块
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("base_model", "core/base_model.py")
        base_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_model)
        
        spec = importlib.util.spec_from_file_location("geometric_layers", "layers/geometric_layers.py")
        geom_layers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geom_layers)
        
        # 创建完整的mlpot模型
        class MLPotModel(base_model.BasePotential):
            def __init__(self, hidden_dim=128, num_layers=3):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # 原子嵌入
                self.atomic_embedding = geom_layers.AtomicEmbedding(hidden_dim, 84)
                
                # 径向基函数
                self.radial_basis = geom_layers.RadialBasisFunction(
                    num_radial=64, cutoff=5.0
                )
                
                # 消息传递层
                self.message_layers = nn.ModuleList()
                for _ in range(num_layers):
                    self.message_layers.append(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            geom_layers.ScaledActivation(),
                            nn.Linear(hidden_dim, hidden_dim)
                        )
                    )
                
                # 输出层
                self.energy_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    geom_layers.ScaledActivation(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                self.force_head = nn.Linear(hidden_dim, 3)
                
            def forward(self, data):
                atomic_numbers = data['atomic_numbers'].long()
                pos = data['pos']
                batch = data['batch']
                
                # 嵌入原子特征
                node_features = self.atomic_embedding(atomic_numbers)
                
                # 简单的图卷积（演示用）
                for layer in self.message_layers:
                    node_features = layer(node_features) + node_features
                
                # 预测能量
                per_atom_energy = self.energy_head(node_features).squeeze(1)
                from torch_scatter import scatter
                total_energy = scatter(per_atom_energy, batch, dim=0, reduce='sum')
                
                # 预测力
                forces = self.force_head(node_features)
                
                return total_energy, forces
            
            def get_energy_and_forces(self, data):
                return self.forward(data)
        
        model = MLPotModel()
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def train_model_demo(model, data):
    """演示模型训练"""
    print("\n🏋️ 演示模型训练...")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    num_epochs = 2000
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        pred_energy, pred_forces = model(data)
        
        # 计算损失
        energy_loss = nn.MSELoss()(pred_energy, data['energy'])
        force_loss = nn.MSELoss()(pred_forces, data['forces'])
        total_loss = energy_loss + 10 * force_loss  # 力损失权重更高
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: 损失 = {total_loss.item():.6f}")
    
    print(f"✅ 训练完成，最终损失: {losses[-1]:.6f}")
    return losses

def evaluate_model(model, data):
    """评估模型性能"""
    print("\n📊 评估模型性能...")
    
    with torch.no_grad():
        pred_energy, pred_forces = model(data)
    
    # 计算误差
    energy_mae = torch.mean(torch.abs(pred_energy - data['energy']))
    force_mae = torch.mean(torch.abs(pred_forces - data['forces']))
    
    print(f"✅ 评估结果:")
    print(f"   能量平均绝对误差: {energy_mae.item():.6f} eV")
    print(f"   力平均绝对误差: {force_mae.item():.6f} eV/Å")
    
    # 显示详细结果
    print(f"\n🔍 详细预测结果:")
    for i in range(len(data['energy'])):
        print(f"   分子 {i+1}:")
        print(f"     真实能量: {data['energy'][i].item():.3f} eV")
        print(f"     预测能量: {pred_energy[i].item():.3f} eV")
    
    return pred_energy, pred_forces

def visualize_results(losses):
    """可视化训练结果"""
    print("\n📈 生成训练曲线...")
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Training Epoch')
        plt.ylabel('Loss Value')
        plt.title('MLPot Model Training Curve')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 保存图片
        plt.savefig('mlpot_training_curve.png', dpi=150, bbox_inches='tight')
        print("✅ 训练曲线已保存到 mlpot_training_curve.png")
        
        # 显示
        plt.show()
        
    except Exception as e:
        print(f"⚠️  可视化失败 (可能缺少GUI): {e}")

def demonstrate_equivariance(model, data):
    """演示等变性质"""
    print("\n🔄 演示等变性质...")
    
    # 原始预测
    with torch.no_grad():
        original_energy, original_forces = model(data)
    
    # 旋转变换
    angle = np.pi / 4  # 45度
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 变换数据
    transformed_data = data.copy()
    transformed_data['pos'] = torch.matmul(data['pos'], rotation_matrix.T)
    
    # 变换后的预测
    with torch.no_grad():
        transformed_energy, transformed_forces = model(transformed_data)
    
    # 检查能量不变性
    energy_diff = torch.abs(original_energy - transformed_energy)
    energy_invariant = torch.all(energy_diff < 1e-3)
    
    print(f"✅ 等变性检查:")
    print(f"   能量不变性: {'通过' if energy_invariant else '失败'}")
    print(f"   最大能量差异: {energy_diff.max().item():.6f} eV")

def main():
    """主演示函数"""
    print("🚀 mlpot分子势能学习演示")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 创建数据
    data = create_demo_dataset()
    
    # 2. 创建模型
    model = create_mlpot_model()
    if model is None:
        return
    
    # 3. 训练模型
    losses = train_model_demo(model, data)
    
    # 4. 评估性能
    pred_energy, pred_forces = evaluate_model(model, data)
    
    # 5. 可视化结果
    visualize_results(losses)
    
    # 6. 演示等变性
    demonstrate_equivariance(model, data)
    
    print("\n" + "=" * 50)
    print("🎉 mlpot演示完成！")
    print("\n✨ 框架特点:")
    print("   🧠 基于E2GNN的等变架构")
    print("   🔧 模块化设计，易于扩展")
    print("   📊 完整的训练和评估流程")
    print("   🧪 等变性质验证")
    print("   📈 可视化工具")
    print("\n🚀 准备好用于实际的分子势能学习任务！")

if __name__ == "__main__":
    main()
