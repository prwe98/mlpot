#!/usr/bin/env python3
"""
OUTCAR处理命令行工具
使用方法: python process_outcar.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# 添加mlpot路径
current_dir = Path(__file__).parent
if current_dir.name == 'mlpot':
    sys.path.append(str(current_dir.parent))
else:
    sys.path.append(str(current_dir))

def process_outcar_files(args):
    """处理OUTCAR文件的主函数"""
    try:
        from mlpot.data.outcar_processor import OUTCARDatasetBuilder, find_outcar_files
        
        print("🔧 OUTCAR文件处理工具")
        print("=" * 50)
        
        # 查找OUTCAR文件
        if args.files:
            outcar_files = args.files
            print(f"📋 使用指定的 {len(outcar_files)} 个文件")
        else:
            print(f"🔍 在 {args.input_dir} 中搜索OUTCAR文件...")
            outcar_files = find_outcar_files(args.input_dir, recursive=args.recursive)
            
        if not outcar_files:
            print("❌ 没有找到OUTCAR文件")
            return False
            
        print(f"✅ 找到 {len(outcar_files)} 个OUTCAR文件")
        
        if args.list_files:
            print("\n📂 文件列表:")
            for i, f in enumerate(outcar_files, 1):
                print(f"  {i:3d}. {f}")
            if not args.process:
                return True
                
        if not args.process:
            print("\n💡 添加 --process 参数开始处理文件")
            return True
            
        # 处理文件
        print(f"\n🔄 开始处理文件，输出格式: {args.format}")
        
        builder = OUTCARDatasetBuilder(output_format=args.format, verbose=True)
        
        try:
            stats = builder.build_dataset(outcar_files, args.output)
            
            print(f"\n🎉 处理完成!")
            print(f"📄 输出文件: {args.output}")
            
            if args.show_stats:
                print(f"\n📊 数据集统计:")
                print(f"  - 总结构数: {stats['total_structures']}")
                print(f"  - 总原子数: {stats['total_atoms']}")
                print(f"  - 平均原子数/结构: {stats['avg_atoms_per_structure']:.1f}")
                print(f"  - 能量范围: {stats['energy_min']:.3f} ~ {stats['energy_max']:.3f} eV")
                print(f"  - 最大力: {stats['force_max']:.3f} eV/Å")
                
            return True
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保mlpot框架正确安装")
        return False


def create_training_script(args):
    """创建训练脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
使用{args.output}数据集的mlpot训练脚本
自动生成于OUTCAR处理工具
"""

import sys
import torch
import torch.optim as optim
from pathlib import Path

# 添加mlpot路径
sys.path.append(str(Path(__file__).parent))

from mlpot import EquivariantNet
from mlpot.data.dataset import MolecularDataset, create_dataloader
from mlpot.core.trainer import PotentialTrainer
from mlpot.utils.helpers import set_random_seed

def main():
    # 设置随机种子
    set_random_seed(42)
    
    print("🚀 开始mlpot训练")
    print("=" * 40)
    
    # 1. 加载数据集
    print("📂 加载数据集...")
    dataset = MolecularDataset(
        data_path="{args.output}",
        format_type="{args.format}"
    )
    
    print(f"✅ 数据集加载完成: {{len(dataset)}} 个样本")
    
    # 2. 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 训练集: {{len(train_dataset)}} 样本")
    print(f"📊 验证集: {{len(val_dataset)}} 样本")
    
    # 3. 创建数据加载器
    train_loader = create_dataloader(
        train_dataset, 
        batch_size={args.batch_size}, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = create_dataloader(
        val_dataset, 
        batch_size={args.batch_size}, 
        shuffle=False,
        num_workers=2
    )
    
    # 4. 创建模型
    print("🧠 创建模型...")
    model = EquivariantNet(
        hidden_dim={args.hidden_dim},
        num_layers={args.num_layers},
        cutoff_radius={args.cutoff},
        max_neighbors={args.max_neighbors}
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建完成: {{total_params:,}} 参数")
    
    # 5. 设置优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr={args.learning_rate},
        weight_decay=1e-5
    )
    
    # 6. 创建训练器
    trainer = PotentialTrainer(
        model=model,
        optimizer=optimizer,
        loss_config={{
            'energy_weight': {args.energy_weight},
            'force_weight': {args.force_weight},
            'loss_type': 'l1'
        }},
        gradient_clip=10.0
    )
    
    # 7. 开始训练
    print(f"🏋️ 开始训练 {{args.epochs}} 个epochs...")
    
    try:
        history = trainer.fit(
            train_loader, 
            val_loader, 
            epochs={args.epochs},
            save_checkpoint=True,
            checkpoint_dir="./checkpoints"
        )
        
        print("🎉 训练完成!")
        print("📁 检查点保存在 ./checkpoints 目录")
        
    except KeyboardInterrupt:
        print("\\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练失败: {{e}}")

if __name__ == "__main__":
    main()
'''
    
    script_path = f"train_{Path(args.output).stem}.py"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
        
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"📝 训练脚本已创建: {script_path}")
    print(f"💡 运行方法: python {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OUTCAR文件处理工具 - 将VASP OUTCAR文件转换为mlpot数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 搜索目录中的OUTCAR文件
  python process_outcar.py -i /path/to/calculations --list

  # 处理所有OUTCAR文件并创建数据集
  python process_outcar.py -i /path/to/calculations -o dataset.h5 --process

  # 处理特定文件
  python process_outcar.py --files OUTCAR1 OUTCAR2 -o dataset.h5 --process

  # 生成训练脚本
  python process_outcar.py -i /path/to/calculations -o dataset.h5 --process --create-script
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input-dir", type=str,
        help="包含OUTCAR文件的目录"
    )
    input_group.add_argument(
        "--files", nargs="+", type=str,
        help="指定特定的OUTCAR文件路径"
    )
    
    # 输出选项
    parser.add_argument(
        "-o", "--output", type=str, default="outcar_dataset.h5",
        help="输出数据集文件路径 (默认: outcar_dataset.h5)"
    )
    
    parser.add_argument(
        "-f", "--format", type=str, default="h5",
        choices=['h5', 'pickle', 'npz'],
        help="输出格式 (默认: h5)"
    )
    
    # 处理选项
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归搜索OUTCAR文件"
    )
    
    parser.add_argument(
        "--list", dest="list_files", action="store_true",
        help="列出找到的OUTCAR文件"
    )
    
    parser.add_argument(
        "--process", action="store_true",
        help="开始处理OUTCAR文件"
    )
    
    parser.add_argument(
        "--show-stats", action="store_true",
        help="显示数据集统计信息"
    )
    
    # 训练脚本生成选项
    parser.add_argument(
        "--create-script", action="store_true",
        help="创建训练脚本"
    )
    
    # 训练参数（用于脚本生成）
    train_group = parser.add_argument_group("训练参数 (用于脚本生成)")
    train_group.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    train_group.add_argument("--num-layers", type=int, default=3, help="网络层数")
    train_group.add_argument("--cutoff", type=float, default=6.0, help="截断半径")
    train_group.add_argument("--max-neighbors", type=int, default=50, help="最大邻居数")
    train_group.add_argument("--batch-size", type=int, default=16, help="批次大小")
    train_group.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    train_group.add_argument("--epochs", type=int, default=100, help="训练轮数")
    train_group.add_argument("--energy-weight", type=float, default=1.0, help="能量损失权重")
    train_group.add_argument("--force-weight", type=float, default=50.0, help="力损失权重")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认列出文件
    if not args.list_files and not args.process:
        args.list_files = True
    
    success = process_outcar_files(args)
    
    if success and args.process and args.create_script:
        create_training_script(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
