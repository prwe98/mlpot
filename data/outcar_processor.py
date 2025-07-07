#!/usr/bin/env python3
"""
OUTCAR数据整合工具
专门用于处理多个VASP OUTCAR文件并整合到mlpot框架中
"""

import os
import re
import torch
import numpy as np
import pickle
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

try:
    from ase.io import read
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available. Limited functionality for structure processing.")


@dataclass
class OUTCARFrame:
    """单个OUTCAR帧的数据结构"""
    positions: np.ndarray      # 原子位置 (N, 3)
    atomic_numbers: np.ndarray # 原子序数 (N,)
    forces: np.ndarray         # 原子力 (N, 3)
    energy: float              # 总能量
    cell: Optional[np.ndarray] = None  # 晶胞 (3, 3)
    stress: Optional[np.ndarray] = None  # 应力张量 (3, 3)
    step: int = 0              # 步数
    source_file: str = ""      # 来源文件


class OUTCARParser:
    """OUTCAR文件解析器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def parse_outcar(self, outcar_path: str) -> List[OUTCARFrame]:
        """
        解析单个OUTCAR文件
        
        Args:
            outcar_path: OUTCAR文件路径
            
        Returns:
            List[OUTCARFrame]: 解析的帧列表
        """
        if not os.path.exists(outcar_path):
            raise FileNotFoundError(f"OUTCAR file not found: {outcar_path}")
            
        frames = []
        
        if ASE_AVAILABLE:
            frames = self._parse_with_ase(outcar_path)
        else:
            frames = self._parse_manual(outcar_path)
            
        if self.verbose:
            print(f"✅ 解析 {outcar_path}: 找到 {len(frames)} 帧")
            
        return frames
    
    def _parse_with_ase(self, outcar_path: str) -> List[OUTCARFrame]:
        """使用ASE解析OUTCAR"""
        try:
            # 读取所有结构
            structures = read(outcar_path, index=':', format='vasp-out')
            if not isinstance(structures, list):
                structures = [structures]
                
            frames = []
            for i, atoms in enumerate(structures):
                # 提取基本信息
                positions = atoms.get_positions()
                atomic_numbers = atoms.get_atomic_numbers()
                
                # 提取能量
                energy = atoms.get_potential_energy() if atoms.calc else 0.0
                
                # 提取力
                try:
                    forces = atoms.get_forces()
                except:
                    forces = np.zeros_like(positions)
                    
                # 提取晶胞
                cell = atoms.get_cell() if atoms.pbc.any() else None
                
                frame = OUTCARFrame(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    forces=forces,
                    energy=energy,
                    cell=np.array(cell) if cell is not None else None,
                    step=i,
                    source_file=outcar_path
                )
                frames.append(frame)
                
            return frames
            
        except Exception as e:
            warnings.warn(f"ASE parsing failed for {outcar_path}: {e}")
            return self._parse_manual(outcar_path)
    
    def _parse_manual(self, outcar_path: str) -> List[OUTCARFrame]:
        """手动解析OUTCAR（不依赖ASE）"""
        frames = []
        
        with open(outcar_path, 'r') as f:
            content = f.read()
            
        # 解析原子信息（从第一个离子位置块获取）
        atomic_numbers = self._extract_atomic_numbers(content)
        
        # 提取所有的能量和力信息
        energies = self._extract_energies(content)
        positions_list = self._extract_positions(content)
        forces_list = self._extract_forces(content)
        cells = self._extract_cells(content)
        
        # 确保数据一致性
        min_length = min(len(energies), len(positions_list), len(forces_list))
        
        for i in range(min_length):
            frame = OUTCARFrame(
                positions=positions_list[i],
                atomic_numbers=atomic_numbers,
                forces=forces_list[i],
                energy=energies[i],
                cell=cells[0] if cells else None,  # 通常晶胞不变
                step=i,
                source_file=outcar_path
            )
            frames.append(frame)
            
        return frames
    
    def _extract_atomic_numbers(self, content: str) -> np.ndarray:
        """提取原子序数"""
        # 查找POTCAR信息来确定原子类型
        element_pattern = r'VRHFIN =(\w+)'
        elements = re.findall(element_pattern, content)
        
        # 查找每种原子的数量
        ions_per_type_pattern = r'ions per type =\s*([\d\s]+)'
        ions_match = re.search(ions_per_type_pattern, content)
        
        if not ions_match or not elements:
            raise ValueError("无法解析原子信息")
            
        ions_per_type = list(map(int, ions_match.group(1).split()))
        
        # 元素符号到原子序数的映射
        element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36
        }
        
        atomic_numbers = []
        for element, count in zip(elements, ions_per_type):
            if element in element_to_z:
                atomic_numbers.extend([element_to_z[element]] * count)
            else:
                # 如果元素不在映射中，尝试从元素符号提取
                z = self._symbol_to_atomic_number(element)
                atomic_numbers.extend([z] * count)
                
        return np.array(atomic_numbers)
    
    def _symbol_to_atomic_number(self, symbol: str) -> int:
        """将元素符号转换为原子序数"""
        # 简化版本，您可以扩展这个映射
        basic_map = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16,
            'Cl': 17, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Br': 35
        }
        return basic_map.get(symbol, 6)  # 默认返回碳
    
    def _extract_energies(self, content: str) -> List[float]:
        """提取能量"""
        # 查找 "free energy" 或 "energy without entropy"
        energy_pattern = r'free\s+energy\s+TOTEN\s*=\s*([-\d\.]+)'
        energies = re.findall(energy_pattern, content)
        return [float(e) for e in energies]
    
    def _extract_positions(self, content: str) -> List[np.ndarray]:
        """提取原子位置"""
        positions_list = []
        
        # 查找所有的位置块
        position_blocks = re.findall(
            r'POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-+|\Z)',
            content, re.DOTALL
        )
        
        for block in position_blocks:
            positions = []
            lines = block.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:  # 确保有足够的列
                        try:
                            pos = [float(parts[0]), float(parts[1]), float(parts[2])]
                            positions.append(pos)
                        except ValueError:
                            continue
            if positions:
                positions_list.append(np.array(positions))
                
        return positions_list
    
    def _extract_forces(self, content: str) -> List[np.ndarray]:
        """提取力"""
        forces_list = []
        
        # 查找所有的力块
        position_blocks = re.findall(
            r'POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-+|\Z)',
            content, re.DOTALL
        )
        
        for block in position_blocks:
            forces = []
            lines = block.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:  # 确保有足够的列
                        try:
                            force = [float(parts[3]), float(parts[4]), float(parts[5])]
                            forces.append(force)
                        except ValueError:
                            continue
            if forces:
                forces_list.append(np.array(forces))
                
        return forces_list
    
    def _extract_cells(self, content: str) -> List[np.ndarray]:
        """提取晶胞参数"""
        cells = []
        
        # 查找晶格向量
        cell_pattern = r'direct lattice vectors\s+reciprocal lattice vectors\s*\n(.*?)\n.*?\n.*?\n'
        cell_matches = re.findall(cell_pattern, content, re.DOTALL)
        
        for match in cell_matches:
            lines = match.strip().split('\n')
            if len(lines) >= 3:
                cell = []
                for i in range(3):
                    parts = lines[i].split()
                    if len(parts) >= 3:
                        cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(cell) == 3:
                    cells.append(np.array(cell))
                    
        return cells


class OUTCARDatasetBuilder:
    """OUTCAR数据集构建器"""
    
    def __init__(self, output_format: str = 'h5', verbose: bool = True):
        """
        Args:
            output_format: 输出格式 ('h5', 'pickle', 'npz')
            verbose: 是否显示详细信息
        """
        self.output_format = output_format
        self.verbose = verbose
        self.parser = OUTCARParser(verbose=verbose)
        
    def build_dataset(
        self,
        outcar_paths: List[str],
        output_path: str,
        energy_unit: str = 'eV',
        force_unit: str = 'eV/Angst',
        position_unit: str = 'Angst'
    ) -> Dict[str, Any]:
        """
        构建数据集
        
        Args:
            outcar_paths: OUTCAR文件路径列表
            output_path: 输出文件路径
            energy_unit: 能量单位
            force_unit: 力的单位
            position_unit: 位置单位
            
        Returns:
            Dict: 数据集统计信息
        """
        all_frames = []
        
        # 解析所有OUTCAR文件
        if self.verbose:
            print(f"🔄 开始解析 {len(outcar_paths)} 个OUTCAR文件...")
            
        for outcar_path in outcar_paths:
            try:
                frames = self.parser.parse_outcar(outcar_path)
                all_frames.extend(frames)
            except Exception as e:
                warnings.warn(f"解析文件失败 {outcar_path}: {e}")
                continue
                
        if not all_frames:
            raise ValueError("没有成功解析任何OUTCAR文件")
            
        if self.verbose:
            print(f"✅ 总共解析了 {len(all_frames)} 个结构")
            
        # 转换为mlpot格式
        dataset = self._convert_to_mlpot_format(all_frames)
        
        # 保存数据集
        self._save_dataset(dataset, output_path)
        
        # 计算统计信息
        stats = self._compute_statistics(dataset)
        
        if self.verbose:
            self._print_statistics(stats)
            
        return stats
    
    def _convert_to_mlpot_format(self, frames: List[OUTCARFrame]) -> Dict[str, Any]:
        """转换为mlpot格式"""
        dataset = {
            'positions': [],
            'atomic_numbers': [],
            'forces': [],
            'energies': [],
            'cells': [],
            'num_atoms': [],
            'metadata': {
                'total_frames': len(frames),
                'source': 'OUTCAR',
                'units': {
                    'energy': 'eV',
                    'force': 'eV/Angst',
                    'position': 'Angst'
                }
            }
        }
        
        for frame in frames:
            dataset['positions'].append(frame.positions)
            dataset['atomic_numbers'].append(frame.atomic_numbers)
            dataset['forces'].append(frame.forces)
            dataset['energies'].append(frame.energy)
            dataset['num_atoms'].append(len(frame.atomic_numbers))
            
            if frame.cell is not None:
                dataset['cells'].append(frame.cell)
            else:
                # 对于非周期性系统，创建一个大的单位晶胞
                large_cell = np.eye(3) * 100.0
                dataset['cells'].append(large_cell)
                
        return dataset
    
    def _save_dataset(self, dataset: Dict[str, Any], output_path: str):
        """保存数据集"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == 'h5':
            self._save_h5(dataset, str(output_path))
        elif self.output_format == 'pickle':
            self._save_pickle(dataset, str(output_path))
        elif self.output_format == 'npz':
            self._save_npz(dataset, str(output_path))
        else:
            raise ValueError(f"不支持的输出格式: {self.output_format}")
            
        if self.verbose:
            print(f"💾 数据集已保存到: {output_path}")
    
    def _save_h5(self, dataset: Dict[str, Any], output_path: str):
        """保存为HDF5格式"""
        with h5py.File(output_path, 'w') as f:
            # 保存数组数据
            for key in ['positions', 'atomic_numbers', 'forces', 'energies', 'cells', 'num_atoms']:
                if key in dataset:
                    f.create_dataset(key, data=dataset[key])
            
            # 保存元数据
            meta_group = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                if isinstance(value, dict):
                    sub_group = meta_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.attrs[sub_key] = sub_value
                else:
                    meta_group.attrs[key] = value
    
    def _save_pickle(self, dataset: Dict[str, Any], output_path: str):
        """保存为Pickle格式"""
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    def _save_npz(self, dataset: Dict[str, Any], output_path: str):
        """保存为NPZ格式"""
        np.savez_compressed(output_path, **dataset)
    
    def _compute_statistics(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        """计算数据集统计信息"""
        energies = np.array(dataset['energies'])
        forces = np.concatenate(dataset['forces'])
        num_atoms = np.array(dataset['num_atoms'])
        
        stats = {
            'total_structures': len(dataset['energies']),
            'total_atoms': np.sum(num_atoms),
            'avg_atoms_per_structure': np.mean(num_atoms),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'energy_min': np.min(energies),
            'energy_max': np.max(energies),
            'force_mean': np.mean(np.abs(forces)),
            'force_std': np.std(forces),
            'force_max': np.max(np.abs(forces))
        }
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, float]):
        """打印统计信息"""
        print("\n📊 数据集统计信息:")
        print("=" * 50)
        print(f"总结构数: {stats['total_structures']}")
        print(f"总原子数: {stats['total_atoms']}")
        print(f"平均每结构原子数: {stats['avg_atoms_per_structure']:.1f}")
        print(f"能量范围: {stats['energy_min']:.3f} ~ {stats['energy_max']:.3f} eV")
        print(f"能量均值: {stats['energy_mean']:.3f} ± {stats['energy_std']:.3f} eV")
        print(f"力均值(绝对值): {stats['force_mean']:.3f} eV/Å")
        print(f"力标准差: {stats['force_std']:.3f} eV/Å")
        print(f"最大力: {stats['force_max']:.3f} eV/Å")


def find_outcar_files(root_dir: str, recursive: bool = True) -> List[str]:
    """
    在目录中查找所有OUTCAR文件
    
    Args:
        root_dir: 根目录
        recursive: 是否递归搜索
        
    Returns:
        List[str]: OUTCAR文件路径列表
    """
    outcar_files = []
    root_path = Path(root_dir)
    
    if recursive:
        # 递归搜索
        for outcar_file in root_path.rglob("OUTCAR*"):
            if outcar_file.is_file():
                outcar_files.append(str(outcar_file))
    else:
        # 只搜索当前目录
        for outcar_file in root_path.glob("OUTCAR*"):
            if outcar_file.is_file():
                outcar_files.append(str(outcar_file))
                
    return sorted(outcar_files)


def main():
    """示例使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OUTCAR数据整合工具")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="包含OUTCAR文件的目录")
    parser.add_argument("--output", "-o", type=str, default="outcar_dataset.h5",
                       help="输出文件路径")
    parser.add_argument("--format", "-f", type=str, default="h5",
                       choices=['h5', 'pickle', 'npz'],
                       help="输出格式")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="递归搜索OUTCAR文件")
    parser.add_argument("--files", nargs="+", type=str,
                       help="指定特定的OUTCAR文件")
    
    args = parser.parse_args()
    
    # 查找OUTCAR文件
    if args.files:
        outcar_files = args.files
    else:
        outcar_files = find_outcar_files(args.input_dir, args.recursive)
    
    if not outcar_files:
        print("❌ 没有找到OUTCAR文件")
        return
    
    print(f"🔍 找到 {len(outcar_files)} 个OUTCAR文件")
    for f in outcar_files:
        print(f"  - {f}")
    
    # 构建数据集
    builder = OUTCARDatasetBuilder(output_format=args.format)
    try:
        stats = builder.build_dataset(outcar_files, args.output)
        print(f"\n🎉 数据集构建完成!")
        print(f"输出文件: {args.output}")
    except Exception as e:
        print(f"❌ 构建失败: {e}")


if __name__ == "__main__":
    main()
