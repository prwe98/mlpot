"""
OUTCAR数据处理器，用于解析VASP输出文件并转换为MLPot训练格式
"""

import os
import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from pathlib import Path

try:
    from ase import Atoms
    from ase.io import read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available. Some functionality will be limited.")

try:
    from pymatgen.io.vasp import Outcar
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("Pymatgen not available. Will use manual parsing.")


@dataclass
class OUTCARFrame:
    """单帧OUTCAR数据结构"""
    positions: np.ndarray          # (N, 3) 原子坐标
    atomic_numbers: np.ndarray     # (N,) 原子序数
    cell: np.ndarray              # (3, 3) 晶胞参数
    energy: float                 # 总能量
    forces: np.ndarray            # (N, 3) 原子受力
    stress: Optional[np.ndarray]  # (6,) 应力张量
    temperature: Optional[float]  # 温度 (如果是MD)
    pressure: Optional[float]     # 压力
    volume: float                 # 体积
    step: int                     # 步数
    converged: bool               # 是否收敛


class OUTCARProcessor:
    """
    OUTCAR文件处理器
    
    功能：
    1. 解析OUTCAR文件中的结构、能量、力等信息
    2. 提取MD轨迹或几何优化路径
    3. 转换为MLPot训练格式
    4. 支持批量处理和数据过滤
    """
    
    def __init__(self, 
                 force_threshold: float = 0.1,  # eV/Å
                 energy_threshold: float = 0.1,  # eV
                 use_pymatgen: bool = True):
        """
        初始化处理器
        
        Args:
            force_threshold: 力的过滤阈值，过大的力可能表示不稳定结构
            energy_threshold: 能量变化阈值，用于过滤异常点
            use_pymatgen: 是否使用pymatgen库（更稳定）
        """
        self.force_threshold = force_threshold
        self.energy_threshold = energy_threshold
        self.use_pymatgen = use_pymatgen and PYMATGEN_AVAILABLE
        
    def parse_outcar(self, outcar_path: Union[str, Path]) -> List[OUTCARFrame]:
        """
        解析OUTCAR文件
        
        Args:
            outcar_path: OUTCAR文件路径
            
        Returns:
            List[OUTCARFrame]: 解析得到的帧列表
        """
        outcar_path = Path(outcar_path)
        if not outcar_path.exists():
            raise FileNotFoundError(f"OUTCAR file not found: {outcar_path}")
            
        if self.use_pymatgen:
            return self._parse_with_pymatgen(outcar_path)
        else:
            return self._parse_manually(outcar_path)
    
    def _parse_with_pymatgen(self, outcar_path: Path) -> List[OUTCARFrame]:
        """使用pymatgen解析OUTCAR"""
        try:
            outcar = Outcar(str(outcar_path))
            frames = []
            
            # 获取基本信息
            if not outcar.run_stats:
                warnings.warn(f"No run statistics found in {outcar_path}")
                return []
                
            # 处理每一步
            for i, structure in enumerate(outcar.structures):
                # 能量
                if i < len(outcar.final_energy_per_atom):
                    energy = outcar.final_energy_per_atom[i] * len(structure)
                else:
                    continue
                    
                # 力
                if i < len(outcar.forces):
                    forces = np.array(outcar.forces[i])
                else:
                    continue
                
                # 应力
                stress = None
                if outcar.stress and i < len(outcar.stress):
                    stress = np.array(outcar.stress[i])
                
                # 构建帧
                frame = OUTCARFrame(
                    positions=structure.cart_coords,
                    atomic_numbers=np.array([site.specie.Z for site in structure]),
                    cell=structure.lattice.matrix,
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    temperature=None,  # pymatgen可能不提供
                    pressure=None,
                    volume=structure.volume,
                    step=i,
                    converged=True  # 假设收敛
                )
                
                frames.append(frame)
                
            return frames
            
        except Exception as e:
            warnings.warn(f"Pymatgen parsing failed: {e}. Falling back to manual parsing.")
            return self._parse_manually(outcar_path)
    
    def _parse_manually(self, outcar_path: Path) -> List[OUTCARFrame]:
        """手动解析OUTCAR文件"""
        frames = []
        
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # 正则表达式模式
        patterns = {
            'natoms': r'NIONS =\s+(\d+)',
            'species': r'VRHFIN =(\w+)',
            'lattice': r'direct lattice vectors.*?\n((?:\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+.*?\n){3})',
            'positions': r'POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-{50,}\s*\n((?:\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+.*?\n)+)',
            'energy': r'free  energy   TOTEN  =\s+([-\d\.]+)',
            'stress': r'in kB\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)',
            'volume': r'volume of cell :\s+([\d\.]+)',
            'temperature': r'temperature\s+([\d\.]+)\s+K'
        }
        
        # 提取基本信息
        natoms_match = re.search(patterns['natoms'], content)
        if not natoms_match:
            raise ValueError("Cannot find number of atoms in OUTCAR")
        natoms = int(natoms_match.group(1))
        
        # 提取物种信息
        species_matches = re.findall(patterns['species'], content)
        
        # 查找所有离子步
        ionic_steps = re.finditer(r'(POSITION\s+TOTAL-FORCE.*?(?=POSITION\s+TOTAL-FORCE|$))', content, re.DOTALL)
        
        step = 0
        for step_match in ionic_steps:
            step_content = step_match.group(1)
            
            try:
                # 解析坐标和力
                pos_force_match = re.search(patterns['positions'], step_content)
                if not pos_force_match:
                    continue
                    
                lines = pos_force_match.group(1).strip().split('\n')
                positions = []
                forces = []
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 6:
                        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
                
                if len(positions) != natoms:
                    continue
                    
                positions = np.array(positions)
                forces = np.array(forces)
                
                # 解析能量
                energy_match = re.search(patterns['energy'], step_content)
                if not energy_match:
                    continue
                energy = float(energy_match.group(1))
                
                # 解析晶胞（在第一步提取，假设不变）
                if step == 0:
                    lattice_match = re.search(patterns['lattice'], content)
                    if lattice_match:
                        lattice_lines = lattice_match.group(1).strip().split('\n')
                        cell = []
                        for line in lattice_lines:
                            parts = line.split()
                            cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        cell = np.array(cell)
                    else:
                        cell = np.eye(3) * 10  # 默认晶胞
                
                # 解析应力
                stress = None
                stress_match = re.search(patterns['stress'], step_content)
                if stress_match:
                    stress = np.array([float(x) for x in stress_match.groups()])
                
                # 解析体积
                volume_match = re.search(patterns['volume'], step_content)
                volume = float(volume_match.group(1)) if volume_match else np.linalg.det(cell)
                
                # 解析温度
                temp_match = re.search(patterns['temperature'], step_content)
                temperature = float(temp_match.group(1)) if temp_match else None
                
                # 构建原子序数（简化处理）
                atomic_numbers = self._guess_atomic_numbers(natoms, species_matches)
                
                # 创建帧
                frame = OUTCARFrame(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    cell=cell,
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    temperature=temperature,
                    pressure=None,
                    volume=volume,
                    step=step,
                    converged=True
                )
                
                frames.append(frame)
                step += 1
                
            except Exception as e:
                warnings.warn(f"Failed to parse step {step}: {e}")
                continue
        
        return frames
    
    def _guess_atomic_numbers(self, natoms: int, species: List[str]) -> np.ndarray:
        """推测原子序数"""
        from mlpot.utils.chemistry import get_atomic_number
        
        # 简化处理：假设均匀分布
        if not species:
            return np.ones(natoms, dtype=int)  # 默认氢原子
            
        atomic_numbers = []
        atoms_per_species = natoms // len(species)
        remainder = natoms % len(species)
        
        for i, symbol in enumerate(species):
            count = atoms_per_species + (1 if i < remainder else 0)
            try:
                atomic_num = get_atomic_number(symbol)
                atomic_numbers.extend([atomic_num] * count)
            except:
                atomic_numbers.extend([1] * count)  # 默认氢原子
                
        return np.array(atomic_numbers[:natoms])
    
    def filter_frames(self, frames: List[OUTCARFrame]) -> List[OUTCARFrame]:
        """
        过滤异常帧
        
        Args:
            frames: 输入帧列表
            
        Returns:
            过滤后的帧列表
        """
        filtered_frames = []
        
        if not frames:
            return filtered_frames
            
        # 计算能量变化
        energies = [frame.energy for frame in frames]
        energy_changes = np.abs(np.diff(energies))
        
        for i, frame in enumerate(frames):
            # 检查力是否过大
            max_force = np.max(np.linalg.norm(frame.forces, axis=1))
            if max_force > self.force_threshold:
                warnings.warn(f"Frame {frame.step} has large forces: {max_force:.3f} eV/Å")
                continue
                
            # 检查能量变化是否异常
            if i > 0 and energy_changes[i-1] > self.energy_threshold:
                warnings.warn(f"Frame {frame.step} has large energy change: {energy_changes[i-1]:.3f} eV")
                continue
                
            # 检查是否包含NaN
            if (np.isnan(frame.positions).any() or 
                np.isnan(frame.forces).any() or 
                np.isnan(frame.energy)):
                warnings.warn(f"Frame {frame.step} contains NaN values")
                continue
                
            filtered_frames.append(frame)
            
        print(f"Filtered {len(frames)} -> {len(filtered_frames)} frames")
        return filtered_frames
    
    def convert_to_mlpot_format(self, frames: List[OUTCARFrame]) -> List[Dict]:
        """
        转换为MLPot训练格式
        
        Args:
            frames: OUTCAR帧列表
            
        Returns:
            MLPot格式的数据列表
        """
        mlpot_data = []
        
        for frame in frames:
            data = {
                'pos': torch.tensor(frame.positions, dtype=torch.float32),
                'atomic_numbers': torch.tensor(frame.atomic_numbers, dtype=torch.long),
                'cell': torch.tensor(frame.cell, dtype=torch.float32),
                'energy': torch.tensor(frame.energy, dtype=torch.float32),
                'forces': torch.tensor(frame.forces, dtype=torch.float32),
                'natoms': len(frame.positions),
                'volume': frame.volume,
                'step': frame.step
            }
            
            # 可选字段
            if frame.stress is not None:
                data['stress'] = torch.tensor(frame.stress, dtype=torch.float32)
            if frame.temperature is not None:
                data['temperature'] = frame.temperature
            if frame.pressure is not None:
                data['pressure'] = frame.pressure
                
            mlpot_data.append(data)
            
        return mlpot_data
    
    def process_outcar_directory(self, 
                                outcar_dir: Union[str, Path],
                                output_dir: Union[str, Path],
                                pattern: str = "**/OUTCAR") -> None:
        """
        批量处理OUTCAR文件
        
        Args:
            outcar_dir: 包含OUTCAR的目录
            output_dir: 输出目录
            pattern: 文件匹配模式
        """
        outcar_dir = Path(outcar_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outcar_files = list(outcar_dir.glob(pattern))
        print(f"Found {len(outcar_files)} OUTCAR files")
        
        all_data = []
        
        for i, outcar_file in enumerate(outcar_files):
            print(f"Processing {i+1}/{len(outcar_files)}: {outcar_file}")
            
            try:
                # 解析文件
                frames = self.parse_outcar(outcar_file)
                if not frames:
                    continue
                    
                # 过滤异常帧
                frames = self.filter_frames(frames)
                if not frames:
                    continue
                    
                # 转换格式
                mlpot_data = self.convert_to_mlpot_format(frames)
                all_data.extend(mlpot_data)
                
                # 保存单个文件的数据
                output_file = output_dir / f"outcar_{i:04d}.pt"
                torch.save(mlpot_data, output_file)
                
            except Exception as e:
                warnings.warn(f"Failed to process {outcar_file}: {e}")
                continue
        
        # 保存所有数据
        if all_data:
            combined_file = output_dir / "all_outcar_data.pt"
            torch.save(all_data, combined_file)
            print(f"Saved {len(all_data)} frames to {combined_file}")
            
            # 保存统计信息
            self._save_statistics(all_data, output_dir / "statistics.txt")
    
    def _save_statistics(self, data: List[Dict], stats_file: Path) -> None:
        """保存数据统计信息"""
        energies = [item['energy'].item() for item in data]
        forces = [torch.norm(item['forces'], dim=1).max().item() for item in data]
        natoms = [item['natoms'] for item in data]
        
        stats = f"""
OUTCAR数据统计
=============
总帧数: {len(data)}
平均原子数: {np.mean(natoms):.1f}
能量范围: {min(energies):.3f} ~ {max(energies):.3f} eV
平均能量: {np.mean(energies):.3f} eV
最大力: {max(forces):.3f} eV/Å
平均最大力: {np.mean(forces):.3f} eV/Å
"""
        
        with open(stats_file, 'w') as f:
            f.write(stats)
        
        print(f"Statistics saved to {stats_file}")


def create_outcar_dataset(outcar_paths: List[Union[str, Path]],
                         output_path: Union[str, Path],
                         **processor_kwargs) -> None:
    """
    便捷函数：从OUTCAR文件创建MLPot数据集
    
    Args:
        outcar_paths: OUTCAR文件路径列表
        output_path: 输出数据集路径
        **processor_kwargs: OUTCARProcessor的参数
    """
    processor = OUTCARProcessor(**processor_kwargs)
    
    all_data = []
    for outcar_path in outcar_paths:
        print(f"Processing {outcar_path}")
        frames = processor.parse_outcar(outcar_path)
        frames = processor.filter_frames(frames)
        mlpot_data = processor.convert_to_mlpot_format(frames)
        all_data.extend(mlpot_data)
    
    # 保存数据集
    torch.save(all_data, output_path)
    print(f"Created dataset with {len(all_data)} frames: {output_path}")


if __name__ == "__main__":
    # 使用示例
    processor = OUTCARProcessor()
    
    # 处理单个OUTCAR文件
    frames = processor.parse_outcar("OUTCAR")
    filtered_frames = processor.filter_frames(frames)
    mlpot_data = processor.convert_to_mlpot_format(filtered_frames)
    
    # 保存数据
    torch.save(mlpot_data, "outcar_data.pt")
    
    # 批量处理
    # processor.process_outcar_directory("./calculations", "./processed_data")