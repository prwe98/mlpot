"""
MLPot设备配置优化器
支持Mac调试和CUDA服务器部署的自动化配置

AUTHOR: prwe98
TIME: 2025.6.10
"""

import torch
import platform
import warnings
from typing import Dict, Any, Optional, Tuple
import os
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceManager:
    """设备管理器：Mac开发 + CUDA服务器部署优化"""
    
    def __init__(self, debug_mode: bool = True, force_device: Optional[str] = None):
        """
        初始化设备管理器
        
        Args:
            debug_mode: 是否为调试模式（Mac开发环境）
            force_device: 强制使用特定设备 ('cpu', 'cuda', 'mps')
        """
        self.debug_mode = debug_mode
        self.force_device = force_device
        self.system_info = self._get_system_info()
        self.device = self._select_optimal_device()
        self.config = self._get_optimized_config()
        
        logger.info(f"设备管理器初始化完成")
        logger.info(f"   系统: {self.system_info['platform']}")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   调试模式: {self.debug_mode}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        # 添加CUDA_VISIBLE_DEVICES信息
        if info['cuda_available']:
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if visible_devices is not None and visible_devices.strip():
                try:
                    if ',' in visible_devices:
                        physical_gpu_ids = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
                    else:
                        physical_gpu_ids = [int(visible_devices.strip())]
                    info['cuda_visible_devices'] = visible_devices
                    info['physical_gpu_ids'] = physical_gpu_ids
                    info['gpu_allocation_source'] = 'cuda_visible_devices'
                except ValueError:
                    logger.warning(f"无法解析CUDA_VISIBLE_DEVICES: {visible_devices}")
                    info['cuda_visible_devices'] = None
                    info['physical_gpu_ids'] = list(range(info['cuda_device_count']))
                    info['gpu_allocation_source'] = 'auto_detect'
            else:
                info['cuda_visible_devices'] = None
                info['physical_gpu_ids'] = list(range(info['cuda_device_count']))
                info['gpu_allocation_source'] = 'auto_detect'
        else:
            info['cuda_visible_devices'] = None
            info['physical_gpu_ids'] = []
            info['gpu_allocation_source'] = 'no_cuda'
        
        # Mac专用检测
        if info['platform'] == 'Darwin':
            try:
                # 检测Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                info['cpu_brand'] = result.stdout.strip()
                info['is_apple_silicon'] = 'Apple' in info['cpu_brand']
            except:
                info['cpu_brand'] = 'Unknown'
                info['is_apple_silicon'] = False
                
        return info
    
    def _select_optimal_device(self) -> torch.device:
        """选择最优设备"""
        if self.force_device:
            return torch.device(self.force_device)
        
        # CUDA服务器环境
        if self.system_info['cuda_available'] and not self.debug_mode:
            # 生产环境优先使用最好的GPU
            # 考虑CUDA_VISIBLE_DEVICES设置
            best_gpu = self._select_best_gpu()
            return torch.device(f"cuda:{best_gpu}")
        
        # Mac开发环境
        if self.system_info['platform'] == 'Darwin':
            # Apple Silicon优先使用MPS
            if self.system_info['mps_available'] and self.system_info.get('is_apple_silicon', False):
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        # Linux开发环境
        if self.system_info['cuda_available']:
            # 使用PyTorch GPU 0（考虑CUDA_VISIBLE_DEVICES后的第一个GPU）
            return torch.device('cuda:0')
        
        # 默认CPU
        return torch.device('cpu')
    
    def _select_best_gpu(self) -> int:
        """选择最佳GPU（考虑CUDA_VISIBLE_DEVICES设置）"""
        if not self.system_info['cuda_available']:
            return 0
        
        # 获取当前可见的GPU数量（已考虑CUDA_VISIBLE_DEVICES）
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count == 0:
            return 0
            
        best_gpu = 0
        max_memory = 0
        
        # 遍历所有可见的GPU（PyTorch重新编号的0,1,2...）
        for i in range(visible_gpu_count):
            try:
                # torch.cuda.get_device_properties(i) 获取的是PyTorch编号的GPU
                memory = torch.cuda.get_device_properties(i).total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_gpu = i
            except RuntimeError as e:
                # 如果无法访问某个GPU，跳过
                logger.warning(f"无法访问GPU {i}: {e}")
                continue
                
        logger.info(f"选择最佳GPU: PyTorch GPU {best_gpu} (内存: {max_memory/1024**3:.1f}GB)")
        return best_gpu
                
        return best_gpu
    
    def _get_optimized_config(self) -> Dict[str, Any]:
        """获取优化配置"""
        config = {
            'device': self.device,
            'mixed_precision': False,
            'gradient_clipping': 1.0,
            'pin_memory': True,
            'non_blocking': True,
        }
        
        # Mac开发优化
        if self.debug_mode and self.system_info['platform'] == 'Darwin':
            config.update({
                'batch_size': 4,  # 小批次用于调试
                'num_workers': 2,  # Mac上较少的worker
                'accumulate_grad_batches': 4,  # 梯度累积补偿小批次
                'pin_memory': False,  # Mac上可能不需要
                'persistent_workers': False,
                'prefetch_factor': 2,
                'max_neighbors': 30,  # 减少邻居数量
                'cutoff_radius': 5.0,  # 较小的截断半径
                'hidden_dim': 128,  # 较小的隐藏维度
                'num_layers': 3,  # 较少的层数
                'precision': 32,  # 使用FP32确保数值稳定性
            })
            
            # MPS特殊配置
            if str(self.device) == 'mps':
                config.update({
                    'mixed_precision': False,  # MPS暂不支持AMP
                    'compile_model': False,  # MPS可能不支持torch.compile
                })
        
        # CUDA服务器优化
        elif str(self.device).startswith('cuda'):
            config.update({
                'batch_size': 16,  # 更大的批次
                'num_workers': 8,  # 更多的worker
                'accumulate_grad_batches': 1,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,
                'max_neighbors': 50,
                'cutoff_radius': 6.0,
                'hidden_dim': 512,  # 更大的模型
                'num_layers': 6,
                'mixed_precision': True,  # 使用AMP加速
                'compile_model': True,  # 使用torch.compile优化
                'precision': 16,  # 使用FP16
            })
            
            # 多GPU配置
            if self.system_info['cuda_device_count'] > 1:
                config.update({
                    'strategy': 'ddp',  # 分布式数据并行
                    'sync_batchnorm': True,
                })
        
        # CPU配置
        else:
            config.update({
                'batch_size': 2,
                'num_workers': 4,
                'accumulate_grad_batches': 8,
                'pin_memory': False,
                'max_neighbors': 20,
                'cutoff_radius': 4.0,
                'hidden_dim': 64,
                'num_layers': 2,
                'precision': 32,
            })
        
        return config
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """获取DataLoader配置"""
        base_config = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'],
            'drop_last': True,
            'shuffle': True,
        }
        
        # 添加额外配置
        if 'persistent_workers' in self.config:
            base_config['persistent_workers'] = self.config['persistent_workers']
        if 'prefetch_factor' in self.config:
            base_config['prefetch_factor'] = self.config['prefetch_factor']
            
        return base_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'hidden_dim': self.config['hidden_dim'],
            'num_layers': self.config['num_layers'],
            'max_neighbors': self.config['max_neighbors'],
            'cutoff_radius': self.config['cutoff_radius'],
            'device': self.device,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return {
            'mixed_precision': self.config['mixed_precision'],
            'gradient_clipping': self.config['gradient_clipping'],
            'accumulate_grad_batches': self.config['accumulate_grad_batches'],
            'precision': self.config['precision'],
        }
    
    def optimize_for_deployment(self):
        """切换到部署模式（CUDA服务器）"""
        logger.info("切换到部署模式...")
        self.debug_mode = False
        self.device = self._select_optimal_device()
        self.config = self._get_optimized_config()
        logger.info(f"   新设备: {self.device}")
        logger.info(f"   批次大小: {self.config['batch_size']}")
    
    def optimize_for_debug(self):
        """切换到调试模式（Mac开发）"""
        logger.info("切换到调试模式...")
        self.debug_mode = True
        self.device = self._select_optimal_device()
        self.config = self._get_optimized_config()
        logger.info(f"   新设备: {self.device}")
        logger.info(f"   批次大小: {self.config['batch_size']}")
    
    def print_system_info(self):
        """打印系统信息"""
        print("\n" + "="*60)
        print("MLPot设备配置信息")
        print("="*60)
        print(f"操作系统: {self.system_info['platform']} {self.system_info['machine']}")
        print(f"处理器: {self.system_info.get('cpu_brand', self.system_info['processor'])}")
        print(f"Python版本: {self.system_info['python_version']}")
        print(f"PyTorch版本: {self.system_info['torch_version']}")
        
        print(f"\n🎯 设备配置:")
        print(f"当前设备: {self.device}")
        print(f"调试模式: {self.debug_mode}")
        
        print(f"\n⚡ CUDA信息:")
        print(f"CUDA可用: {self.system_info['cuda_available']}")
        if self.system_info['cuda_available']:
            print(f"CUDA版本: {self.system_info['cuda_version']}")
            print(f"可见GPU数量: {self.system_info['cuda_device_count']}")
            
            # 显示GPU分配信息
            allocation_source = self.system_info.get('gpu_allocation_source', 'unknown')
            print(f"GPU分配来源: {allocation_source}")
            
            if allocation_source == 'cuda_visible_devices':
                visible_devices = self.system_info.get('cuda_visible_devices')
                physical_ids = self.system_info.get('physical_gpu_ids', [])
                print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
                print(f"物理GPU ID: {physical_ids}")
                print(f"PyTorch GPU ID: {list(range(self.system_info['cuda_device_count']))}")
                print("说明: 队列系统或手动设置了GPU分配")
            elif allocation_source == 'auto_detect':
                print("说明: 自动检测到的所有GPU")
            
            # 显示详细GPU信息
            for i in range(self.system_info['cuda_device_count']):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    
                    if allocation_source == 'cuda_visible_devices':
                        physical_ids = self.system_info.get('physical_gpu_ids', [])
                        if i < len(physical_ids):
                            physical_id = physical_ids[i]
                            print(f"  PyTorch GPU {i} (物理GPU {physical_id}): {props.name} ({memory_gb:.1f}GB)")
                        else:
                            print(f"  PyTorch GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                    else:
                        print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                except RuntimeError as e:
                    print(f"  GPU {i}: 无法访问 ({e})")
        
        if self.system_info['platform'] == 'Darwin':
            print(f"\n🍎 Mac特殊功能:")
            print(f"Apple Silicon: {self.system_info.get('is_apple_silicon', False)}")
            print(f"MPS可用: {self.system_info['mps_available']}")
        
        print(f"\n推荐配置:")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"工作进程: {self.config['num_workers']}")
        print(f"模型维度: {self.config['hidden_dim']}")
        print(f"层数: {self.config['num_layers']}")
        print(f"混合精度: {self.config['mixed_precision']}")
        print("="*60)
    
    def create_deployment_script(self, output_path: str = "deploy_config.py"):
        """创建部署配置脚本"""
        import datetime
        script_content = f'''"""
自动生成的MLPot部署配置
生成时间: {datetime.datetime.now()}
"""

import torch
from mlpot.device_config import DeviceManager

# 创建部署设备管理器
device_manager = DeviceManager(debug_mode=False)

# 设备配置
DEVICE = device_manager.device
MODEL_CONFIG = device_manager.get_model_config()
TRAINING_CONFIG = device_manager.get_training_config()
DATALOADER_CONFIG = device_manager.get_dataloader_config()

# 模型参数
HIDDEN_DIM = {self.config['hidden_dim']}
NUM_LAYERS = {self.config['num_layers']}
BATCH_SIZE = {self.config['batch_size']}
MIXED_PRECISION = {self.config['mixed_precision']}

print(f"部署设备: {{DEVICE}}")
print(f"批次大小: {{BATCH_SIZE}}")
print(f"混合精度: {{MIXED_PRECISION}}")
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"部署配置已保存到: {output_path}")


def get_device_manager(debug_mode: bool = None, force_device: str = None) -> DeviceManager:
    """
    便捷函数：获取设备管理器
    
    Args:
        debug_mode: 调试模式，None时自动检测
        force_device: 强制使用的设备
    
    Returns:
        配置好的设备管理器
    """
    # 自动检测调试模式
    if debug_mode is None:
        # 检测是否在Mac或开发环境
        is_mac = platform.system() == 'Darwin'
        is_interactive = False
        try:
            # 安全检测交互式环境
            is_interactive = hasattr(__builtins__, '__IPYTHON__') or 'ipykernel' in str(type(get_ipython()))
        except NameError:
            is_interactive = False
        debug_mode = is_mac or is_interactive
    
    return DeviceManager(debug_mode=debug_mode, force_device=force_device)


# 便捷函数
def auto_device() -> torch.device:
    """自动选择最优设备"""
    return get_device_manager().device


def auto_config() -> Dict[str, Any]:
    """自动获取优化配置"""
    return get_device_manager().config


if __name__ == "__main__":
    # 演示用法
    print("MLPot设备配置演示")
    
    # 创建设备管理器
    dm = get_device_manager()
    dm.print_system_info()
    
    # 测试模式切换
    print("\n 测试模式切换:")
    print("当前模式:", "调试" if dm.debug_mode else "部署")
    
    if dm.debug_mode:
        print("切换到部署模式...")
        dm.optimize_for_deployment()
    else:
        print("切换到调试模式...")
        dm.optimize_for_debug()
    
    # 创建部署脚本
    dm.create_deployment_script()
