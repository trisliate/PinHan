"""
改进版训练工具 - 支持最优模型保存和日志记录
"""
import logging
import orjson
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch

class BestModelTracker:
    """追踪并保存最优模型."""
    def __init__(self, save_dir: Path, patience: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience = patience
        self.patience_counter = 0
        self.metrics_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def update(self, loss: float, model, optimizer, epoch: int, extra_metrics: Dict = None) -> bool:
        """
        更新最优模型。
        
        Args:
            loss: 当前 loss
            model: 模型
            optimizer: 优化器
            epoch: epoch 号
            extra_metrics: 额外指标
        
        Returns:
            是否继续训练（如果达到 patience 限制则返回 False）
        """
        metrics = {
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat(),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        
        self.metrics_history.append(metrics)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # 保存最优模型
            self._save_best_model(model, optimizer, epoch, loss)
            self.logger.info(f"✅ 新的最优模型: Loss={loss:.4f} at Epoch {epoch}")
            return True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.logger.warning(
                    f"⚠️ {self.patience} 轮无改进，建议提前停止训练 "
                    f"(最优模型在 Epoch {self.best_epoch}，Loss={self.best_loss:.4f})"
                )
                return False
            return True
    
    def _save_best_model(self, model, optimizer, epoch: int, loss: float):
        """保存最优模型."""
        best_model_path = self.save_dir / 'best_model.pt'
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
        }, best_model_path)
    
    def save_metrics(self):
        """保存训练指标."""
        metrics_file = self.save_dir / 'logs' / 'metrics.json'
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_file, 'wb') as f:
            f.write(orjson.dumps({
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch,
                'total_epochs': len(self.metrics_history),
                'metrics_history': self.metrics_history,
            }, option=orjson.OPT_INDENT_2))


class CheckpointManager:
    """管理检查点文件，只保留最近 K 个."""
    def __init__(self, save_dir: Path, keep_latest: int = 3):
        self.save_dir = Path(save_dir)
        self.keep_latest = keep_latest
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, model, optimizer, epoch: int, metrics: Dict = None):
        """保存检查点，清理旧文件."""
        ckpt_path = self.save_dir / f'checkpoint_epoch{epoch}.pt'
        
        data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if metrics:
            data.update(metrics)
        
        torch.save(data, ckpt_path)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """只保留最近 K 个检查点."""
        ckpt_files = sorted(self.save_dir.glob('checkpoint_epoch*.pt'))
        
        if len(ckpt_files) > self.keep_latest:
            to_delete = ckpt_files[:-self.keep_latest]
            for f in to_delete:
                f.unlink()
                self.logger.debug(f"删除旧检查点: {f.name}")


def setup_logging(save_dir: Path, log_level=logging.INFO):
    """
    配置日志系统。
    
    Args:
        save_dir: 日志保存目录
        log_level: 日志级别
    """
    from logging.handlers import RotatingFileHandler
    
    save_dir = Path(save_dir)
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取或创建 logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 移除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器（轮转日志）
    log_file = log_dir / 'train.log'
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_training_config(save_dir: Path, config: Dict[str, Any]):
    """保存训练配置."""
    config_file = Path(save_dir) / 'logs' / 'config.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'wb') as f:
        f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))


def load_best_model(model_path: Path, model, device='cpu'):
    """
    加载最优模型权重。
    
    Args:
        model_path: 模型文件路径
        model: 模型实例
        device: 设备
    """
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    logger = logging.getLogger(__name__)
    logger.info(
        f"✅ 加载最优模型: {model_path.name} "
        f"(Epoch {ckpt.get('epoch', '?')}, Loss={ckpt.get('loss', '?')})"
    )
    
    return model
