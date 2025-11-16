"""
完整的训练保存和断点管理系统 - 支持增量训练和完整恢复
"""
import logging
import orjson
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class TrainingCheckpointManager:
    """
    完整的训练管理系统：
    - 自动保存最优模型
    - 智能管理检查点（只保留必要的）
    - 完整的增量训练恢复
    - 断点续训支持
    """
    
    def __init__(self, save_dir: Path, keep_checkpoints: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_checkpoints = keep_checkpoints
        
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.training_history = []
        
        # 日志和配置目录
        self.log_dir = self.save_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        loss: float,
        src_vocab,
        tgt_vocab,
        metrics: Dict[str, float] = None,
    ) -> Path:
        """
        保存训练检查点（包含完整的恢复信息）。
        
        Args:
            epoch: epoch 号
            model: 模型
            optimizer: 优化器
            loss: 当前 loss
            src_vocab: 源词表
            tgt_vocab: 目标词表
            metrics: 额外指标
        
        Returns:
            保存的文件路径
        """
        checkpoint = {
            # 基本信息
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat(),
            
            # 模型和优化器
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            # 词表验证信息
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'src_vocab_tokens': len(src_vocab.id_to_token),
            'tgt_vocab_tokens': len(tgt_vocab.id_to_token),
            
            # 恢复信息
            'recovery_info': {
                'can_resume': True,
                'next_epoch': epoch + 1,
                'best_loss_so_far': self.best_loss,
                'best_epoch_so_far': self.best_epoch,
            },
            
            # 额外指标
            'metrics': metrics or {},
        }
        
        ckpt_path = self.save_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        self.logger.debug(f"✅ 保存检查点: {ckpt_path.name}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        return ckpt_path
    
    def save_best_model(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        loss: float,
        src_vocab,
        tgt_vocab,
        metrics: Dict[str, float] = None,
    ) -> Tuple[Path, bool]:
        """
        保存最优模型（如果当前loss更低）。
        
        Returns:
            (保存路径, 是否为新的最优模型)
        """
        is_new_best = loss < self.best_loss
        
        if is_new_best:
            self.best_loss = loss
            self.best_epoch = epoch
            
            best_model = {
                # 基本信息
                'epoch': epoch,
                'loss': float(loss),
                'timestamp': datetime.now().isoformat(),
                
                # 模型
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                # 词表
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'src_vocab_tokens': len(src_vocab.id_to_token),
                'tgt_vocab_tokens': len(tgt_vocab.id_to_token),
                
                # 元数据
                'best_flag': True,
                'metrics': metrics or {},
            }
            
            best_path = self.save_dir / 'best_model.pt'
            torch.save(best_model, best_path)
            self.logger.info(f"🌟 新的最优模型: Loss={loss:.4f} at Epoch {epoch}")
            
            return best_path, True
        
        return None, False
    
    def _cleanup_old_checkpoints(self):
        """只保留最近 K 个检查点，删除旧的。"""
        ckpts = sorted(self.save_dir.glob('checkpoint_epoch*.pt'))
        
        if len(ckpts) > self.keep_checkpoints:
            to_delete = ckpts[:-self.keep_checkpoints]
            for ckpt in to_delete:
                ckpt.unlink()
                self.logger.debug(f"🗑️  删除旧检查点: {ckpt.name}")
    
    def try_resume_from_checkpoint(self, model, optimizer, device='cpu'):
        """
        尝试从最新的检查点恢复训练。
        
        Returns:
            (是否恢复成功, 下一个epoch, 检查点信息)
        """
        ckpts = sorted(self.save_dir.glob('checkpoint_epoch*.pt'))
        if not ckpts:
            self.logger.info("ℹ️  没有检查点，从头开始训练")
            return False, 1, None
        
        latest_ckpt = ckpts[-1]
        self.logger.info(f"📂 找到检查点: {latest_ckpt.name}")
        
        try:
            ckpt = torch.load(str(latest_ckpt), map_location=device)
            
            # 验证检查点完整性
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss']
            missing = [k for k in required_keys if k not in ckpt]
            if missing:
                self.logger.error(f"❌ 检查点缺少必要字段: {missing}")
                return False, 1, None
            
            # 恢复模型和优化器
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            # 恢复最优值
            if 'recovery_info' in ckpt:
                recovery = ckpt['recovery_info']
                self.best_loss = recovery.get('best_loss_so_far', float('inf'))
                self.best_epoch = recovery.get('best_epoch_so_far', 0)
            
            next_epoch = ckpt.get('recovery_info', {}).get('next_epoch', ckpt['epoch'] + 1)
            
            self.logger.info(
                f"✅ 已恢复: Epoch {ckpt['epoch']}, Loss={ckpt['loss']:.4f}\n"
                f"   下一轮从 Epoch {next_epoch} 开始\n"
                f"   当前最优: Loss={self.best_loss:.4f} at Epoch {self.best_epoch}"
            )
            
            return True, next_epoch, ckpt
            
        except Exception as e:
            self.logger.error(f"❌ 恢复失败: {e}")
            return False, 1, None
    
    def validate_vocab_compatibility(
        self,
        ckpt: Dict,
        src_vocab_size: int,
        tgt_vocab_size: int,
        allow_size_increase: bool = False,
    ) -> Tuple[bool, str]:
        """
        验证检查点的词表与当前词表是否兼容。
        
        Args:
            ckpt: 检查点数据
            src_vocab_size: 当前源词表大小
            tgt_vocab_size: 当前目标词表大小
            allow_size_increase: 是否允许词表扩展（用于增量训练）
        
        Returns:
            (是否兼容, 消息)
        """
        ckpt_src = ckpt.get('src_vocab_size')
        ckpt_tgt = ckpt.get('tgt_vocab_size')
        
        if ckpt_src is None or ckpt_tgt is None:
            return True, "✅ 旧检查点，跳过词表验证"
        
        messages = []
        
        # 源词表检查
        if ckpt_src != src_vocab_size:
            if allow_size_increase and src_vocab_size > ckpt_src:
                messages.append(
                    f"ℹ️  源词表扩展: {ckpt_src} → {src_vocab_size} "
                    f"(增量训练，+{src_vocab_size - ckpt_src} 个新词汇)"
                )
            else:
                return False, (
                    f"❌ 源词表不兼容:\n"
                    f"   检查点: {ckpt_src}, 当前: {src_vocab_size}"
                )
        else:
            messages.append(f"✅ 源词表兼容: {src_vocab_size}")
        
        # 目标词表检查
        if ckpt_tgt != tgt_vocab_size:
            if allow_size_increase and tgt_vocab_size > ckpt_tgt:
                messages.append(
                    f"ℹ️  目标词表扩展: {ckpt_tgt} → {tgt_vocab_size} "
                    f"(增量训练，+{tgt_vocab_size - ckpt_tgt} 个新字符)"
                )
            else:
                return False, (
                    f"❌ 目标词表不兼容:\n"
                    f"   检查点: {ckpt_tgt}, 当前: {tgt_vocab_size}"
                )
        else:
            messages.append(f"✅ 目标词表兼容: {tgt_vocab_size}")
        
        return True, "\n".join(messages)
    
    def save_training_config(self, config: Dict[str, Any]):
        """保存训练配置（用于复现）。"""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'wb') as f:
            f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))
        self.logger.info(f"📝 训练配置已保存: {config_path.name}")
    
    def update_training_history(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float] = None,
    ):
        """更新训练历史记录。"""
        record = {
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat(),
        }
        if metrics:
            record.update(metrics)
        
        self.training_history.append(record)
    
    def save_training_summary(self):
        """保存训练摘要和指标。"""
        summary = {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.training_history),
            'training_duration': None,  # 由调用者设置
            'history': self.training_history,
        }
        
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'wb') as f:
            f.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
        
        self.logger.info(f"📊 训练摘要已保存: {summary_path.name}")
    
    def get_status_summary(self) -> str:
        """获取当前训练状态摘要。"""
        return (
            f"最优模型: Epoch {self.best_epoch}, Loss={self.best_loss:.4f}\n"
            f"检查点数: {len(list(self.save_dir.glob('checkpoint_epoch*.pt')))}\n"
            f"训练历史: {len(self.training_history)} 轮"
        )


# 便利函数：快速恢复或初始化

def resume_or_init(
    save_dir: Path,
    model,
    optimizer,
    src_vocab,
    tgt_vocab,
    device='cpu',
    allow_vocab_increase=False,
) -> Tuple[int, TrainingCheckpointManager]:
    """
    一体化的恢复或初始化函数。
    
    Returns:
        (下一个epoch, 检查点管理器)
    """
    logger = logging.getLogger(__name__)
    ckpt_mgr = TrainingCheckpointManager(save_dir)
    
    resumed, next_epoch, ckpt_data = ckpt_mgr.try_resume_from_checkpoint(
        model, optimizer, device
    )
    
    if resumed and ckpt_data:
        # 验证词表兼容性
        compatible, msg = ckpt_mgr.validate_vocab_compatibility(
            ckpt_data,
            len(src_vocab),
            len(tgt_vocab),
            allow_size_increase=allow_vocab_increase,
        )
        logger.info(msg)
        
        if not compatible:
            logger.error("❌ 词表不兼容，无法恢复")
            logger.info("💡 提示: 如果进行增量训练，请添加 --allow-vocab-increase 参数")
            raise RuntimeError("词表不兼容")
    
    return next_epoch, ckpt_mgr


# 辅助函数：加载已训练的模型

def load_trained_model(
    model_path: Path,
    model,
    device='cpu',
    load_optimizer=False,
    optimizer=None,
) -> Dict[str, Any]:
    """
    加载已训练的模型（best_model.pt 或 checkpoint）。
    
    Args:
        model_path: 模型文件路径
        model: 模型实例
        device: 设备
        load_optimizer: 是否加载优化器状态
        optimizer: 优化器实例（如果 load_optimizer=True）
    
    Returns:
        加载的检查点信息
    """
    logger = logging.getLogger(__name__)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    if load_optimizer and optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', '?')
    is_best = ckpt.get('best_flag', False)
    
    best_text = "🌟 (最优模型)" if is_best else ""
    logger.info(
        f"✅ 加载模型: {model_path.name}\n"
        f"   Epoch {epoch}, Loss={loss} {best_text}"
    )
    
    return ckpt
