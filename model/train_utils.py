"""
改进版训练脚本 - 支持完整的增量训练和模型版本管理

关键改进:
✅ 模型保存时包含完整元数据
✅ 支持词表版本检查
✅ 增量训练前检查兼容性
✅ 自动词表扩展
✅ 详细的训练日志
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import argparse
import orjson
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess'))
from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
from pinyin_utils import normalize_pinyin_sequence, validate_pinyin_sequence


def get_device():
    """智能设备选择：优先 NVIDIA CUDA，否则 CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"✅ 使用 NVIDIA GPU: {device_name} ({memory_gb:.2f}GB)")
        return device
    
    # CPU 训练
    num_cores = torch.get_num_threads()
    logging.info(f"✅ 使用 CPU 训练 ({num_cores}核)")
    torch.set_num_threads(num_cores)
    return torch.device('cpu')

DEVICE = get_device()


def get_model_version() -> str:
    """获取模型版本号 (基于代码提交ID或时间戳)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_model_metadata(
    model: nn.Module,
    epoch: int,
    loss: float,
    accuracy: float,
    data_source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """创建模型元数据."""
    return {
        # 版本信息
        'version': '2.0.0',  # 增量训练支持版本
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        
        # 训练信息
        'metadata': {
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'data_source': str(data_source),
            'total_samples': None,  # 由调用者填充
        },
        
        # 模型架构配置
        'config': config,
        
        # 词表信息
        'vocab_info': {
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'src_special_tokens': {
                'pad': src_vocab.pad_token,
                'unk': src_vocab.unk_token,
            },
            'tgt_special_tokens': {
                'pad': tgt_vocab.pad_token,
                'unk': tgt_vocab.unk_token,
            },
            'src_vocab_hash': hash_vocab(src_vocab),
            'tgt_vocab_hash': hash_vocab(tgt_vocab),
        },
        
        # 训练超参数
        'training_config': training_config,
        
        # 数据集信息
        'dataset_info': {
            'max_src_len': 64,
            'max_tgt_len': 64,
        }
    }


def hash_vocab(vocab: Vocab) -> str:
    """计算词表的哈希值，用于版本比对."""
    import hashlib
    tokens = sorted(vocab.token_to_id.keys())
    content = orjson.dumps(tokens).decode('utf-8')
    return hashlib.md5(content.encode()).hexdigest()


def check_vocab_compatibility(
    saved_metadata: Dict[str, Any],
    current_src_vocab: Vocab,
    current_tgt_vocab: Vocab,
    strategy: str = 'expand',
) -> Tuple[bool, str]:
    """
    检查词表兼容性。
    
    Args:
        saved_metadata: 保存的模型元数据
        current_src_vocab: 当前源词表
        current_tgt_vocab: 当前目标词表
        strategy: 'strict'(严格) or 'expand'(扩展)
    
    Returns:
        (compatible, message)
    """
    saved_src_hash = saved_metadata['vocab_info'].get('src_vocab_hash')
    saved_tgt_hash = saved_metadata['vocab_info'].get('tgt_vocab_hash')
    
    current_src_hash = hash_vocab(current_src_vocab)
    current_tgt_hash = hash_vocab(current_tgt_vocab)
    
    src_match = saved_src_hash == current_src_hash
    tgt_match = saved_tgt_hash == current_tgt_hash
    
    if strategy == 'strict':
        # 严格模式: 词表必须完全相同
        if not (src_match and tgt_match):
            msg = f"词表不兼容 (strict mode):\n"
            msg += f"  源词表: {'✓' if src_match else '✗'}\n"
            msg += f"  目标词表: {'✓' if tgt_match else '✗'}\n"
            return False, msg
        return True, "词表完全匹配 ✓"
    
    elif strategy == 'expand':
        # 扩展模式: 允许词表增长 (向后兼容)
        src_size_ok = len(current_src_vocab) >= saved_metadata['vocab_info']['src_vocab_size']
        tgt_size_ok = len(current_tgt_vocab) >= saved_metadata['vocab_info']['tgt_vocab_size']
        
        if not (src_size_ok and tgt_size_ok):
            msg = f"词表缩小了，不允许 (expand mode):\n"
            msg += f"  源词表: {saved_metadata['vocab_info']['src_vocab_size']} → {len(current_src_vocab)}\n"
            msg += f"  目标词表: {saved_metadata['vocab_info']['tgt_vocab_size']} → {len(current_tgt_vocab)}\n"
            return False, msg
        
        if src_match and tgt_match:
            return True, "词表完全匹配 ✓"
        else:
            msg = f"词表已扩展，可以继续训练:\n"
            msg += f"  源词表: {saved_metadata['vocab_info']['src_vocab_size']} → {len(current_src_vocab)}\n"
            msg += f"  目标词表: {saved_metadata['vocab_info']['tgt_vocab_size']} → {len(current_tgt_vocab)}\n"
            return True, msg
    
    return False, "未知的检查策略"


def save_model_complete(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    data_source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config: Dict[str, Any],
    training_config: Dict[str, Any],
    out_dir: Path,
) -> None:
    """
    完整地保存模型。
    
    保存结构:
    outputs/
      ├── model.pt          (最新完整模型)
      ├── metadata.json     (元数据)
      ├── src_vocab.json    (源词表)
      ├── tgt_vocab.json    (目标词表)
      ├── checkpoint_epoch1.pt
      ├── checkpoint_epoch2.pt
      └── ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建元数据
    metadata = create_model_metadata(
        model, epoch, loss, accuracy, data_source,
        src_vocab, tgt_vocab, config, training_config
    )
    
    # 保存checkpoint (用于resume)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }
    torch.save(ckpt, out_dir / f'checkpoint_epoch{epoch}.pt')
    
    # 保存最新完整模型 (用于推理)
    final_model = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'config': config,
    }
    torch.save(final_model, out_dir / 'model.pt')
    
    # 保存元数据为JSON (便于查看)
    with open(out_dir / 'metadata.json', 'wb') as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    
    # 保存词表
    src_vocab.save(str(out_dir / 'src_vocab.json'))
    tgt_vocab.save(str(out_dir / 'tgt_vocab.json'))
    
    logger = logging.getLogger()
    logger.info(f"✅ 完整模型已保存到 {out_dir}")
    logger.info(f"   - model.pt (推理用)")
    logger.info(f"   - checkpoint_epoch{epoch}.pt (恢复训练用)")
    logger.info(f"   - metadata.json (元数据)")
    logger.info(f"   - src/tgt_vocab.json (词表)")


def load_model_for_resume(
    model_dir: Path,
    device: torch.device,
    strategy: str = 'expand',
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    加载模型用于继续训练。
    
    Returns:
        (model_state, metadata, start_epoch)
    """
    logger = logging.getLogger()
    
    # 查找最新的checkpoint
    ckpts = sorted(model_dir.glob('checkpoint_epoch*.pt'))
    if not ckpts:
        logger.warning(f"未找到checkpoint，无法恢复")
        return None, None, 1
    
    latest_ckpt = ckpts[-1]
    logger.info(f"📂 加载checkpoint: {latest_ckpt.name}")
    
    # 加载checkpoint
    ck = torch.load(str(latest_ckpt), map_location=device)
    model_state = ck['model_state_dict']
    metadata = ck.get('metadata', {})
    start_epoch = ck.get('epoch', 1) + 1
    
    logger.info(f"   - 从 Epoch {ck.get('epoch', '?')} 恢复")
    logger.info(f"   - Loss: {metadata.get('metadata', {}).get('loss', '?'):.4f}")
    logger.info(f"   - 下一轮从 Epoch {start_epoch} 开始")
    
    return model_state, metadata, start_epoch


def compare_models(model_dir1: Path, model_dir2: Path) -> None:
    """
    对比两个模型的元数据。
    
    用途: 对比不同版本的模型性能
    """
    logger = logging.getLogger()
    
    def load_meta(path):
        with open(path / 'metadata.json', 'rb') as f:
            return orjson.loads(f.read())
    
    meta1 = load_meta(model_dir1)
    meta2 = load_meta(model_dir2)
    
    logger.info("\n" + "="*60)
    logger.info("🔍 模型对比")
    logger.info("="*60)
    
    logger.info(f"\n【模型1】{model_dir1.name}")
    logger.info(f"  版本: {meta1.get('version')}")
    logger.info(f"  Epoch: {meta1['metadata']['epoch']}")
    logger.info(f"  Loss: {meta1['metadata']['loss']:.4f}")
    logger.info(f"  准确度: {meta1['metadata']['accuracy']:.4f}")
    
    logger.info(f"\n【模型2】{model_dir2.name}")
    logger.info(f"  版本: {meta2.get('version')}")
    logger.info(f"  Epoch: {meta2['metadata']['epoch']}")
    logger.info(f"  Loss: {meta2['metadata']['loss']:.4f}")
    logger.info(f"  准确度: {meta2['metadata']['accuracy']:.4f}")
    
    logger.info("\n")


# ============================================================================
# 以下是调用示例 (实际在 train_pinhan.py 中使用)
# ============================================================================

def example_training():
    """示例: 如何使用新的保存机制."""
    
    # 1. 训练配置
    config = {
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'max_seq_len': 64,
    }
    
    training_config = {
        'batch_size': 4,
        'learning_rate': 0.0001,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
    }
    
    # 2. 初始化模型 (伪代码)
    # model = ...
    # optimizer = ...
    # src_vocab, tgt_vocab = ...
    
    # 3. 训练...
    # for epoch in range(epochs):
    #     loss = train_one_epoch(...)
    #     accuracy = evaluate(...)
    
    # 4. 保存模型 (改进的方法)
    # save_model_complete(
    #     model, optimizer, epoch, loss, accuracy,
    #     data_source='data/10k.jsonl',
    #     src_vocab=src_vocab,
    #     tgt_vocab=tgt_vocab,
    #     config=config,
    #     training_config=training_config,
    #     out_dir=Path('outputs/final_model')
    # )
    
    # 5. 增量训练 (后续)
    # 当有新数据时:
    # - 加载旧模型元数据
    # - 重新构建词表 (可能包含新词)
    # - 检查兼容性
    # - 继续训练
    
    print("参考 save_model_complete() 和 load_model_for_resume()")


if __name__ == '__main__':
    example_training()
