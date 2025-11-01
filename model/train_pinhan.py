"""
训练脚本：拼音到汉字的 Transformer 模型训练 (改进版 v2.0).
    python model/train_pinhan.py --data data/5k.jsonl --save-dir outputs/5k_model --epochs 40 --batch-size 32 --lr 1e-4
"""
import random
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import argparse
import orjson
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess'))
from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
from pinyin_utils import normalize_pinyin_sequence, validate_pinyin_sequence
from checkpoint_manager import TrainingCheckpointManager, resume_or_init, load_trained_model

DATA_PATH = Path('data/clean_wiki.jsonl')

# 🚀 智能设备选择
def _get_device():
    """
    智能设备选择：
    1. 尝试 NVIDIA CUDA
    2. 否则使用 CPU（配置多线程）
    """
    # 优先级 1: NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"✅ 使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    # CPU 训练（配置多线程以充分利用 CPU）
    num_cores = torch.get_num_threads()
    logging.info(f"✅ 使用 CPU 训练 ({num_cores}核)")
    torch.set_num_threads(num_cores)
    return torch.device('cpu')

DEVICE = _get_device()


class PinyinHanziDataset(Dataset):
    """拼音->汉字数据集."""
    def __init__(
        self,
        path: Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
        normalize_pinyin: bool = False,
        skip_invalid: bool = True,
    ) -> None:
        self.samples = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.normalize_pinyin = normalize_pinyin
        self.skip_invalid = skip_invalid
        self.stats = {
            'total_lines': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'mismatched_length': 0,
            'invalid_pinyin': 0,
            'empty_data': 0,
        }
        self._load_data(path)

    def _load_data(self, path: Path) -> None:
        """加载和处理数据."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats['total_lines'] += 1
                line = line.strip()
                if not line:
                    self.stats['empty_data'] += 1
                    continue
                try:
                    j = orjson.loads(line)
                except Exception:
                    self.stats['invalid_samples'] += 1
                    continue
                hanzi = j.get('hanzi', '').strip()
                pinyin = j.get('pinyin', '').strip()
                if not hanzi or not pinyin:
                    self.stats['empty_data'] += 1
                    continue
                if self.normalize_pinyin:
                    try:
                        pinyin = normalize_pinyin_sequence(pinyin)
                    except Exception:
                        self.stats['invalid_pinyin'] += 1
                        if self.skip_invalid:
                            continue
                if self.skip_invalid and not validate_pinyin_sequence(pinyin):
                    self.stats['invalid_pinyin'] += 1
                    continue
                src_tokens = pinyin.split()
                tgt_tokens = list(hanzi)
                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    self.stats['empty_data'] += 1
                    continue
                if len(src_tokens) != len(tgt_tokens):
                    self.stats['mismatched_length'] += 1
                    if self.skip_invalid:
                        continue
                if len(src_tokens) > (self.max_src_len - 2):
                    src_tokens = src_tokens[:self.max_src_len - 2]
                    tgt_tokens = tgt_tokens[:self.max_src_len - 2]
                if len(tgt_tokens) > (self.max_tgt_len - 2):
                    tgt_tokens = tgt_tokens[:self.max_tgt_len - 2]
                self.samples.append((src_tokens, tgt_tokens))
                self.stats['valid_samples'] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]

    def print_statistics(self) -> None:
        """打印统计信息."""
        logger = logging.getLogger()
        logger.info("\n=== 数据集统计 ===")
        for key, val in self.stats.items():
            logger.info(f"{key}: {val}")
        if self.stats['total_lines'] > 0:
            valid_ratio = self.stats['valid_samples'] / self.stats['total_lines'] * 100
            logger.info(f"有效样本比例: {valid_ratio:.2f}%")


def collate_fn(batch: list, src_vocab: Vocab, tgt_vocab: Vocab) -> tuple:
    """批处理函数."""
    srcs, tgts = zip(*batch)
    src_ids = [src_vocab.encode(list(s), add_bos_eos=True) for s in srcs]
    tgt_ids = [tgt_vocab.encode(list(t), add_bos_eos=True) for t in tgts]
    max_src = max(len(x) for x in src_ids)
    max_tgt = max(len(x) for x in tgt_ids)
    pad_id_src = src_vocab.token_to_id[src_vocab.pad_token]
    pad_id_tgt = tgt_vocab.token_to_id[tgt_vocab.pad_token]
    src_padded = [x + [pad_id_src] * (max_src - len(x)) for x in src_ids]
    tgt_padded = [x + [pad_id_tgt] * (max_tgt - len(x)) for x in tgt_ids]
    src_tensor = torch.LongTensor(src_padded).transpose(0, 1)
    tgt_tensor = torch.LongTensor(tgt_padded).transpose(0, 1)
    return src_tensor, tgt_tensor


def build_vocabs(path: Path, max_pinyin_tokens: int = 50000, max_hanzi_chars: int = 50000) -> tuple:
    """构建词表."""
    logger = logging.getLogger()
    pinyin_counter = {}
    hanzi_counter = {}
    polyphonic_stats = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = orjson.loads(line)
            except Exception:
                continue
            p = j.get('pinyin', '')
            h = j.get('hanzi', '')
            if not p or not h:
                continue
            pinyins = p.strip().split()
            hanzis = list(h.strip())
            for pinyin_token, hanzi_char in zip(pinyins, hanzis):
                pinyin_counter[pinyin_token] = pinyin_counter.get(pinyin_token, 0) + 1
                hanzi_counter[hanzi_char] = hanzi_counter.get(hanzi_char, 0) + 1
                if hanzi_char not in polyphonic_stats:
                    polyphonic_stats[hanzi_char] = set()
                polyphonic_stats[hanzi_char].add(pinyin_token)
    polyphonic_hanzis = {h: p for h, p in polyphonic_stats.items() if len(p) > 1}
    pinyin_sorted = sorted(pinyin_counter.items(), key=lambda x: -x[1])
    hanzi_sorted = sorted(hanzi_counter.items(), key=lambda x: -x[1])
    pinyin_tokens = [t for t, _ in pinyin_sorted][:max_pinyin_tokens]
    hanzi_tokens = [t for t, _ in hanzi_sorted][:max_hanzi_chars]
    src_vocab = Vocab(pinyin_tokens)
    tgt_vocab = Vocab(hanzi_tokens)
    logger.info(f"拼音 tokens: {len(pinyin_tokens)}")
    logger.info(f"汉字总数: {len(hanzi_tokens)}")
    logger.info(f"多音字数量: {len(polyphonic_hanzis)}")
    return src_vocab, tgt_vocab


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """训练一个 epoch, 返回 (avg_loss, avg_grad_norm)."""
    logger = logging.getLogger()
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    epoch_start = time.time()
    num_batches = len(dataloader)
    
    for i, (src, tgt) in enumerate(dataloader):
        batch_start = time.time()
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_key_padding_mask = (src.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
        tgt_key_padding_mask = (tgt_input.transpose(0, 1) == tgt_vocab.token_to_id[tgt_vocab.pad_token])
        
        # 🔧 转换mask类型为float, 保证一致性
        tgt_mask = tgt_mask.float()
        src_key_padding_mask = src_key_padding_mask.float()
        tgt_key_padding_mask = tgt_key_padding_mask.float()
        optimizer.zero_grad()
        logits = model(
            src,
            tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        
        # 监控梯度
        grad_norm = log_gradient_stats(model, logger)
        total_grad_norm += grad_norm
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        # 定期输出进度
        if i % max(1, num_batches // 10) == 0:
            batch_time = time.time() - batch_start
            progress = (i + 1) / num_batches
            elapsed = time.time() - epoch_start
            eta = elapsed / (i + 1) * (num_batches - i - 1) if i > 0 else 0
            
            logger.info(
                f"Epoch {epoch} [{i+1:>4d}/{num_batches}] "
                f"loss={loss.item():.4f} "
                f"grad_norm={grad_norm:.4f} "
                f"time={batch_time:.2f}s "
                f"ETA={eta:.0f}s"
            )
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(dataloader)
    avg_grad_norm = total_grad_norm / len(dataloader)
    
    return avg_loss, avg_grad_norm, epoch_time


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    out_dir: Path,
    src_vocab_size: int = None,
    tgt_vocab_size: int = None,
) -> None:
    """
    旧版保存接口 (兼容性保留，不推荐使用).
    推荐: 使用 TrainingCheckpointManager.save_checkpoint()
    """
    logger = logging.getLogger(__name__)
    logger.warning(
        "⚠️  使用 save_checkpoint() 已过时。\n"
        "    推荐: 使用 TrainingCheckpointManager.save_checkpoint()\n"
        "    新接口提供自动文件清理和最优模型追踪"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
    }
    torch.save(ckpt, out_dir / f'checkpoint_epoch{epoch}.pt')


def log_gradient_stats(model: nn.Module, logger: logging.Logger) -> float:
    """计算并记录梯度统计."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    total_norm = (total_norm) ** 0.5
    
    if total_norm > 100:
        logger.warning(f"⚠️  梯度范数过大: {total_norm:.4f} (可能爆炸)")
    elif total_norm < 1e-8:
        logger.warning(f"⚠️  梯度范数过小: {total_norm:.4e} (可能消失)")
    
    return total_norm


def get_memory_usage() -> float:
    """获取当前GPU/CPU内存使用 (GB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


def log_training_start(
    logger: logging.Logger,
    model: nn.Module,
    data_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """记录训练开始时的详细信息."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("="*70)
    logger.info("🚀 训练配置")
    logger.info("="*70)
    logger.info(f"设备: {device}")
    logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    logger.info(f"数据集大小: {data_size:,}")
    logger.info(f"批大小: {batch_size}, 总批数: {(data_size + batch_size - 1) // batch_size}")
    logger.info(f"轮数: {epochs}, 学习率: {lr:.6f}")
    logger.info(f"初始内存: {get_memory_usage():.2f} GB")
    logger.info("="*70)


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """配置日志."""
    fmt = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt, force=True)
    logger = logging.getLogger()
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


def main() -> None:
    """主训练函数."""
    parser = argparse.ArgumentParser(description='训练拼音->汉字 Transformer')
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save-dir', type=str, default='outputs/pinhan_model', help='保存目录')
    parser.add_argument('--max-samples', type=int, default=0, help='最大样本数（0 表示全部）')
    parser.add_argument('--resume', action='store_true', help='从 checkpoint 恢复')
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')
    parser.add_argument('--normalize-pinyin', action='store_true', help='规范化拼音')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_log = f'outputs/train_{timestamp}.log'
    log_file = args.log_file or default_log
    logger = setup_logging(log_file)
    logger.info("开始训练...")
    logger.info(f"数据路径: {args.data}")
    logger.info(f"批大小: {args.batch_size}, 学习率: {args.lr}, 轮数: {args.epochs}")
    logger.info("构建词表...")
    src_vocab, tgt_vocab = build_vocabs(Path(args.data))
    logger.info(f"源词表: {len(src_vocab)}, 目标词表: {len(tgt_vocab)}")
    logger.info("加载数据集...")
    ds = PinyinHanziDataset(
        Path(args.data),
        src_vocab,
        tgt_vocab,
        max_src_len=64,
        max_tgt_len=64,
        normalize_pinyin=args.normalize_pinyin,
    )
    ds.print_statistics()
    assert src_vocab.pad_token == tgt_vocab.pad_token, "src 和 tgt 的 pad_token 必须相同"
    if args.max_samples > 0 and len(ds) > args.max_samples:
        indices = random.sample(range(len(ds)), args.max_samples)
        ds.samples = [ds.samples[i] for i in indices]
        logger.info(f"使用 {len(ds)} 个样本")
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab),
    )
    save_dir = Path(args.save_dir)
    model = Seq2SeqTransformer(
        len(src_vocab),
        len(tgt_vocab),
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token],
        pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token],
    )
    model = model.to(DEVICE)
    
    # 🔴 新增: 记录训练开始详情
    log_training_start(
        logger,
        model,
        len(ds),
        args.batch_size,
        args.epochs,
        args.lr,
        DEVICE,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.token_to_id[tgt_vocab.pad_token])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # 🔧 [修复 #1] 使用新的检查点管理系统
    ckpt_mgr = TrainingCheckpointManager(save_dir, keep_checkpoints=3)
    
    # 保存训练配置（用于复现和调试）
    training_config = {
        'data_path': args.data,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'model_config': {
            'd_model': 256,
            'nhead': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
        },
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    ckpt_mgr.save_training_config(training_config)
    
    start_epoch = 1
    if args.resume:
        # 🔧 [修复 #2] 使用新的恢复接口（自动处理词表验证）
        try:
            start_epoch, ckpt_mgr = resume_or_init(
                save_dir,
                model,
                optimizer,
                src_vocab,
                tgt_vocab,
                device=DEVICE,
                allow_vocab_increase=False,
            )
            logger.info(f"✅ 从检查点恢复成功\n{ckpt_mgr.get_status_summary()}")
        except RuntimeError as e:
            logger.error(f"❌ 恢复失败: {e}")
            logger.warning("💡 建议: 删除旧检查点或不使用 --resume 参数重新开始")
            raise
    logger.info(f"开始训练 {args.epochs} 轮...")
    train_start_time = time.time()
    
    for ep in range(start_epoch, args.epochs + 1):
        avg_loss, avg_grad_norm, epoch_time = train_one_epoch(
            model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, DEVICE, ep
        )
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 调整学习率
        old_lr = current_lr
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_change = " (↓ LR)" if new_lr < old_lr else ""
        
        # 🔧 [修复 #1] 保存检查点到新系统，自动清理旧文件
        ckpt_mgr.save_checkpoint(
            epoch=ep,
            model=model,
            optimizer=optimizer,
            loss=avg_loss,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            metrics={
                'grad_norm': avg_grad_norm,
                'learning_rate': new_lr,
                'epoch_time': epoch_time,
            }
        )
        
        # 🔧 [修复 #1] 自动保存最优模型
        ckpt_mgr.save_best_model(
            epoch=ep,
            model=model,
            optimizer=optimizer,
            loss=avg_loss,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            metrics={
                'grad_norm': avg_grad_norm,
                'learning_rate': new_lr,
            }
        )
        
        # 🔧 [修复 #3] 记录训练历史
        ckpt_mgr.update_training_history(
            epoch=ep,
            loss=avg_loss,
            metrics={
                'grad_norm': avg_grad_norm,
                'learning_rate': new_lr,
                'epoch_time': epoch_time,
            }
        )
        
        # 保存词表（兼容旧代码）
        src_vocab.save(str(save_dir / 'src_vocab.json'))
        tgt_vocab.save(str(save_dir / 'tgt_vocab.json'))
        
        # 计算总进度
        total_time = time.time() - train_start_time
        avg_time_per_epoch = total_time / (ep - start_epoch + 1)
        remaining_epochs = args.epochs - ep
        eta_seconds = avg_time_per_epoch * remaining_epochs
        
        logger.info("-" * 70)
        logger.info(
            f"Epoch {ep:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Grad: {avg_grad_norm:.4f} | "
            f"LR: {new_lr:.6f}{lr_change} | "
            f"Time: {epoch_time:.1f}s | "
            f"ETA: {eta_seconds:.0f}s | "
            f"Mem: {get_memory_usage():.2f}GB"
        )
        
        # 显示最优模型状态
        if ckpt_mgr.best_loss != float('inf'):
            logger.info(
                f"最优: Loss={ckpt_mgr.best_loss:.4f} @ Epoch {ckpt_mgr.best_epoch} | "
                f"恶化: {avg_loss - ckpt_mgr.best_loss:+.4f}"
            )
        logger.info("-" * 70)
    
    total_train_time = time.time() - train_start_time
    logger.info(f"\n✅ 训练完成! 总耗时: {total_train_time:.1f}s")
    
    # 🔧 [修复 #3] 保存训练摘要
    ckpt_mgr.save_training_summary()
    logger.info(f"\n📊 训练总结:")
    logger.info(ckpt_mgr.get_status_summary())
    
    # 保存最终模型及完整元数据
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': 256,
            'nhead': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
        },
        'metadata': {
            'epoch': args.epochs,
            'loss': avg_loss,
            'best_loss': ckpt_mgr.best_loss,
            'best_epoch': ckpt_mgr.best_epoch,
            'timestamp': datetime.now().isoformat(),
            'device': str(DEVICE),
            'total_time': total_train_time,
            'data_source': args.data,
        },
        'vocab_info': {
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
        }
    }
    torch.save(final_model_data, save_dir / 'model.pt')
    logger.info(f"📦 模型已保存到 {save_dir}/model.pt (含元数据)")
    
    logger.info(f"\n📁 输出文件:")
    logger.info(f"   ✅ 最优模型: {save_dir}/best_model.pt")
    logger.info(f"   ✅ 最新检查点: {save_dir}/checkpoint_epoch{args.epochs}.pt")
    logger.info(f"   ✅ 训练日志: {ckpt_mgr.log_dir}/")
    logger.info(f"   ✅ 配置文件: {ckpt_mgr.log_dir}/config.json")
    logger.info(f"   ✅ 摘要: {ckpt_mgr.log_dir}/training_summary.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception("训练出错")
        raise
