"""
SLM Lite - 轻量级语言模型训练脚本

目标: 2层 Transformer, ~500KB 参数, <20ms 推理
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import orjson

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slm.model import SLModel, SLMConfig, SLMVocab
from engine.logging import get_train_logger

# 初始化日志
logger = get_train_logger()


# ============ 模型配置 ============
# 升级版：4层 256维，约 3M 参数，平衡性能与速度
LITE_CONFIG = {
    'd_model': 256,      # 隐藏维度
    'n_heads': 4,        # 注意力头数
    'n_layers': 4,       # Transformer 层数
    'd_ff': 512,         # FFN 维度
    'max_len': 64,       # 最大序列长度
    'dropout': 0.1,
}


class SLMDataset(Dataset):
    """SLM 训练数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        vocab: SLMVocab,
        max_len: int = 64,
        max_samples: int = None,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []
        
        logger.info(f"加载数据: {data_path}")
        with open(data_path, 'rb') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                record = orjson.loads(line)
                hanzi = record['hanzi']
                
                # 长度过滤
                if len(hanzi) < 2 or len(hanzi) > max_len - 2:
                    continue
                
                self.samples.append(hanzi)
        
        logger.info(f"加载完成: {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> List[int]:
        hanzi = self.samples[idx]
        ids = [self.vocab.bos_id] + self.vocab.encode(hanzi) + [self.vocab.eos_id]
        return ids


def collate_fn(batch: List[List[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [0] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    epoch_start = time.time()
    
    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device)
        x, y = input_ids[:, :-1], input_ids[:, 1:]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        non_pad = (y != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        
        if (batch_idx + 1) % 100 == 0:
            cur_loss = total_loss / total_tokens
            cur_ppl = torch.exp(torch.tensor(cur_loss)).item()
            elapsed = time.time() - epoch_start
            speed = total_tokens / elapsed
            logger.debug(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {cur_loss:.4f} | PPL: {cur_ppl:.2f} | "
                f"Speed: {speed:.0f} tok/s"
            )
    
    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)
            x, y = input_ids[:, :-1], input_ids[:, 1:]
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            non_pad = (y != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description='训练 SLM Lite 模型')
    parser.add_argument('--data', type=str, default='data/training/train_data.jsonl')
    parser.add_argument('--freq', type=str, default='data/dicts/char_freq.json')
    parser.add_argument('--output', type=str, default='model/slm_lite')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-samples', type=int, default=500000, help='最大样本数')
    parser.add_argument('--vocab-size', type=int, default=8000, help='词表大小')
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 训练开始日志
    logger.info("=" * 60)
    logger.info("SLM Lite 训练开始")
    logger.info(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"  数据: {args.data}, max_samples={args.max_samples}")
    logger.info("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, args.data)
    freq_path = os.path.join(project_root, args.freq)
    output_dir = os.path.join(project_root, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ============ 构建词表 ============
    logger.info("\n[1/4] 构建词表...")
    vocab = SLMVocab()
    with open(freq_path, 'rb') as f:
        char_freq = orjson.loads(f.read())
    
    # 只保留 top N 常用字
    vocab.build_from_freq(char_freq, max_vocab=args.vocab_size, min_freq=0.0)
    logger.info(f"  词表大小: {vocab.vocab_size}")
    vocab.save(os.path.join(output_dir, 'vocab.json'))
    
    # ============ 加载数据 ============
    logger.info("\n[2/4] 加载数据...")
    dataset = SLMDataset(data_path, vocab, max_len=LITE_CONFIG['max_len'], max_samples=args.max_samples)
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    logger.info(f"  训练集: {train_size}, 验证集: {val_size}")
    
    # ============ 创建模型 ============
    logger.info("\n[3/4] 创建模型...")
    config = SLMConfig(
        vocab_size=vocab.vocab_size,
        d_model=LITE_CONFIG['d_model'],
        n_heads=LITE_CONFIG['n_heads'],
        n_layers=LITE_CONFIG['n_layers'],
        d_ff=LITE_CONFIG['d_ff'],
        max_len=LITE_CONFIG['max_len'],
        dropout=LITE_CONFIG['dropout'],
    )
    model = SLModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / 1024 / 1024  # float32
    logger.info(f"  参数量: {num_params:,} ({model_size_mb:.2f} MB)")
    logger.info(f"  模型结构: {LITE_CONFIG['n_layers']}层, {LITE_CONFIG['d_model']}维, {LITE_CONFIG['n_heads']}头")
    logger.info(f"  量化后约: {model_size_mb / 4:.2f} MB (INT8)")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # ============ 训练 ============
    logger.info("\n[4/4] 开始训练...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - epoch_start
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        lr = optimizer.param_groups[0]['lr']
        
        # 详细 epoch 日志
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, os.path.join(output_dir, 'best.pt'))
            logger.info(f"  ✓ 保存最佳模型 (Val PPL={val_ppl:.2f})")
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(output_dir, 'last.pt'))
    
    total_time = time.time() - training_start
    best_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info(f"  最佳 Val PPL: {best_ppl:.2f}")
    logger.info(f"  总训练时间: {total_time/60:.1f} 分钟")
    logger.info(f"  模型保存至: {output_dir}")
    logger.info("=" * 60)
    
    # ============ 测试推理速度 ============
    logger.info("\n测试推理速度...")
    model.eval()
    test_input = torch.randint(4, vocab.vocab_size, (1, 10)).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(test_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) / 100 * 1000
    
    logger.info(f"  单次推理延迟: {elapsed:.2f} ms")


if __name__ == '__main__':
    main()
