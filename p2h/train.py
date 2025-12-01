"""
P2H 模型训练脚本

使用从维基百科提取的训练数据训练 P2H 模型
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

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from p2h.model import P2HModel, P2HConfig, P2HVocab


class P2HDataset(Dataset):
    """P2H 训练数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        vocab: P2HVocab,
        max_pinyin_len: int = 64,
        max_hanzi_len: int = 64,
        max_samples: int = None,
    ):
        self.vocab = vocab
        self.max_pinyin_len = max_pinyin_len
        self.max_hanzi_len = max_hanzi_len
        
        self.samples = []
        
        print(f"加载数据: {data_path}")
        with open(data_path, 'rb') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                record = orjson.loads(line)
                
                # 拼音列表
                pinyins = record['pinyin'].split()
                hanzi = record['hanzi']
                
                # 长度过滤
                if len(pinyins) > max_pinyin_len - 2 or len(hanzi) > max_hanzi_len - 2:
                    continue
                
                self.samples.append((pinyins, hanzi))
        
        print(f"加载完成: {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        pinyins, hanzi = self.samples[idx]
        
        # 编码
        pinyin_ids = self.vocab.encode_pinyin(pinyins)
        hanzi_ids = [1] + self.vocab.encode_hanzi(hanzi) + [2]  # BOS + text + EOS
        
        return pinyin_ids, hanzi_ids


def collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """批处理函数：padding"""
    pinyin_batch, hanzi_batch = zip(*batch)
    
    # 计算最大长度
    max_pinyin = max(len(p) for p in pinyin_batch)
    max_hanzi = max(len(h) for h in hanzi_batch)
    
    # Padding
    pinyin_padded = []
    hanzi_padded = []
    
    for p, h in zip(pinyin_batch, hanzi_batch):
        pinyin_padded.append(p + [0] * (max_pinyin - len(p)))
        hanzi_padded.append(h + [0] * (max_hanzi - len(h)))
    
    return (
        torch.tensor(pinyin_padded, dtype=torch.long),
        torch.tensor(hanzi_padded, dtype=torch.long),
    )


def train_epoch(
    model: P2HModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (pinyin_ids, hanzi_ids) in enumerate(dataloader):
        pinyin_ids = pinyin_ids.to(device)
        hanzi_ids = hanzi_ids.to(device)
        
        # 输入是 hanzi_ids[:-1]，目标是 hanzi_ids[1:]
        tgt_input = hanzi_ids[:, :-1]
        tgt_output = hanzi_ids[:, 1:]
        
        optimizer.zero_grad()
        
        logits = model(pinyin_ids, tgt_input)
        
        # 计算损失
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {cur_loss:.4f} | Time: {elapsed:.1f}s")
    
    return total_loss / len(dataloader)


def evaluate(
    model: P2HModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for pinyin_ids, hanzi_ids in dataloader:
            pinyin_ids = pinyin_ids.to(device)
            hanzi_ids = hanzi_ids.to(device)
            
            tgt_input = hanzi_ids[:, :-1]
            tgt_output = hanzi_ids[:, 1:]
            
            logits = model(pinyin_ids, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='训练 P2H 模型')
    parser.add_argument('--data', type=str, default='data/train_data.jsonl')
    parser.add_argument('--dict', type=str, default='dicts/char_dict.json')
    parser.add_argument('--freq', type=str, default='dicts/char_freq.json')
    parser.add_argument('--output', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, args.data)
    dict_path = os.path.join(project_root, args.dict)
    freq_path = os.path.join(project_root, args.freq)
    output_dir = os.path.join(project_root, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 构建词表
    print("\n[Step 1] 构建词表...")
    vocab = P2HVocab()
    
    with open(dict_path, 'rb') as f:
        char_dict = orjson.loads(f.read())
    
    with open(freq_path, 'rb') as f:
        char_freq = orjson.loads(f.read())
    
    vocab.build_from_dict(char_dict, char_freq)
    print(f"  拼音词表: {vocab.pinyin_vocab_size}")
    print(f"  汉字词表: {vocab.hanzi_vocab_size}")
    
    # 保存词表
    vocab.save(os.path.join(output_dir, 'vocab.json'))
    
    # 加载数据
    print("\n[Step 2] 加载数据...")
    dataset = P2HDataset(data_path, vocab, max_samples=args.max_samples)
    
    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"  训练集: {train_size}")
    print(f"  验证集: {val_size}")
    
    # 创建模型
    print("\n[Step 3] 创建模型...")
    config = P2HConfig(
        pinyin_vocab_size=vocab.pinyin_vocab_size,
        hanzi_vocab_size=vocab.hanzi_vocab_size,
    )
    model = P2HModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    
    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    
    # 训练
    print("\n[Step 4] 开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  保存最佳模型 (val_loss={val_loss:.4f})")
        
        # 每个 epoch 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'val_loss': val_loss,
        }, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
