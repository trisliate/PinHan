"""
SLM (Semantic Language Model) 训练脚本

训练语言模型用于候选重排序
"""

import os
import sys
import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import orjson

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slm.model import SLModel, SLMConfig, SLMVocab


class SLMDataset(Dataset):
    """SLM 训练数据集 - 语言模型训练"""
    
    def __init__(
        self, 
        data_path: str, 
        vocab: SLMVocab,
        max_len: int = 128,
        max_samples: int = None,
    ):
        self.vocab = vocab
        self.max_len = max_len
        
        self.samples = []
        
        print(f"加载数据: {data_path}")
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
        
        print(f"加载完成: {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> List[int]:
        hanzi = self.samples[idx]
        
        # 编码: BOS + text + EOS
        ids = [self.vocab.bos_id] + self.vocab.encode(hanzi) + [self.vocab.eos_id]
        
        return ids


def collate_fn(batch: List[List[int]]) -> torch.Tensor:
    """批处理函数：padding"""
    max_len = max(len(seq) for seq in batch)
    
    padded = []
    for seq in batch:
        padded.append(seq + [0] * (max_len - len(seq)))
    
    return torch.tensor(padded, dtype=torch.long)


def train_epoch(
    model: SLModel,
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
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device)
        
        # 语言模型: 输入是 x[:-1]，目标是 x[1:]
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        
        optimizer.zero_grad()
        
        logits = model(x)
        
        # 计算损失 (忽略 padding)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 统计非 padding token 数量
        non_pad = (y != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss / total_tokens
            ppl = torch.exp(torch.tensor(cur_loss)).item()
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {cur_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed:.1f}s")
    
    return total_loss / total_tokens


def evaluate(
    model: SLModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """评估模型，返回 (loss, perplexity)"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)
            
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]
            
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            
            non_pad = (y != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description='训练 SLM 模型')
    parser.add_argument('--data', type=str, default='data/train_data.jsonl')
    parser.add_argument('--freq', type=str, default='dicts/char_freq.json')
    parser.add_argument('--output', type=str, default='checkpoints/slm')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-every', type=int, default=5, help='每N轮保存一次检查点')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--amp', action='store_true', help='使用混合精度训练')
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, args.data)
    freq_path = os.path.join(project_root, args.freq)
    output_dir = os.path.join(project_root, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 日志目录
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'slm_train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 构建词表
    print("\n[Step 1] 构建词表...")
    vocab = SLMVocab()
    
    with open(freq_path, 'rb') as f:
        char_freq = orjson.loads(f.read())
    
    vocab.build_from_freq(char_freq, min_freq=0.0)
    print(f"  词表大小: {vocab.vocab_size}")
    
    # 保存词表
    vocab.save(os.path.join(output_dir, 'slm_vocab.json'))
    
    # 加载数据
    print("\n[Step 2] 加载数据...")
    dataset = SLMDataset(data_path, vocab, max_samples=args.max_samples)
    
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
    config = SLMConfig(vocab_size=vocab.vocab_size)
    model = SLModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    
    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    if scaler:
        print("  启用混合精度训练 (AMP)")
    
    # 训练
    print("\n[Step 4] 开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 验证
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | Time: {epoch_time:.1f}s")
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time,
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, os.path.join(output_dir, 'slm_best.pt'))
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f}, ppl={val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"  验证损失未改善 ({patience_counter}/{args.patience})")
        
        # 定期保存检查点
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, os.path.join(output_dir, f'slm_checkpoint_epoch{epoch}.pt'))
            print(f"  保存检查点 epoch {epoch}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping: 验证损失连续 {args.patience} 轮未改善")
            break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
    }, os.path.join(output_dir, 'slm_last.pt'))
    
    # 保存训练日志
    log_data = {
        'args': vars(args),
        'model_params': num_params,
        'train_size': train_size,
        'val_size': val_size,
        'best_val_loss': best_val_loss,
        'best_ppl': torch.exp(torch.tensor(best_val_loss)).item(),
        'history': train_history,
    }
    with open(log_file, 'wb') as f:
        f.write(orjson.dumps(log_data, option=orjson.OPT_INDENT_2))
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳困惑度: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    print(f"模型保存至: {output_dir}")
    print(f"训练日志: {log_file}")


if __name__ == '__main__':
    main()
