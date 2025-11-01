#!/usr/bin/env python3
"""
小规模训练脚本：从10k.jsonl中随机抽取500-3000行进行快速训练测试。
用于评估模型效果和预估生产训练的轮数需求。
"""
import random
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import orjson

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / 'model'))
sys.path.insert(0, str(Path(__file__).parent / 'preprocess'))

from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
from pinyin_utils import validate_pinyin_sequence
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time


class SmallPinyinDataset(Dataset):
    """小规模拼音->汉字数据集."""
    def __init__(self, samples: list, src_vocab: Vocab, tgt_vocab: Vocab):
        self.samples = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, src_vocab, tgt_vocab):
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


def extract_small_dataset(input_file: Path, sample_size: int = 1000) -> list:
    """
    从10k.jsonl中随机抽取指定数量的样本。
    返回 [(src_tokens, tgt_tokens), ...] 列表
    """
    logger.info(f"从 {input_file} 读取所有数据...")
    all_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = orjson.loads(line)
            except:
                continue
            
            pinyin = j.get('pinyin', '').strip()
            hanzi = j.get('hanzi', '').strip()
            
            if not pinyin or not hanzi:
                continue
            
            # 验证拼音
            if not validate_pinyin_sequence(pinyin):
                continue
            
            src_tokens = pinyin.split()
            tgt_tokens = list(hanzi)
            
            # 长度匹配检查
            if len(src_tokens) != len(tgt_tokens):
                continue
            
            all_samples.append((src_tokens, tgt_tokens))
    
    logger.info(f"总共找到 {len(all_samples)} 个有效样本")
    
    # 随机抽取
    if len(all_samples) > sample_size:
        selected = random.sample(all_samples, sample_size)
    else:
        selected = all_samples
    
    logger.info(f"抽取了 {len(selected)} 个样本用于训练")
    return selected


def build_vocabs_from_samples(samples: list) -> tuple:
    """从样本中构建词表."""
    pinyin_set = set()
    hanzi_set = set()
    
    for src_tokens, tgt_tokens in samples:
        pinyin_set.update(src_tokens)
        hanzi_set.update(tgt_tokens)
    
    src_vocab = Vocab(list(pinyin_set))
    tgt_vocab = Vocab(list(hanzi_set))
    
    logger.info(f"源词表大小: {len(src_vocab)}")
    logger.info(f"目标词表大小: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab


def train_small_model(
    samples: list,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    output_dir: Path = None
) -> dict:
    """
    小规模模型训练。
    返回 {epoch -> {loss, metrics}}
    """
    if output_dir is None:
        output_dir = Path('outputs/small_train')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {DEVICE}")
    
    # 构建词表
    src_vocab, tgt_vocab = build_vocabs_from_samples(samples)
    
    # 创建数据集和dataloader
    dataset = SmallPinyinDataset(samples, src_vocab, tgt_vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab),
    )
    
    # 创建模型
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
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.token_to_id[tgt_vocab.pad_token]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    logger.info(f"开始训练 {epochs} 轮，样本数: {len(samples)}")
    logger.info(f"批大小: {batch_size}，学习率: {learning_rate}")
    
    history = {}
    train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)
            src_key_padding_mask = (
                src.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token]
            )
            tgt_key_padding_mask = (
                tgt_input.transpose(0, 1) == tgt_vocab.token_to_id[tgt_vocab.pad_token]
            )
            
            # 转换mask为float
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % max(1, len(dataloader) // 5) == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} [{batch_idx+1}/{len(dataloader)}] "
                    f"loss={loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        epoch_time = time.time() - epoch_start
        
        history[epoch] = {
            'loss': avg_loss,
            'time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        logger.info(
            f"Epoch {epoch} 完成 | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
    
    total_time = time.time() - train_start
    logger.info(f"✅ 训练完成! 总耗时: {total_time:.1f}s")
    
    # 保存模型和词表
    torch.save(model.state_dict(), output_dir / 'model.pt')
    src_vocab.save(str(output_dir / 'src_vocab.json'))
    tgt_vocab.save(str(output_dir / 'tgt_vocab.json'))
    
    logger.info(f"模型已保存到 {output_dir}")
    
    return history


def analyze_training_curve(history: dict) -> dict:
    """
    分析训练曲线，预估达到目标精度需要的轮数。
    
    返回 {
        'initial_loss': float,
        'final_loss': float,
        'loss_reduction_percent': float,
        'avg_loss_per_epoch': float,
        'estimated_epochs_for_99_percent': int,
    }
    """
    losses = [h['loss'] for h in history.values()]
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction_percent = (initial_loss - final_loss) / initial_loss * 100
    avg_loss_per_epoch = sum(losses) / len(losses)
    
    # 简单的线性外推
    if final_loss < initial_loss:
        loss_per_epoch = (initial_loss - final_loss) / len(losses)
        # 目标：损失降低到初始损失的5%
        target_loss = initial_loss * 0.05
        remaining_epochs = max(0, int((final_loss - target_loss) / loss_per_epoch))
        estimated_epochs = len(losses) + remaining_epochs
    else:
        estimated_epochs = len(losses) * 2  # 保守估计翻倍
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction_percent': reduction_percent,
        'avg_loss_per_epoch': avg_loss_per_epoch,
        'estimated_epochs_for_convergence': estimated_epochs,
        'total_epochs_trained': len(losses),
    }


def main():
    """主函数."""
    import argparse
    
    parser = argparse.ArgumentParser(description='小规模训练脚本')
    parser.add_argument('--input', type=str, default='data/10k.jsonl', help='输入数据文件')
    parser.add_argument('--sample-size', type=int, default=1000, help='抽取样本数 (500-3000)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--output-dir', type=str, default='outputs/small_train', help='输出目录')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 确保样本大小在合理范围
    sample_size = max(500, min(3000, args.sample_size))
    logger.info(f"目标样本数: {sample_size}")
    
    # 提取数据
    samples = extract_small_dataset(input_file, sample_size)
    
    if len(samples) < 10:
        logger.error("样本太少，无法训练")
        sys.exit(1)
    
    # 训练模型
    history = train_small_model(
        samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir)
    )
    
    # 分析训练曲线
    analysis = analyze_training_curve(history)
    
    logger.info("\n" + "="*60)
    logger.info("📊 训练分析结果")
    logger.info("="*60)
    logger.info(f"初始损失: {analysis['initial_loss']:.4f}")
    logger.info(f"最终损失: {analysis['final_loss']:.4f}")
    logger.info(f"损失下降: {analysis['loss_reduction_percent']:.2f}%")
    logger.info(f"平均每轮损失: {analysis['avg_loss_per_epoch']:.4f}")
    logger.info(f"实际训练轮数: {analysis['total_epochs_trained']}")
    logger.info(f"预估收敛轮数: {analysis['estimated_epochs_for_convergence']}")
    logger.info("="*60)
    
    # 保存分析结果
    output_dir = Path(args.output_dir)
    with open(output_dir / 'training_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': analysis,
            'history': history,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'sample_size': len(samples),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n详细分析已保存到 {output_dir}/training_analysis.json")


if __name__ == '__main__':
    main()
