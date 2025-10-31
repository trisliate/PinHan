"""
训练脚本：用于将拼音翻译为汉字的轻量级 Transformer 训练示例。
- 从 `data/tran.jsonl` 构建词表（拼音按空格分词，汉字按字符）
- 构造 Dataset 与 DataLoader，并执行一次小规模训练作为 smoke test
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask


DATA_PATH = Path('data/tran.jsonl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PinyinHanziDataset(Dataset):
    """简单的数据集：每条样本为 (拼音 tokens 列表, 汉字字符列表)。

    拼音按空格切分，汉字按字符切分。会在构造时进行截断以限制长度。
    """
    def __init__(self, path: Path, src_vocab: Vocab, tgt_vocab: Vocab, max_src_len=64, max_tgt_len=64):
        self.samples = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                hanzi = j.get('hanzi')
                pinyin = j.get('pinyin')
                if not hanzi or not pinyin:
                    continue
                src_tokens = pinyin.strip().split()
                tgt_tokens = list(hanzi.strip())
                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    continue
                # 截断以适配最大长度（保留 BOS/EOS 空间）
                if len(src_tokens) > (max_src_len - 2):
                    src_tokens = src_tokens[:max_src_len - 2]
                if len(tgt_tokens) > (max_tgt_len - 2):
                    tgt_tokens = tgt_tokens[:max_tgt_len - 2]
                self.samples.append((src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, src_vocab: Vocab, tgt_vocab: Vocab):
    """将 batch 转换为模型所需的张量格式，并进行 padding。

    返回：src_tensor (S, N), tgt_tensor (T, N)
    """
    srcs, tgts = zip(*batch)
    src_ids = [src_vocab.encode(s, add_bos_eos=True) for s in srcs]
    tgt_ids = [tgt_vocab.encode(t, add_bos_eos=True) for t in tgts]
    max_src = max(len(x) for x in src_ids)
    max_tgt = max(len(x) for x in tgt_ids)
    pad_id_src = src_vocab.token_to_id[src_vocab.pad_token]
    pad_id_tgt = tgt_vocab.token_to_id[tgt_vocab.pad_token]
    src_padded = [x + [pad_id_src] * (max_src - len(x)) for x in src_ids]
    tgt_padded = [x + [pad_id_tgt] * (max_tgt - len(x)) for x in tgt_ids]
    src_tensor = torch.LongTensor(src_padded).transpose(0,1)  # (S, N)
    tgt_tensor = torch.LongTensor(tgt_padded).transpose(0,1)  # (T, N)
    return src_tensor, tgt_tensor


def build_vocabs(path: Path, max_pinyin_tokens=50000, max_hanzi_chars=50000):
    """从 jsonl 构建拼音 token 词表和汉字字符词表，按频次截断。"""
    pinyin_counter = {}
    hanzi_counter = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                j=json.loads(line)
            except Exception:
                continue
            p=j.get('pinyin')
            h=j.get('hanzi')
            if not p or not h:
                continue
            for t in p.strip().split():
                pinyin_counter[t]=pinyin_counter.get(t,0)+1
            for ch in h.strip():
                hanzi_counter[ch]=hanzi_counter.get(ch,0)+1
    pinyin_sorted = sorted(pinyin_counter.items(), key=lambda x: -x[1])
    hanzi_sorted = sorted(hanzi_counter.items(), key=lambda x: -x[1])
    pinyin_tokens = [t for t,_ in pinyin_sorted][:max_pinyin_tokens]
    hanzi_tokens = [t for t,_ in hanzi_sorted][:max_hanzi_chars]
    src_vocab = Vocab(pinyin_tokens)
    tgt_vocab = Vocab(hanzi_tokens)
    return src_vocab, tgt_vocab


def train_one_epoch(model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, device):
    """训练一个 epoch（用于 smoke test）。"""
    model.train()
    total_loss = 0.0
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        src_mask = None
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_key_padding_mask = (src.transpose(0,1) == src_vocab.token_to_id[src_vocab.pad_token])
        tgt_key_padding_mask = (tgt_input.transpose(0,1) == tgt_vocab.token_to_id[tgt_vocab.pad_token])
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                       src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=src_key_padding_mask)
        # logits: (T, N, V)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            # 使用 logging 输出，便于同时写入文件和控制台
            import logging
            logging.info(f"Step {i} loss {loss.item():.4f}")
    return total_loss / len(dataloader)


def save_model(model, src_vocab, tgt_vocab, out_dir: Path):
    """保存模型权重和词表到指定目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, out_dir / 'model.pt')
    src_vocab.save(str(out_dir / 'src_vocab.json'))
    tgt_vocab.save(str(out_dir / 'tgt_vocab.json'))


import argparse
import logging
from datetime import datetime


def setup_logging(log_file: str = None):
    # 配置 logging，既打印到控制台也写入文件（若提供）
    fmt = '%(asctime)s %(levelname)s: %(message)s'
    # 使用 force=True 确保即使之前有 handlers 也会重设（Python3.8+）
    logging.basicConfig(level=logging.INFO, format=fmt, force=True)
    logger = logging.getLogger()
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


def save_checkpoint(model, optimizer, epoch: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(ckpt, out_dir / f'checkpoint_epoch{epoch}.pt')


def main():
    # 在控制台打印一条运行确认信息，确保从命令行运行时可以看到即时反馈
    print('开始运行训练脚本 main()，将输出日志到指定的 log 文件（若提供）')
    parser = argparse.ArgumentParser(description='训练轻量级拼音->汉字 Transformer')
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save-dir', type=str, default='outputs/small_model', help='模型与词表保存目录')
    parser.add_argument('--max-samples', type=int, default=2000, help='用于 smoke 测试的最大样本数（0 表示使用全部）')
    parser.add_argument('--resume', action='store_true', help='如果存在 checkpoint 则从最近 checkpoint 恢复')
    parser.add_argument('--log-file', type=str, default=None, help='若提供则把日志写入指定文件')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_log = f'outputs/train_{timestamp}.log'
    log_file = args.log_file or default_log
    logger = setup_logging(log_file)

    logger.info('开始构建词表...')
    src_vocab, tgt_vocab = build_vocabs(Path(args.data))
    logger.info(f'源词表大小: {len(src_vocab)}  目标词表大小: {len(tgt_vocab)}')

    ds = PinyinHanziDataset(Path(args.data), src_vocab, tgt_vocab, max_src_len=64, max_tgt_len=64)
    if args.max_samples and args.max_samples > 0 and len(ds) > args.max_samples:
        indices = random.sample(range(len(ds)), args.max_samples)
        small_samples = [ds.samples[i] for i in indices]
        ds.samples = small_samples
        logger.info(f'使用子集训练，共 {len(ds)} 条样本')
    else:
        logger.info(f'使用全部样本，共 {len(ds)} 条')

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))

    save_dir = Path(args.save_dir)
    # 指定源/目标的 padding id，保证 embedding 的 padding 行为一致
    model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab), d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token], pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token])
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.token_to_id[tgt_vocab.pad_token])

    start_epoch = 1
    if args.resume:
        # 尝试找到最新 checkpoint
        ckpts = sorted(save_dir.glob('checkpoint_epoch*.pt'))
        if ckpts:
            latest = ckpts[-1]
            logger.info(f'找到 checkpoint，正在从 {latest} 恢复')
            ck = torch.load(str(latest), map_location=DEVICE)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck.get('epoch', 1) + 1
            logger.info(f'从 epoch {start_epoch} 开始训练')

    logger.info(f'开始训练（{args.epochs} 轮）')
    for ep in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, DEVICE)
        logger.info(f'Epoch {ep} 完成  平均 loss {avg_loss:.4f}')
        # 保存 checkpoint 与词表
        save_checkpoint(model, optimizer, ep, save_dir)
        src_vocab.save(str(save_dir / 'src_vocab.json'))
        tgt_vocab.save(str(save_dir / 'tgt_vocab.json'))
        logger.info(f'已保存 checkpoint 与词表到 {save_dir}')

    # 最终保存模型权重（兼容旧接口）
    torch.save({'model_state_dict': model.state_dict()}, save_dir / 'model.pt')
    logger.info(f'训练完成，最终模型保存在 {save_dir / "model.pt"}')


# 在脚本作为主程序运行时执行主逻辑，并增加异常捕获以便把错误同时打印到控制台和日志文件
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 如果发生未捕获的异常，打印堆栈信息并重新抛出，便于调试
        import logging, traceback, sys
        logging.basicConfig(level=logging.INFO)
        logging.exception('训练脚本运行时发生未处理的异常')
        print('训练脚本发生异常，详情见上方堆栈信息', file=sys.stderr)
        traceback.print_exc()
        raise
