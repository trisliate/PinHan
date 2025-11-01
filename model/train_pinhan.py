"""训练脚本：拼音到汉字的 Transformer 模型训练."""
import random
import sys
import logging
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

DATA_PATH = Path('data/clean_wiki.jsonl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
) -> float:
    """训练一个 epoch."""
    logger = logging.getLogger()
    model.train()
    total_loss = 0.0
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_key_padding_mask = (src.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
        tgt_key_padding_mask = (tgt_input.transpose(0, 1) == tgt_vocab.token_to_id[tgt_vocab.pad_token])
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
        if i % 50 == 0:
            logger.info(f"Step {i}, loss: {loss.item():.4f}")
    return total_loss / len(dataloader)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, out_dir: Path) -> None:
    """保存 checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(ckpt, out_dir / f'checkpoint_epoch{epoch}.pt')


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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.token_to_id[tgt_vocab.pad_token])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    start_epoch = 1
    if args.resume:
        ckpts = sorted(save_dir.glob('checkpoint_epoch*.pt'))
        if ckpts:
            latest = ckpts[-1]
            logger.info(f"从 {latest} 恢复")
            ck = torch.load(str(latest), map_location=DEVICE)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck.get('epoch', 1) + 1
    logger.info(f"开始训练 {args.epochs} 轮...")
    for ep in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, DEVICE)
        logger.info(f"Epoch {ep}: 平均 loss {avg_loss:.4f}")
        scheduler.step(avg_loss)
        save_checkpoint(model, optimizer, ep, save_dir)
        src_vocab.save(str(save_dir / 'src_vocab.json'))
        tgt_vocab.save(str(save_dir / 'tgt_vocab.json'))
    torch.save({'model_state_dict': model.state_dict()}, save_dir / 'model.pt')
    logger.info(f"训练完成，模型保存到 {save_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception("训练出错")
        raise
