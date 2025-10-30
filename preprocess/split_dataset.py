"""
preprocess/split_dataset.py

流式将大型 JSONL 文件按 `hanzi` 字段确定性划分为 train/val/test。
使用 sha1(hash) 对 `hanzi` 做散列，保证分割可重复且跨运行一致。

使用示例：
    python preprocess/split_dataset.py \
        --input data/clean_data.jsonl \
        --outdir data/splits \
        --train 0.90 --val 0.05 --test 0.05

该脚本会在 outdir 下写入 `train.jsonl`, `val.jsonl`, `test.jsonl`。
"""
import argparse
import os
import hashlib
import orjson


def choose_split(hanzi: str, train_ratio: float, val_ratio: float) -> str:
    # 基于 hanzi 的 sha1 值做确定性分割
    h = hashlib.sha1(hanzi.encode('utf-8')).digest()
    v = int.from_bytes(h[:8], 'big') % 10000
    train_thresh = int(train_ratio * 10000)
    val_thresh = train_thresh + int(val_ratio * 10000)
    if v < train_thresh:
        return 'train'
    if v < val_thresh:
        return 'val'
    return 'test'


def split_file(inpath, outdir, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, key='hanzi'):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    os.makedirs(outdir, exist_ok=True)
    out_paths = {
        'train': os.path.join(outdir, 'train.jsonl'),
        'val': os.path.join(outdir, 'val.jsonl'),
        'test': os.path.join(outdir, 'test.jsonl'),
    }
    outs = {k: open(p, 'wb') for k, p in out_paths.items()}
    counts = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    total = 0
    # 流式读取输入文件逐行处理（内存友好）
    with open(inpath, 'rb') as fin:
        for raw in fin:
            total += 1
            line = raw.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
                key_val = obj.get(key)
                if not key_val:
                    # 如果该条记录没有 hanzi 字段则跳过
                    counts['skipped'] += 1
                    continue
            except Exception:
                counts['skipped'] += 1
                continue
            split = choose_split(key_val, train_ratio, val_ratio)
            outs[split].write(line + b"\n")
            counts[split] += 1
            if total % 1000000 == 0:
                print(f'已处理 {total:,} 行 — train={counts["train"]:,} val={counts["val"]:,} test={counts["test"]:,} 跳过={counts["skipped"]:,}')
    for f in outs.values():
        f.close()
    print('Done. Total processed:', total)
    print(counts)
    return out_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--outdir', '-o', default='data/splits')
    p.add_argument('--train', type=float, default=0.90)
    p.add_argument('--val', type=float, default=0.05)
    p.add_argument('--test', type=float, default=0.05)
    p.add_argument('--key', default='hanzi', help='用于哈希分割的 JSON 字段，默认 hanzi')
    args = p.parse_args()
    split_file(args.input, args.outdir, args.train, args.val, args.test, args.key)


if __name__ == '__main__':
    main()
