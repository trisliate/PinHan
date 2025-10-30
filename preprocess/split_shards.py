"""preprocess/split_shards.py

将一个大的 JSONL 文件按行数分割成多个 shard 文件，便于流式训练。

示例:
  python preprocess/split_shards.py -i data/splits/train.jsonl -o data/train_shards -l 10000

每个 shard 文件命名为 shard_00001.jsonl、shard_00002.jsonl ...
"""
import argparse
import os

def split_file(input_path, out_dir, lines_per_shard=200000):
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    out = None
    out_path = None
    written = 0
    with open(input_path, 'rb') as f:
        for i, line in enumerate(f, start=1):
            if (i-1) % lines_per_shard == 0:
                if out:
                    out.close()
                shard_idx += 1
                out_path = os.path.join(out_dir, f'shard_{shard_idx:05d}.jsonl')
                out = open(out_path, 'wb')
                written = 0
            out.write(line)
            written += 1
    if out:
        out.close()
    print(f'Split done, created {shard_idx} shards in {out_dir}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True, help='输入 JSONL 文件路径')
    p.add_argument('-o', '--outdir', required=True, help='输出 shards 目录')
    p.add_argument('-l', '--lines-per-shard', type=int, default=200000, help='每个 shard 的行数')
    args = p.parse_args()
    split_file(args.input, args.outdir, args.lines_per_shard)


if __name__ == '__main__':
    main()
