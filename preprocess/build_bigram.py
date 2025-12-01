#!/usr/bin/env python3
"""从语料构建 bigram 频率表。

使用:
    python preprocess/build_bigram.py -i data/corpus.jsonl -o dicts/bigram.json
    
或者从 CC-CEDICT 词语中提取:
    python preprocess/build_bigram.py --from-dict dicts/word_dict.json -o dicts/bigram.json
"""
import argparse
from pathlib import Path
from collections import Counter
import orjson


def build_from_corpus(corpus_path: Path, max_lines: int = None) -> Counter:
    """从语料库构建 bigram。"""
    bigram = Counter()
    
    print(f"从语料构建 bigram: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            
            try:
                obj = orjson.loads(line)
                text = obj.get('text', '') or obj.get('hanzi', '')
                
                # 统计相邻字对
                for j in range(len(text) - 1):
                    c1, c2 = text[j], text[j+1]
                    # 只统计汉字
                    if '\u4e00' <= c1 <= '\u9fff' and '\u4e00' <= c2 <= '\u9fff':
                        bigram[c1 + c2] += 1
            except Exception:
                continue
            
            if (i + 1) % 100000 == 0:
                print(f"  已处理 {i+1} 行, bigram 数量: {len(bigram)}")
    
    print(f"完成: {len(bigram)} 个 bigram")
    return bigram


def build_from_dict(dict_path: Path) -> Counter:
    """从词典构建 bigram。"""
    bigram = Counter()
    
    print(f"从词典构建 bigram: {dict_path}")
    
    with open(dict_path, 'rb') as f:
        word_dict = orjson.loads(f.read())
    
    for pinyin, words in word_dict.items():
        for word, freq in words:
            # 从每个词语提取 bigram
            for j in range(len(word) - 1):
                c1, c2 = word[j], word[j+1]
                if '\u4e00' <= c1 <= '\u9fff' and '\u4e00' <= c2 <= '\u9fff':
                    bigram[c1 + c2] += freq
    
    print(f"完成: {len(bigram)} 个 bigram")
    return bigram


def main():
    parser = argparse.ArgumentParser(description='构建 bigram 频率表')
    parser.add_argument('-i', '--input', help='语料文件 (JSONL)')
    parser.add_argument('--from-dict', help='从词典文件构建')
    parser.add_argument('-o', '--output', default='dicts/bigram.json', help='输出文件')
    parser.add_argument('--max', type=int, default=0, help='最大处理行数')
    parser.add_argument('--min-freq', type=int, default=2, help='最小频率阈值')
    args = parser.parse_args()
    
    if args.from_dict:
        bigram = build_from_dict(Path(args.from_dict))
    elif args.input:
        bigram = build_from_corpus(
            Path(args.input), 
            max_lines=args.max if args.max > 0 else None
        )
    else:
        # 默认从词典构建
        bigram = build_from_dict(Path('dicts/word_dict.json'))
    
    # 过滤低频
    if args.min_freq > 1:
        bigram = {k: v for k, v in bigram.items() if v >= args.min_freq}
        print(f"过滤后: {len(bigram)} 个 bigram (freq >= {args.min_freq})")
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(dict(bigram), option=orjson.OPT_INDENT_2))
    
    print(f"保存到: {output_path}")
    
    # 显示示例
    print("\n=== 高频 bigram 示例 ===")
    top = bigram.most_common(20) if isinstance(bigram, Counter) else sorted(bigram.items(), key=lambda x: -x[1])[:20]
    for bg, freq in top:
        print(f"  {bg}: {freq}")


if __name__ == '__main__':
    main()
