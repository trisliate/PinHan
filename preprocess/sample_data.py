#!/usr/bin/env python3
"""
preprocess/sample_data.py

从大型 JSONL 文件中提取样本数据（支持顺序和随机提取）。

使用示例：
    # 顺序提取前 7000 行
    python preprocess/sample_data.py --input data/wiki_latest.jsonl --output data/wiki_7k.jsonl --count 7000

    # 随机采样 10000 行
    python preprocess/sample_data.py --input data/wiki_latest.jsonl --output data/wiki_random_10k.jsonl --count 10000 --random

    # 从指定位置开始提取
    python preprocess/sample_data.py --input data/wiki_latest.jsonl --output data/wiki_start_100k.jsonl --count 50000 --start-line 100000
"""
import argparse
import sys
import random
from pathlib import Path
from typing import Optional


def count_lines(filepath: str) -> int:
    """快速统计文件行数（适用于大文件）。"""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count


def sample_sequential(input_path: str, output_path: str, count: int, start_line: int = 0) -> int:
    """
    顺序提取样本（从 start_line 开始，连续提取 count 行）。
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出文件路径
        count: 要提取的行数
        start_line: 起始行号（0-indexed）
    
    Returns:
        实际提取的行数
    """
    written = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i < start_line:
                continue
            if written >= count:
                break
            fout.write(line)
            written += 1
            
            # 定期输出进度
            if (written + 1) % 1000 == 0:
                print(f"已提取 {written:,} / {count:,} 行...", end='\r', flush=True)
    
    return written


def sample_random(input_path: str, output_path: str, count: int, seed: Optional[int] = None) -> int:
    """
    随机采样（适用于大文件，使用蓄水池采样算法）。
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出文件路径
        count: 要采样的行数
        seed: 随机种子（用于可重复性）
    
    Returns:
        实际采样的行数
    """
    if seed is not None:
        random.seed(seed)
    
    # 蓄水池采样算法 (Reservoir Sampling)
    reservoir = []
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for i, line in enumerate(fin):
            if i < count:
                reservoir.append(line)
            else:
                # 随机替换
                j = random.randint(0, i)
                if j < count:
                    reservoir[j] = line
            
            # 定期输出进度
            if (i + 1) % 10000 == 0:
                print(f"已处理 {i+1:,} 行...", end='\r', flush=True)
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for line in reservoir:
            fout.write(line)
    
    return len(reservoir)


def sample_stratified(input_path: str, output_path: str, count: int, ratio: float = 0.1) -> int:
    """
    分层采样（优先采样高频汉字）。
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出文件路径
        count: 要采样的行数（如果为 0，按比例采样）
        ratio: 采样比例（当 count=0 时使用）
    
    Returns:
        实际采样的行数
    """
    import json
    from collections import Counter
    
    # 第一遍：统计汉字频率
    print("第一遍扫描：统计汉字频率...")
    char_freq = Counter()
    total_lines = 0
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line.strip())
                hanzi = obj.get('hanzi', '')
                char_freq.update(hanzi)
                total_lines += 1
            except:
                continue
            
            if (i + 1) % 10000 == 0:
                print(f"已扫描 {i+1:,} 行...", end='\r', flush=True)
    
    # 计算采样数量
    actual_count = count if count > 0 else int(total_lines * ratio)
    
    print(f"\n第二遍扫描：采样高频汉字样本（目标 {actual_count:,} 行）...")
    
    # 第二遍：高频采样
    sampled = []
    random.seed(42)  # 固定种子保证可重复性
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for i, line in enumerate(fin):
            try:
                obj = json.loads(line.strip())
                hanzi = obj.get('hanzi', '')
                
                # 计算采样概率（高频字更容易被采样）
                char_score = sum(char_freq[c] for c in hanzi) / len(hanzi) if hanzi else 0
                avg_freq = sum(char_freq.values()) / len(char_freq)
                
                # 高频样本采样概率更高
                if random.random() < min(1.0, char_score / avg_freq * 0.5):
                    sampled.append(line)
                    if len(sampled) >= actual_count:
                        break
            except:
                continue
            
            if (i + 1) % 10000 == 0:
                print(f"已处理 {i+1:,} 行，已采样 {len(sampled):,} / {actual_count:,}...", end='\r', flush=True)
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for line in sampled:
            fout.write(line)
    
    return len(sampled)


def main():
    parser = argparse.ArgumentParser(
        description='从大型 JSONL 文件中提取样本数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  顺序提取前 7000 行:
    python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_7k.jsonl -c 7000
  
  随机采样 10000 行:
    python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_random_10k.jsonl -c 10000 --random
  
  分层采样（高频字优先）:
    python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_stratified.jsonl -c 10000 --stratified
  
  从第 100,000 行开始顺序提取 50,000 行:
    python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_50k.jsonl -c 50000 --start-line 100000
        '''
    )
    
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='输入 JSONL 文件路径'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help='输出文件路径'
    )
    parser.add_argument(
        '-c', '--count', 
        type=int, 
        default=7000, 
        help='要提取的行数（默认 7000）'
    )
    parser.add_argument(
        '--random', 
        action='store_true', 
        help='使用随机采样而非顺序提取'
    )
    parser.add_argument(
        '--stratified', 
        action='store_true', 
        help='使用分层采样（优先采样高频汉字）'
    )
    parser.add_argument(
        '--start-line', 
        type=int, 
        default=0, 
        help='顺序提取时的起始行号（默认 0）'
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not Path(args.input).exists():
        print(f"❌ 错误：输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    print(f"🎯 目标行数: {args.count:,}\n")
    
    try:
        if args.stratified:
            print("🔄 使用分层采样模式...\n")
            written = sample_stratified(args.input, args.output, args.count)
        elif args.random:
            print("🔄 使用随机采样模式...\n")
            written = sample_random(args.input, args.output, args.count)
        else:
            print("🔄 使用顺序提取模式...\n")
            written = sample_sequential(args.input, args.output, args.count, args.start_line)
        
        print(f"\n✅ 成功提取 {written:,} 行数据")
        print(f"📁 文件已保存: {args.output}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
