#!/usr/bin/env python3
"""
preprocess/analyze_data.py

分析 JSONL 数据文件的统计特征（声调分布、汉字频率、序列长度等）。

使用示例：
    python preprocess/analyze_data.py data/wiki_7k.jsonl
    python preprocess/analyze_data.py data/wiki_7k.jsonl --top 20
"""
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import argparse


def analyze_dataset(filepath: str, top_n: int = 10, show_samples: int = 5) -> Dict:
    """
    分析 JSONL 数据集的统计特征。
    
    Args:
        filepath: 数据文件路径
        top_n: 显示前 N 个频繁元素
        show_samples: 显示样本数量
    
    Returns:
        分析结果字典
    """
    if not Path(filepath).exists():
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)
    
    # 统计变量
    hanzi_freq = Counter()
    pinyin_freq = Counter()
    tone_stats = Counter()
    length_stats = []
    samples = []
    total_lines = 0
    valid_lines = 0
    punctuation_count = 0
    no_tone_count = 0
    mismatch_count = 0
    
    # 标点符号集合
    punctuation = '。！？；，：""''·…·（）【】《》、～、；'
    
    print(f"📂 分析文件: {filepath}\n")
    print("正在扫描数据...", end='', flush=True)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, 1):
            total_lines += 1
            
            try:
                obj = json.loads(line.strip())
                hanzi = obj.get('hanzi', '').strip()
                pinyin = obj.get('pinyin', '').strip()
                
                if not hanzi or not pinyin:
                    continue
                
                valid_lines += 1
                
                # 存储前几个样本
                if len(samples) < show_samples:
                    samples.append({'hanzi': hanzi, 'pinyin': pinyin})
                
                # 统计汉字
                for char in hanzi:
                    hanzi_freq[char] += 1
                
                # 统计拼音
                py_tokens = pinyin.split()
                pinyin_freq.update(py_tokens)
                length_stats.append(len(py_tokens))
                
                # 检查长度匹配
                if len(hanzi) != len(py_tokens):
                    mismatch_count += 1
                
                # 统计声调
                tone_found = False
                for py_token in py_tokens:
                    if py_token and py_token[-1].isdigit():
                        tone = py_token[-1]
                        tone_stats[f"声调{tone}"] += 1
                        tone_found = True
                    else:
                        no_tone_count += 1
                
                # 检测标点
                if any(p in hanzi for p in punctuation):
                    punctuation_count += 1
                    
            except Exception as e:
                continue
            
            if line_no % 1000 == 0:
                print(f"\r正在扫描数据... {line_no:,} 行", end='', flush=True)
    
    print(f"\r✅ 扫描完成                     \n")
    
    # 计算统计数据
    total_tokens = sum(length_stats) if length_stats else 0
    avg_length = sum(length_stats) / len(length_stats) if length_stats else 0
    
    # 输出结果
    print("=" * 80)
    print("📊 数据集统计信息")
    print("=" * 80)
    print(f"总行数: {total_lines:,}")
    print(f"有效样本: {valid_lines:,} ({100*valid_lines/total_lines:.1f}%)")
    print(f"总 token 数: {total_tokens:,}")
    print(f"不同汉字数: {len(hanzi_freq):,}")
    print(f"不同拼音数: {len(pinyin_freq):,}")
    
    print(f"\n📏 序列长度统计:")
    print(f"  平均长度: {avg_length:.2f}")
    print(f"  最短: {min(length_stats)} 最长: {max(length_stats)}")
    print(f"  长度匹配错误: {mismatch_count:,} ({100*mismatch_count/valid_lines:.1f}%)")
    
    print(f"\n📝 声调分布:")
    total_tones = sum(tone_stats.values())
    for tone in ['声调1', '声调2', '声调3', '声调4']:
        count = tone_stats.get(tone, 0)
        pct = 100 * count / total_tones if total_tones > 0 else 0
        bar_length = int(pct / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"  {tone}: {count:6,} ({pct:5.1f}%) {bar}")
    print(f"  无声调: {no_tone_count:6,} ({100*no_tone_count/total_tokens:.1f}% 的 token)")
    
    print(f"\n⚠️  包含标点的样本: {punctuation_count:,} ({100*punctuation_count/valid_lines:.1f}%)")
    
    print(f"\n🔤 Top {top_n} 常见汉字:")
    for i, (char, count) in enumerate(hanzi_freq.most_common(top_n), 1):
        print(f"  {i:2d}. '{char}': {count:6,}")
    
    print(f"\n🔤 Top {top_n} 常见拼音:")
    for i, (py, count) in enumerate(pinyin_freq.most_common(top_n), 1):
        print(f"  {i:2d}. {py:10s}: {count:6,}")
    
    print(f"\n📋 样本示例 (前 {len(samples)} 条):")
    print("-" * 80)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. hanzi: '{sample['hanzi']}'")
        print(f"   pinyin: '{sample['pinyin']}'")
        print()
    
    return {
        'total_lines': total_lines,
        'valid_lines': valid_lines,
        'total_tokens': total_tokens,
        'unique_hanzi': len(hanzi_freq),
        'unique_pinyin': len(pinyin_freq),
        'avg_length': avg_length,
        'tone_stats': dict(tone_stats),
    }


def main():
    parser = argparse.ArgumentParser(
        description='分析 JSONL 数据文件的统计特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  分析数据文件:
    python preprocess/analyze_data.py data/wiki_7k.jsonl
  
  显示前 20 个频繁元素:
    python preprocess/analyze_data.py data/wiki_7k.jsonl --top 20
  
  显示前 10 个样本:
    python preprocess/analyze_data.py data/wiki_7k.jsonl --samples 10
        '''
    )
    
    parser.add_argument('input', help='输入 JSONL 文件路径')
    parser.add_argument('--top', type=int, default=10, help='显示前 N 个频繁元素（默认 10）')
    parser.add_argument('--samples', type=int, default=5, help='显示样本数量（默认 5）')
    
    args = parser.parse_args()
    analyze_dataset(args.input, top_n=args.top, show_samples=args.samples)


if __name__ == '__main__':
    main()
