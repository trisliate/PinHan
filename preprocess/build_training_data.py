"""
训练数据构建主脚本（改进版）

从维基百科 XML 提取数据并生成训练集
- 实时进度显示
- 布隆过滤器去重
- 质量过滤
"""

import os
import sys
import re
import argparse
import hashlib
import time
from datetime import datetime
from typing import Iterator

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import orjson
from pypinyin import lazy_pinyin, Style
import jieba
from opencc import OpenCC


# 简单的布隆过滤器实现
class BloomFilter:
    """简单布隆过滤器，用于句子去重"""
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.01):
        # 计算位数组大小
        import math
        self.size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.size / capacity * math.log(2))
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0
    
    def _hashes(self, item: str) -> Iterator[int]:
        """生成多个哈希值"""
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size
    
    def add(self, item: str):
        """添加元素"""
        for pos in self._hashes(item):
            self.bit_array[pos // 8] |= (1 << (pos % 8))
        self.count += 1
    
    def __contains__(self, item: str) -> bool:
        """检查元素是否存在"""
        for pos in self._hashes(item):
            if not (self.bit_array[pos // 8] & (1 << (pos % 8))):
                return False
        return True


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int = None, desc: str = "处理中"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.last_print = 0
    
    def update(self, n: int = 1):
        self.current += n
        now = time.time()
        # 每秒更新一次或每 1000 条更新
        if now - self.last_print >= 1.0 or self.current % 1000 == 0:
            self._print_progress()
            self.last_print = now
    
    def _print_progress(self):
        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        
        if self.total:
            pct = self.current / self.total * 100
            eta = (self.total - self.current) / speed if speed > 0 else 0
            print(f"\r  {self.desc}: {self.current}/{self.total} ({pct:.1f}%) | "
                  f"速度: {speed:.0f}/s | 剩余: {eta:.0f}s", end="", flush=True)
        else:
            print(f"\r  {self.desc}: {self.current} | 速度: {speed:.0f}/s | "
                  f"耗时: {elapsed:.0f}s", end="", flush=True)
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n  完成: {self.current} 条, 耗时 {elapsed:.1f}s")


def hanzi_to_pinyin(text: str) -> list:
    """汉字转拼音"""
    pinyins = lazy_pinyin(text, style=Style.NORMAL)
    return [p.replace('ü', 'v') for p in pinyins]


def is_valid_sentence(sent: str) -> bool:
    """检查句子是否有效（在清洗前调用）"""
    if len(sent) < 4 or len(sent) > 100:
        return False
    
    # 中文字符数量
    chinese = len(re.findall(r'[\u4e00-\u9fff]', sent))
    if chinese < 4:  # 至少4个中文字符
        return False
    
    # 中文占比（基于原始长度）
    if chinese / len(sent) < 0.5:
        return False
    
    return True


def clean_sentence(sent: str) -> str:
    """清洗句子，只保留中文"""
    sent = re.sub(r'[^\u4e00-\u9fff]', '', sent)
    return sent


def extract_sentences_from_text(text: str) -> Iterator[str]:
    """从文本提取句子"""
    # 按标点分割
    sentences = re.split(r'[。！？；\n\r]', text)
    
    for sent in sentences:
        sent = sent.strip()
        # 先验证原始句子
        if not is_valid_sentence(sent):
            continue
        # 再清洗
        cleaned = clean_sentence(sent)
        if len(cleaned) >= 4:
            yield cleaned


def main():
    parser = argparse.ArgumentParser(description='从维基百科生成训练数据')
    parser.add_argument('--xml', type=str, default='zhwiki-latest-pages-articles.xml')
    parser.add_argument('--max-articles', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output', type=str, default='data/train_data.jsonl')
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    xml_path = os.path.join(project_root, args.xml)
    output_path = os.path.join(project_root, args.output)
    
    if not os.path.exists(xml_path):
        print(f"错误: 找不到文件 {xml_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("维基百科训练数据生成（改进版）")
    print("=" * 60)
    print(f"输入: {xml_path}")
    print(f"输出: {output_path}")
    print(f"最大文章: {args.max_articles or '全部'}")
    print(f"最大样本: {args.max_samples or '全部'}")
    print(f"开始: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # 初始化
    from preprocess.wiki_parser import parse_wiki_xml
    
    bloom = BloomFilter(capacity=2000000)  # 200万容量
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 统计
    stats = {
        'articles': 0,
        'sentences': 0,
        'duplicates': 0,
        'invalid': 0,
        'saved': 0,
    }
    
    # 初始化 jieba 和 OpenCC
    print("\n[准备] 加载 jieba 和 OpenCC...")
    jieba.initialize()
    cc = OpenCC('t2s')  # 繁体转简体
    print("  加载完成")
    
    # 解析并处理
    print(f"\n[Step 1] 解析 XML 并提取句子（流式处理）...")
    
    with open(output_path, 'wb') as f:
        progress = ProgressTracker(total=args.max_articles, desc="文章")
        
        for title, text in parse_wiki_xml(xml_path, max_articles=args.max_articles):
            stats['articles'] += 1
            
            # 繁体转简体
            text = cc.convert(text)
            
            # 提取句子
            for sent in extract_sentences_from_text(text):
                stats['sentences'] += 1
                
                # 去重
                if sent in bloom:
                    stats['duplicates'] += 1
                    continue
                bloom.add(sent)
                
                # 转拼音
                try:
                    pinyins = hanzi_to_pinyin(sent)
                    if len(pinyins) != len(sent):
                        stats['invalid'] += 1
                        continue
                    
                    # 写入
                    record = {
                        "pinyin": " ".join(pinyins),
                        "hanzi": sent
                    }
                    f.write(orjson.dumps(record) + b'\n')
                    stats['saved'] += 1
                    
                    # 检查是否达到最大样本数
                    if args.max_samples and stats['saved'] >= args.max_samples:
                        progress.finish()
                        break
                        
                except Exception:
                    stats['invalid'] += 1
                    continue
            
            progress.update()
            
            if args.max_samples and stats['saved'] >= args.max_samples:
                break
        
        progress.finish()
    
    # 输出统计
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"  处理文章: {stats['articles']}")
    print(f"  提取句子: {stats['sentences']}")
    print(f"  重复过滤: {stats['duplicates']}")
    print(f"  无效过滤: {stats['invalid']}")
    print(f"  保存样本: {stats['saved']}")
    print(f"  去重率: {stats['duplicates']/max(stats['sentences'],1)*100:.1f}%")
    print(f"结束: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # 显示样本
    print("\n[样本预览]")
    with open(output_path, 'rb') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            record = orjson.loads(line)
            print(f"  {record['hanzi']}")
            print(f"    → {record['pinyin']}")


if __name__ == '__main__':
    main()
