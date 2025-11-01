#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
高性能的中文句子抽取与拼音生成工具（面向大规模语料）。

功能：
- 从 Wiki XML 或纯文本中抽取中文句子
- 繁体->简体转换
- 生成拼音（数字声调，单字缓存快速模式）
- 默认启用去重（Bloom filter，使用 bitarray + mmh3）
- 批量写入与 orjson 高速序列化

依赖库（必须安装）：pypinyin, opencc-python-reimplemented, orjson, bitarray, mmh3
安装：pip install pypinyin opencc-python-reimplemented orjson bitarray mmh3

使用示例：
    python preprocess/extract_and_clean.py --input wiki.xml --output data/clean.jsonl
"""
import argparse
import os
import re
import sys
import time
import math
import xml.etree.ElementTree as ET
from functools import lru_cache

try:
    from pypinyin import lazy_pinyin, Style
    from opencc import OpenCC
    import orjson
    from bitarray import bitarray
    import mmh3
except ImportError as e:
    print(f'缺少依赖库: {e}')
    print('请运行: pip install pypinyin opencc-python-reimplemented orjson bitarray mmh3')
    sys.exit(1)

_opencc = OpenCC('t2s')
CHINESE_SEG_RE = re.compile(r'[\u4e00-\u9fff]+')
SEP_RE = re.compile(r'[。！？；\n]+')
ASCII_RE = re.compile(r'[A-Za-z0-9]')


def extract_chinese_sequences(text):
    """从文本中提取连续的汉字序列。"""
    return CHINESE_SEG_RE.findall(text)


@lru_cache(maxsize=200000)
def _char_pinyin_cached(ch):
    """缓存的单字拼音查询。"""
    try:
        py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')
        return py[0] if py else ''
    except Exception:
        return ''


def to_pinyin(hanzi):
    """将汉字转换为拼音（数字声调）。"""
    result = []
    for ch in hanzi:
        if CHINESE_SEG_RE.match(ch):
            result.append(_char_pinyin_cached(ch))
        else:
            result.append(ch)
    return ' '.join(result)


def process_text_block(text, min_len=2, max_len=64, convert_trad=True):
    """处理文本块并返回 (hanzi, pinyin) 对列表。"""
    if not text:
        return []
    
    if convert_trad:
        try:
            text = _opencc.convert(text)
        except Exception:
            pass

    parts = SEP_RE.split(text)
    recs = []
    for part in parts:
        if not part:
            continue
        for seg in extract_chinese_sequences(part):
            s = seg.strip()
            if not s or len(s) < min_len or len(s) > max_len or ASCII_RE.search(s):
                continue
            try:
                pinyin = to_pinyin(s)
                recs.append({'hanzi': s, 'pinyin': pinyin})
            except Exception:
                continue
    return recs


def _build_bloom(expected_unique, fpr):
    """构建 Bloom filter 并返回 (bloom, k)。"""
    m = math.ceil(-(expected_unique * math.log(fpr)) / (math.log(2) ** 2))
    k = max(1, int((m / expected_unique) * math.log(2)))
    bloom = bitarray(m)
    bloom.setall(0)
    return bloom, k


def _bloom_check_add(bloom, k, s):
    """检查并添加字符串到 Bloom filter。返回 True 如果重复。"""
    hb = s.encode('utf-8')
    h1, h2 = mmh3.hash64(hb)
    h1 &= 0xFFFFFFFFFFFFFFFF
    h2 &= 0xFFFFFFFFFFFFFFFF
    m = len(bloom)
    is_duplicate = True
    for i in range(k):
        idx = (h1 + i * h2) % m
        if not bloom[idx]:
            bloom[idx] = 1
            is_duplicate = False
    return is_duplicate


def _format_size(num_bytes):
    """格式化字节数。"""
    KB = 1024.0
    MB = KB * 1024.0
    GB = MB * 1024.0
    if num_bytes < MB:
        return f"{num_bytes / KB:.2f} KB"
    if num_bytes < GB:
        return f"{num_bytes / MB:.2f} MB"
    return f"{num_bytes / GB:.2f} GB"


def _print_progress(total_written, max_sentences, start_time, now, buf):
    """打印进度信息。"""
    elapsed = now - start_time
    rate = total_written / elapsed if elapsed > 0 else 0
    total_bytes = sum(len(x) for x in buf)
    size = _format_size(total_bytes)
    
    if max_sentences:
        pct = total_written / max_sentences * 100
        print(f'[{pct:5.1f}%] 已写入 {total_written:,}/{max_sentences:,} ({size})，速率 {rate:.1f} 条/s')
    else:
        print(f'已写入 {total_written:,} ({size})，速率 {rate:.1f} 条/s，用时 {elapsed:.1f}s')


def parse_file(inpath, outpath, is_xml=True, max_sentences=None, convert_trad=True, 
               min_len=2, max_len=64, dedup=True, buffer_size=10000, expected_unique=20000000, 
               fpr=1e-6, report_seconds=30, no_time_report=False):
    """统一的文件解析函数（支持 XML 和纯文本）。"""
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    
    # 初始化 Bloom filter
    bloom, k = _build_bloom(expected_unique, fpr)
    print(f'初始化 Bloom filter: 大小={len(bloom)} bits, k={k}')
    
    buf = []
    total_written = 0
    start_time = time.perf_counter()
    last_report = start_time
    
    with open(outpath, 'wb') as out_file:
        if is_xml:
            print(f'开始解析 XML 文件...')
            context = ET.iterparse(inpath, events=('end',))
            for _, elem in context:
                if not elem.tag.endswith('text'):
                    elem.clear()
                    continue
                
                text = elem.text or ''
                elem.clear()
                
                for rec in process_text_block(text, min_len, max_len, convert_trad):
                    if dedup and _bloom_check_add(bloom, k, rec['hanzi']):
                        continue
                    
                    buf.append(orjson.dumps(rec) + b'\n')
                    total_written += 1
                    
                    if len(buf) >= buffer_size:
                        now = time.perf_counter()
                        _print_progress(total_written, max_sentences, start_time, now, buf)
                        out_file.write(b''.join(buf))
                        buf = []
                        last_report = now
                    
                    if max_sentences and total_written >= max_sentences:
                        break
                
                if max_sentences and total_written >= max_sentences:
                    break
        else:
            print(f'开始解析纯文本文件...')
            with open(inpath, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    
                    for rec in process_text_block(line, min_len, max_len, convert_trad):
                        if dedup and _bloom_check_add(bloom, k, rec['hanzi']):
                            continue
                        
                        buf.append(orjson.dumps(rec) + b'\n')
                        total_written += 1
                        
                        if len(buf) >= buffer_size:
                            now = time.perf_counter()
                            _print_progress(total_written, max_sentences, start_time, now, buf)
                            out_file.write(b''.join(buf))
                            buf = []
                            last_report = now
                        
                        if max_sentences and total_written >= max_sentences:
                            break
                    
                    if max_sentences and total_written >= max_sentences:
                        break
        
        # Final flush
        if buf:
            out_file.write(b''.join(buf))
            _print_progress(total_written, max_sentences, start_time, time.perf_counter(), buf)
    
    elapsed = time.perf_counter() - start_time
    rate = total_written / elapsed if elapsed > 0 else 0
    print(f'\n完成：写入 {total_written:,} 条，用时 {elapsed:.1f}s，平均速率 {rate:.1f} 条/s')
    return total_written


def detect_is_xml(path):
    """检测文件是否为 XML。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            head = f.read(4096)
            return '<text' in head or '<page' in head or '<mediawiki' in head
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='从 Wiki dump 或纯文本抽取中文句子并生成拼音')
    parser.add_argument('--input', '-i', required=True, help='输入文件')
    parser.add_argument('--output', '-o', default='data/clean_data.jsonl', help='输出 JSONL 文件')
    parser.add_argument('--max', type=int, default=0, help='最大句子数（0 表示不限）')
    parser.add_argument('--min-len', type=int, default=2, help='最小句子长度')
    parser.add_argument('--max-len', type=int, default=64, help='最大句子长度')
    parser.add_argument('--no-opencc', action='store_true', help='禁用繁简转换')
    parser.add_argument('--no-dedup', action='store_true', help='禁用去重（默认启用）')
    parser.add_argument('--buffer-size', type=int, default=10000, help='写入缓冲区大小')
    parser.add_argument('--expected-unique', type=int, default=20000000, help='预期唯一元素数')
    parser.add_argument('--fpr', type=float, default=1e-6, help='布隆过滤器误判率')
    parser.add_argument('--report-seconds', type=int, default=30, help='进度报告间隔（秒）')
    parser.add_argument('--no-time-report', action='store_true', help='禁用时间based报告')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'错误：输入文件不存在 {args.input}')
        sys.exit(1)

    is_xml = detect_is_xml(args.input)
    print(f'检测文件类型：{"XML" if is_xml else "纯文本"}')
    
    parse_file(
        args.input,
        args.output,
        is_xml=is_xml,
        max_sentences=args.max if args.max > 0 else None,
        convert_trad=(not args.no_opencc),
        min_len=args.min_len,
        max_len=args.max_len,
        dedup=(not args.no_dedup),
        buffer_size=args.buffer_size,
        expected_unique=args.expected_unique,
        fpr=args.fpr,
        report_seconds=args.report_seconds,
        no_time_report=args.no_time_report,
    )
    print(f'完成，输出文件: {args.output}')


if __name__ == '__main__':
    main()
