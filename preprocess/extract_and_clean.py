#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_and_clean.py

高性能的中文句子抽取与拼音生成工具（面向大规模语料）。

功能：
- 从 Wiki XML 或纯文本中抽取中文句子
- 繁体->简体转换（强制要求 opencc）
- 生成拼音（数字声调），支持快速单字缓存模式与精确整句模式
- 可选近似去重（Bloom filter，使用 bitarray + mmh3）
- 批量写入与 orjson 高速序列化

此脚本在高性能模式下依赖以下第三方库（必须安装）：
    pypinyin, opencc-python-reimplemented, orjson, bitarray, mmh3
安装：
    pip install pypinyin opencc-python-reimplemented orjson bitarray mmh3

注意：脚本现在在启用 `--dedup` 时仅支持 Bloom 去重（高性能模式），并默认使用大写入缓冲（`--buffer-size`）以减少磁盘 I/O 开销。

使用：
    python preprocess\extract_and_clean.py --input "C:\path\to\2025-10-zhwiki-latest-pages-articles.xml" --output "data/clean_wiki.jsonl"
"""

import argparse
import os
import re
import sys
import time
import math
import xml.etree.ElementTree as ET
from functools import lru_cache

# 第三方库（高性能模式必需）。若缺失会抛出清晰的 ImportError 提示安装方法。
try:
    from pypinyin import lazy_pinyin, Style
except Exception as e:
    raise ImportError('需要安装 pypinyin：pip install pypinyin') from e

try:
    from opencc import OpenCC
    _opencc = OpenCC('t2s')
except Exception as e:
    raise ImportError('需要安装 opencc-python-reimplemented：pip install opencc-python-reimplemented') from e

try:
    import orjson as _orjson
except Exception as e:
    raise ImportError('需要安装 orjson：pip install orjson') from e

try:
    from bitarray import bitarray
except Exception as e:
    raise ImportError('需要安装 bitarray：pip install bitarray') from e

try:
    import mmh3
except Exception as e:
    raise ImportError('需要安装 mmh3：pip install mmh3') from e

CHINESE_SEG_RE = re.compile(r'[\u4e00-\u9fff]+')


def extract_chinese_sequences(text):
    """从文本中提取连续的汉字序列列表。"""
    return CHINESE_SEG_RE.findall(text)


def to_pinyin(hanzi):
    """将汉字串转换为带数字声调的拼音串。

    两种模式：
    - 精确模式（由全局标志 `EXACT_PINYIN` 控制）：对整句调用 pypinyin（更精确，但慢）
    - 快速模式（默认）：逐字使用缓存获取拼音（适合大规模语料，可能在少数多音字处与整句略有差别）
    """
    if lazy_pinyin is None or Style is None:
        raise RuntimeError('需要安装 pypinyin。请运行: pip install pypinyin')
    if globals().get('EXACT_PINYIN', False):
        py = lazy_pinyin(hanzi, style=Style.TONE3, errors='ignore')
        return ' '.join(py)
    # 快速单字缓存模式
    res = []
    for ch in hanzi:
        if CHINESE_SEG_RE.match(ch):
            res.append(_char_pinyin_cached(ch))
        else:
            # 对非汉字直接保留原字符（极少发生）
            res.append(ch)
    return ' '.join(res)


@lru_cache(maxsize=200000)
def _char_pinyin_cached(ch):
    """返回单个汉字的拼音（数字声调），作为缓存的基本单元。"""
    try:
        py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')
        return py[0] if py else ''
    except Exception:
        return ''


def process_text_block(text, min_len=2, max_len=64, convert_trad=True):
    """处理一段文本并返回若干 {'hanzi','pinyin'} 的记录字典列表。

    本函数仅负责文本清洗与拼音生成，不负责磁盘写入；
    调用方应负责缓冲与序列化以获得更高的 I/O 性能。
    """
    if not text:
        return []
    if convert_trad and _opencc is not None:
        try:
            text = _opencc.convert(text)
        except Exception:
            pass

    parts = re.split(r'[。！？；\n]+', text)
    recs = []
    for part in parts:
        if not part:
            continue
        for seg in extract_chinese_sequences(part):
            s = seg.strip()
            if not s:
                continue
            # 过滤掉包含 ASCII 字母/数字的片段
            if re.search(r'[A-Za-z0-9]', s):
                continue
            if len(s) < min_len or len(s) > max_len:
                continue
            try:
                pinyin = to_pinyin(s)
            except Exception as e:
                # 如果 pypinyin 出错，跳过该句
                print('拼音生成错误:', s[:20], '...', e)
                continue
            rec = {'hanzi': s, 'pinyin': pinyin}
            recs.append(rec)
    return recs


def _format_duration(seconds: float) -> str:
    """格式化持续时间：小于 1s 用 ms 表示，>=1s 用 s 表示。"""
    if seconds < 1.0:
        return f"{seconds*1000:.3f} ms"
    return f"{seconds:.3f} s"


def _format_size(num_bytes: float) -> str:
    """格式化字节数：
    - 小于 1 MB 时以 KB 为单位显示（例如 512 B -> 0.50 KB）
    - 小于 1 GB 时以 MB 为单位显示
    - 更大时以 GB 为单位显示
    该显示规则遵循用户要求：小于 MB 显示为 KB，>=MB 显示为 MB，达到 GB 显示为 GB。
    """
    KB = 1024.0
    MB = KB * 1024.0
    GB = MB * 1024.0
    # 小于 1 MB 使用 KB 单位显示
    if num_bytes < MB:
        return f"{num_bytes / KB:.2f} KB"
    if num_bytes < GB:
        return f"{num_bytes / MB:.2f} MB"
    return f"{num_bytes / GB:.2f} GB"


def _format_throughput(bytes_per_sec: float) -> str:
    """格式化速率（字节/秒）为合适单位并附带 /s 后缀。"""
    if bytes_per_sec == float('inf'):
        return "inf B/s"
    KB = 1024.0
    MB = KB * 1024.0
    GB = MB * 1024.0
    # 与 _format_size 保持一致：小于 1 MB 时使用 KB/s，>=1MB 使用 MB/s，>=1GB 使用 GB/s
    if bytes_per_sec < MB:
        return f"{bytes_per_sec / KB:.2f} KB/s"
    if bytes_per_sec < GB:
        return f"{bytes_per_sec / MB:.2f} MB/s"
    return f"{bytes_per_sec / GB:.2f} GB/s"


def parse_wiki_xml(inpath, outpath, max_sentences=None, convert_trad=True, min_len=2, max_len=64,
                   dedup=False, buffer_size=10000, dedup_method='bloom', expected_unique=20000000, fpr=1e-6,
                   report_interval_seconds=30, no_time_report=False):
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)

    # prepare dedup structure
    if dedup:
        # 高性能模式：强制使用 Bloom（不再支持 set 回退）
        if dedup_method != 'bloom':
            raise ValueError('高性能模式仅支持 bloom 去重，请使用 --dedup-method bloom')
        # 计算 Bloom 参数并初始化
        m = math.ceil(- (expected_unique * math.log(fpr)) / (math.log(2) ** 2))
        k = max(1, int((m / expected_unique) * math.log(2)))
        bloom = bitarray(m)
        bloom.setall(0)

        def _get_two_hashes(b):
            # 使用 mmh3.hash64 获取两个 64-bit 哈希值
            h1, h2 = mmh3.hash64(b)
            h1 &= 0xFFFFFFFFFFFFFFFF
            h2 &= 0xFFFFFFFFFFFFFFFF
            return h1, h2

        def seen_check_add(s):
            hb = s.encode('utf-8')
            h1, h2 = _get_two_hashes(hb)
            added = False
            for i in range(k):
                idx = (h1 + i * h2) % m
                if not bloom[idx]:
                    bloom[idx] = 1
                    added = True
            return not added
    else:
        def seen_check_add(s):
            return False

    total_written = 0
    context = ET.iterparse(inpath, events=('end',))
    # 高性能方案：强制使用 orjson + 二进制写入
    buf = []  # 序列化后的行缓冲（bytes）
    written_since_report = 0
    start_time = time.perf_counter()
    last_report = start_time
    # 最小报告间隔，避免连续两次打印间隔为零的情况（秒）
    min_report_interval = 0.01
    texts_seen = 0
    # 按记录数触发的报告阈值（条数）
    report_interval = 10000
    out_file_ctx = open(outpath, 'wb')
    with out_file_ctx as out_file:
        for _, elem in context:
            tag = elem.tag
            if tag.endswith('text'): 
                texts_seen += 1
                text = elem.text or ''
                recs = process_text_block(text, min_len, max_len, convert_trad)
                for rec in recs:
                    hanzi = rec['hanzi']
                    if seen_check_add(hanzi):
                        continue
                    buf.append(_orjson.dumps(rec) + b'\n')
                    total_written += 1
                    written_since_report += 1
                    # flush buffer
                    if len(buf) >= buffer_size:
                        n_buf = len(buf)
                        # 先计算将要写入的总字节数，再执行写入
                        total_bytes = sum(len(x) for x in buf)
                        flush_start = time.perf_counter()
                        out_file.write(b''.join(buf))
                        buf = []
                        flush_elapsed = time.perf_counter() - flush_start
                        dur_str = _format_duration(flush_elapsed)
                        bytes_per_sec = (total_bytes / flush_elapsed) if flush_elapsed > 0 else float('inf')
                        size_str = _format_size(total_bytes)
                        throughput_str = _format_throughput(bytes_per_sec)
                        if max_sentences and max_sentences > 0:
                            pct = total_written / max_sentences * 100
                            print(f'Flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written}/{max_sentences} ({pct:.2f}%)，texts_seen {texts_seen}')
                        else:
                            print(f'Flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written} 条，texts_seen {texts_seen}')
                        # 不记录 last_flush_time（未使用），仅更新统计
                elem.clear()
            now = time.perf_counter()
            # 两种触发进度报告的方式：达到条数阈值，或超过时间阈值（适用于 --max=0 的情况）
            # 进度报告触发：按条数触发或（启用且）按时间触发（且距离上次报告超过最小间隔）
            time_trigger = (not no_time_report) and (now - last_report >= report_interval_seconds) and (now - last_report >= min_report_interval)
            count_trigger = (report_interval and total_written and total_written % report_interval == 0)
            if count_trigger or time_trigger:
                elapsed = now - start_time
                interval_elapsed = now - last_report
                # 最近区间速率（条/s），以及总体平均速率
                recent_rate = (written_since_report / interval_elapsed) if interval_elapsed > 0 else 0
                avg_rate = total_written / elapsed if elapsed > 0 else 0
                print(f'已写入 {total_written} 条  总用时 {elapsed:.1f}s  最近间隔用时 {interval_elapsed:.3f}s  最近速率 {recent_rate:.1f} 条/s  平均速率 {avg_rate:.1f} 条/s  texts_seen {texts_seen}')
                written_since_report = 0
                last_report = now
            if max_sentences and total_written >= max_sentences:
                break
        # final flush
        if buf:
            n_buf = len(buf)
            total_bytes = sum(len(x) for x in buf)
            flush_start = time.perf_counter()
            out_file.write(b''.join(buf))
            flush_elapsed = time.perf_counter() - flush_start
            dur_str = _format_duration(flush_elapsed)
            bytes_per_sec = (total_bytes / flush_elapsed) if flush_elapsed > 0 else float('inf')
            size_str = _format_size(total_bytes)
            throughput_str = _format_throughput(bytes_per_sec)
            if max_sentences and max_sentences > 0:
                pct = total_written / max_sentences * 100
                print(f'Final flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written}/{max_sentences} ({pct:.2f}%)，texts_seen {texts_seen}')
            else:
                print(f'Final flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written} 条，texts_seen {texts_seen}')
            # 不记录 last_flush_time（未使用）
        # overall summary for this parse
        total_elapsed = time.perf_counter() - start_time
        avg_rate = total_written / total_elapsed if total_elapsed > 0 else 0
        print(f'解析完成：总写入 {total_written} 条，用时 {total_elapsed:.1f}s，平均速率 {avg_rate:.1f} 条/s')
    return total_written


def parse_plain_text(inpath, outpath, max_sentences=None, convert_trad=True, min_len=2, max_len=64,
                     dedup=False, buffer_size=10000, dedup_method='bloom', expected_unique=20000000, fpr=1e-6,
                     report_interval_seconds=30, no_time_report=False):
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    # 重用 parse_wiki_xml 的去重与缓冲逻辑
    if dedup:
        if dedup_method != 'bloom':
            raise ValueError('高性能模式仅支持 bloom 去重，请使用 --dedup-method bloom')
        m = math.ceil(- (expected_unique * math.log(fpr)) / (math.log(2) ** 2))
        k = max(1, int((m / expected_unique) * math.log(2)))
        bloom = bitarray(m)
        bloom.setall(0)
        def _get_two_hashes(b):
            h1, h2 = mmh3.hash64(b)
            h1 &= 0xFFFFFFFFFFFFFFFF
            h2 &= 0xFFFFFFFFFFFFFFFF
            return h1, h2
        def seen_check_add(s):
            hb = s.encode('utf-8')
            h1, h2 = _get_two_hashes(hb)
            added = False
            for i in range(k):
                idx = (h1 + i * h2) % m
                if not bloom[idx]:
                    bloom[idx] = 1
                    added = True
            return not added
    else:
        def seen_check_add(s):
            return False

    # 高性能：使用 orjson + 二进制写入（无需额外变量）
    buf = []
    written_since_report = 0
    total_written = 0
    lines_seen = 0
    start_time = time.perf_counter()
    last_report = start_time
    min_report_interval = 0.01
    # 按记录数触发的报告阈值（条数）
    report_interval = 10000
    out_file_ctx = open(outpath, 'wb')
    with open(inpath, 'r', encoding='utf-8') as fin, out_file_ctx as out_file:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            lines_seen += 1
            recs = process_text_block(line, min_len, max_len, convert_trad)
            for rec in recs:
                hanzi = rec['hanzi']
                if seen_check_add(hanzi):
                    continue
                # 高性能模式：直接使用 orjson
                buf.append(_orjson.dumps(rec) + b'\n')
                total_written += 1
                written_since_report += 1
                if len(buf) >= buffer_size:
                    n_buf = len(buf)
                    total_bytes = sum(len(x) for x in buf)
                    flush_start = time.perf_counter()
                    out_file.write(b''.join(buf))
                    buf = []
                    flush_elapsed = time.perf_counter() - flush_start
                    dur_str = _format_duration(flush_elapsed)
                    bytes_per_sec = (total_bytes / flush_elapsed) if flush_elapsed > 0 else float('inf')
                    size_str = _format_size(total_bytes)
                    throughput_str = _format_throughput(bytes_per_sec)
                    if max_sentences and max_sentences > 0:
                        pct = total_written / max_sentences * 100
                        print(f'Flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written}/{max_sentences} ({pct:.2f}%)，lines_seen {lines_seen}')
                    else:
                        print(f'Flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written} 条，lines_seen {lines_seen}')
                    # 不记录 last_flush_time（未使用）
            now = time.perf_counter()
            time_trigger = (not no_time_report) and (now - last_report >= report_interval_seconds) and (now - last_report >= min_report_interval)
            count_trigger = (report_interval and total_written and total_written % report_interval == 0)
            if count_trigger or time_trigger:
                elapsed = now - start_time
                interval_elapsed = now - last_report
                recent_rate = (written_since_report / interval_elapsed) if interval_elapsed > 0 else 0
                avg_rate = total_written / elapsed if elapsed > 0 else 0
                print(f'已写入 {total_written} 条  总用时 {elapsed:.1f}s  最近间隔用时 {interval_elapsed:.3f}s  最近速率 {recent_rate:.1f} 条/s  平均速率 {avg_rate:.1f} 条/s  lines_seen {lines_seen}')
                written_since_report = 0
                last_report = now
            if max_sentences and total_written >= max_sentences:
                break
        if buf:
            n_buf = len(buf)
            total_bytes = sum(len(x) for x in buf)
            flush_start = time.perf_counter()
            out_file.write(b''.join(buf))
            flush_elapsed = time.perf_counter() - flush_start
            dur_str = _format_duration(flush_elapsed)
            bytes_per_sec = (total_bytes / flush_elapsed) if flush_elapsed > 0 else float('inf')
            size_str = _format_size(total_bytes)
            throughput_str = _format_throughput(bytes_per_sec)
            if max_sentences and max_sentences > 0:
                pct = total_written / max_sentences * 100
                print(f'Final flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written}/{max_sentences} ({pct:.2f}%)，lines_seen {lines_seen}')
            else:
                print(f'Final flush 写入 {n_buf} 条 ({size_str})，flush 用时 {dur_str}，速度 {throughput_str}，当前已写入 {total_written} 条，lines_seen {lines_seen}')
            # 不记录 last_flush_time（未使用）
        total_elapsed = time.perf_counter() - start_time
        avg_rate = total_written / total_elapsed if total_elapsed > 0 else 0
        print(f'解析完成：总写入 {total_written} 条，用时 {total_elapsed:.1f}s，平均速率 {avg_rate:.1f} 条/s')
    return total_written


def detect_is_xml(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            head = f.read(4096)
            return ('<text' in head) or ('<page' in head) or ('<mediawiki' in head)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='从 Wiki dump 或纯文本抽取中文句子并生成拼音（用于训练数据清洗）')
    parser.add_argument('--input', '-i', required=True, help='输入文件（Wiki XML 或纯文本）')
    parser.add_argument('--output', '-o', default='data/clean_data.jsonl', help='输出 jsonl 文件')
    parser.add_argument('--max', type=int, default=0, help='最大保留句子数（0 表示不限，默认 0）')
    parser.add_argument('--min-len', type=int, default=2, help='最小句子长度（汉字数）')
    parser.add_argument('--max-len', type=int, default=64, help='最大句子长度（汉字数）')
    parser.add_argument('--no-opencc', dest='no_opencc', action='store_true', help='禁用繁简转换')
    parser.add_argument('--dedup', dest='dedup', action='store_true', help='启用去重（会使用较多内存），默认不去重以支持大文件）')
    parser.add_argument('--buffer-size', type=int, default=10000, help='写入缓冲区大小（记录数），达到后批量写入磁盘，默认 10000')
    # 如果用户未传 --dedup-method，则留待后续决定（优先使用已安装的高效方法）
    parser.add_argument('--dedup-method', choices=['set', 'bloom'], default=None, help='去重方法：set 精确去重 或 bloom 近似去重（节省内存）；若不指定且已安装 bitarray/mmh3，则优先使用 bloom')
    parser.add_argument('--expected-unique', type=int, default=20000000, help='布隆过滤器预计的唯一元素数（仅在 --dedup-method bloom 时生效）')
    parser.add_argument('--fpr', type=float, default=1e-6, help='布隆过滤器期望的误判率（仅在 bloom 时生效）')
    parser.add_argument('--exact-pinyin', dest='exact_pinyin', action='store_true', help='使用精确的整句 pypinyin（慢），默认使用单字缓存快速模式')
    parser.add_argument('--report-seconds', type=int, default=30, help='在全部清洗（--max=0）时按秒定期打印进度，单位秒（默认30s）')
    parser.add_argument('--no-time-report', action='store_true', help='禁用基于时间的定期进度打印（仅按条数报告）')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('输入文件不存在:', args.input)
        sys.exit(1)

    convert_trad = (not args.no_opencc)

    # 设置全局拼音模式标志：默认 False（快速缓存模式），若指定 --exact-pinyin 则启用精确模式
    globals()['EXACT_PINYIN'] = bool(args.exact_pinyin)

    is_xml = detect_is_xml(args.input)
    print('检测文件类型 xml=', is_xml)
    # 决定去重方法：若用户启用了去重但未指定方法，优先使用已安装的 bloom
    chosen_dedup_method = args.dedup_method
    if args.dedup and chosen_dedup_method is None:
        if bitarray is not None and mmh3 is not None:
            chosen_dedup_method = 'bloom'
        else:
            chosen_dedup_method = 'set'

    if is_xml:
        count = parse_wiki_xml(args.input, args.output, args.max, convert_trad, args.min_len, args.max_len,
                               dedup=args.dedup, buffer_size=args.buffer_size, dedup_method=chosen_dedup_method,
                               expected_unique=args.expected_unique, fpr=args.fpr,
                               report_interval_seconds=args.report_seconds,
                               no_time_report=args.no_time_report)
    else:
        count = parse_plain_text(args.input, args.output, args.max, convert_trad, args.min_len, args.max_len,
                                 dedup=args.dedup, buffer_size=args.buffer_size, dedup_method=chosen_dedup_method,
                                 expected_unique=args.expected_unique, fpr=args.fpr,
                                 report_interval_seconds=args.report_seconds,
                                 no_time_report=args.no_time_report)

    print(f'完成，写入 {count} 条清洗后的句子到 {args.output}')


if __name__ == '__main__':
    main()
