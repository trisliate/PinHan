#!/usr/bin/env python3
"""从维基百科 XML 提取高质量中文句子。

提取完整的句子（而非碎片），保留上下文结构。

输出格式 (JSONL):
    {"text": "今天天气很好。", "pinyin": "jin1 tian1 tian1 qi4 hen3 hao3 。"}

使用:
    python preprocess/extract_corpus.py -i data/zhwiki.xml -o data/corpus.jsonl --max 100000
"""
import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from functools import lru_cache

try:
    from pypinyin import lazy_pinyin, Style
    from opencc import OpenCC
    import orjson
except ImportError as e:
    print(f'缺少依赖: {e}')
    print('pip install pypinyin opencc-python-reimplemented orjson')
    sys.exit(1)

# 繁简转换器
_opencc = OpenCC('t2s')

# 正则表达式
WIKI_MARKUP_RE = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')  # [[链接|文字]] -> 文字
WIKI_TEMPLATE_RE = re.compile(r'\{\{[^}]+\}\}')  # {{模板}}
WIKI_REF_RE = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^/]*/?>', re.DOTALL)  # <ref>...</ref>
HTML_TAG_RE = re.compile(r'<[^>]+>')  # HTML 标签
MULTI_SPACE_RE = re.compile(r'\s+')  # 多空格
CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')  # 中文字符


def clean_wiki_text(text: str) -> str:
    """清理维基百科标记。"""
    # 移除维基模板
    text = WIKI_TEMPLATE_RE.sub('', text)
    # 提取链接文字
    text = WIKI_MARKUP_RE.sub(r'\1', text)
    # 移除引用
    text = WIKI_REF_RE.sub('', text)
    # 移除 HTML
    text = HTML_TAG_RE.sub('', text)
    # 规范化空白
    text = MULTI_SPACE_RE.sub(' ', text)
    return text.strip()


def extract_sentences(text: str, min_chars: int = 5, max_chars: int = 100) -> list:
    """从文本中提取句子。
    
    Args:
        text: 输入文本
        min_chars: 最少中文字符数
        max_chars: 最多中文字符数
    
    Returns:
        句子列表
    """
    # 按句号、问号、感叹号分割
    parts = re.split(r'([。！？])', text)
    
    sentences = []
    current = ''
    
    for i, part in enumerate(parts):
        if part in '。！？':
            current += part
            # 检查句子质量
            chinese_count = len(CHINESE_RE.findall(current))
            if min_chars <= chinese_count <= max_chars:
                # 过滤低质量句子
                if not re.search(r'^\s*\d', current):  # 不以数字开头
                    if not re.search(r'[a-zA-Z]{3,}', current):  # 不含长英文
                        sentences.append(current.strip())
            current = ''
        else:
            current = part
    
    return sentences


@lru_cache(maxsize=50000)
def char_to_pinyin(char: str) -> str:
    """单字转拼音（带缓存）。"""
    try:
        py = lazy_pinyin(char, style=Style.TONE3, errors='ignore')
        return py[0] if py else char
    except Exception:
        return char


def text_to_pinyin(text: str) -> str:
    """文本转拼音。
    
    保留标点符号，用空格分隔拼音。
    """
    result = []
    for char in text:
        if CHINESE_RE.match(char):
            result.append(char_to_pinyin(char))
        elif char in '。！？，、；：""''（）【】':
            result.append(char)
        elif char == ' ':
            continue  # 忽略空格
        else:
            result.append(char)
    return ' '.join(result)


def parse_wiki_xml(xml_path: Path, output_path: Path, max_sentences: int = None,
                   min_chars: int = 5, max_chars: int = 100, 
                   report_interval: int = 10000) -> int:
    """解析维基百科 XML 并提取句子。
    
    Returns:
        提取的句子数量
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"解析: {xml_path}")
    print(f"输出: {output_path}")
    print(f"参数: min_chars={min_chars}, max_chars={max_chars}")
    if max_sentences:
        print(f"最大句子数: {max_sentences}")
    
    # 用于去重
    seen = set()
    count = 0
    article_count = 0
    start_time = time.time()
    
    with open(output_path, 'wb') as out:
        context = ET.iterparse(xml_path, events=('end',))
        
        for event, elem in context:
            # 只处理 <text> 元素
            if not elem.tag.endswith('text'):
                elem.clear()
                continue
            
            text = elem.text or ''
            elem.clear()
            
            if not text:
                continue
            
            article_count += 1
            
            # 繁简转换
            try:
                text = _opencc.convert(text)
            except Exception:
                continue
            
            # 清理标记
            text = clean_wiki_text(text)
            
            # 提取句子
            for sentence in extract_sentences(text, min_chars, max_chars):
                # 去重
                if sentence in seen:
                    continue
                seen.add(sentence)
                
                # 转拼音
                pinyin = text_to_pinyin(sentence)
                
                # 写入
                record = {'text': sentence, 'pinyin': pinyin}
                out.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                count += 1
                
                # 进度报告
                if count % report_interval == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    print(f"  已提取 {count:,} 句 ({article_count:,} 文章), {rate:.0f} 句/s")
                
                if max_sentences and count >= max_sentences:
                    break
            
            if max_sentences and count >= max_sentences:
                break
    
    elapsed = time.time() - start_time
    print(f"\n完成: {count:,} 句, {article_count:,} 文章, 用时 {elapsed:.1f}s")
    return count


def main():
    parser = argparse.ArgumentParser(description='从维基百科 XML 提取中文句子')
    parser.add_argument('-i', '--input', required=True, help='输入 XML 文件')
    parser.add_argument('-o', '--output', default='data/corpus.jsonl', help='输出 JSONL')
    parser.add_argument('--max', type=int, default=0, help='最大句子数 (0=无限)')
    parser.add_argument('--min-chars', type=int, default=5, help='最少中文字符')
    parser.add_argument('--max-chars', type=int, default=100, help='最多中文字符')
    args = parser.parse_args()
    
    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"错误: 文件不存在 {xml_path}")
        sys.exit(1)
    
    parse_wiki_xml(
        xml_path,
        Path(args.output),
        max_sentences=args.max if args.max > 0 else None,
        min_chars=args.min_chars,
        max_chars=args.max_chars
    )


if __name__ == '__main__':
    main()
