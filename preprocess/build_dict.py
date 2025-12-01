"""
字典构建脚本

从 CC-CEDICT 构建完整字典：
- 单字字典 (char_dict.json)
- 词组字典 (word_dict.json)
- 字频表 (char_freq.json) - 基于 jieba 词频
- 词频表 (word_freq.json) - 基于 jieba 词频
- 拼音表 (pinyin_table.txt)
"""

import gzip
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import orjson
import jieba


# 路径配置
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CEDICT_PATH = os.path.join(SCRIPT_DIR, 'cedict.txt.gz')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dicts')


def parse_cedict_line(line: str) -> Tuple[str, str, List[str]]:
    """
    解析 CC-CEDICT 单行
    
    格式: 繁體 简体 [pin1 yin1] /definition/definition/
    
    Returns:
        (简体, 繁体, [拼音列表]) 或 (None, None, None)
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None, None
    
    # 匹配格式
    match = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]', line)
    if not match:
        return None, None, None
    
    traditional = match.group(1)
    simplified = match.group(2)
    pinyin_raw = match.group(3)
    
    # 处理拼音：移除声调数字，转小写，ü→v
    pinyins = []
    for py in pinyin_raw.split():
        py = py.lower()
        py = re.sub(r'[1-5]', '', py)  # 移除声调
        py = py.replace('ü', 'v').replace('u:', 'v')
        if py:
            pinyins.append(py)
    
    return simplified, traditional, pinyins


def load_cedict() -> List[Tuple[str, str, List[str]]]:
    """加载 CC-CEDICT 全部词条"""
    entries = []
    
    print(f"加载 CC-CEDICT: {CEDICT_PATH}")
    
    with gzip.open(CEDICT_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            simplified, traditional, pinyins = parse_cedict_line(line)
            if simplified and pinyins:
                entries.append((simplified, traditional, pinyins))
    
    print(f"  词条数: {len(entries)}")
    return entries


def build_char_dict(entries: List[Tuple[str, str, List[str]]]) -> Dict[str, List[str]]:
    """
    构建单字字典
    拼音 → 汉字列表
    """
    print("构建单字字典...")
    
    char_dict = defaultdict(set)
    
    for simplified, traditional, pinyins in entries:
        # 只处理单字
        if len(simplified) == 1 and len(pinyins) == 1:
            py = pinyins[0]
            char_dict[py].add(simplified)
            # 也加入繁体
            if traditional != simplified:
                char_dict[py].add(traditional)
    
    # 转为列表
    result = {py: sorted(chars) for py, chars in char_dict.items()}
    print(f"  拼音数: {len(result)}, 单字数: {sum(len(v) for v in result.values())}")
    return result


def build_word_dict(entries: List[Tuple[str, str, List[str]]]) -> Dict[str, List[str]]:
    """
    构建词组字典
    拼音序列 → 词组列表
    """
    print("构建词组字典...")
    
    word_dict = defaultdict(list)
    
    for simplified, traditional, pinyins in entries:
        # 只处理多字词
        if len(simplified) >= 2 and len(pinyins) == len(simplified):
            key = ' '.join(pinyins)
            if simplified not in word_dict[key]:
                word_dict[key].append(simplified)
    
    result = dict(word_dict)
    print(f"  拼音组合数: {len(result)}, 词组数: {sum(len(v) for v in result.values())}")
    return result


def load_jieba_freq() -> Dict[str, int]:
    """加载 jieba 词频表"""
    print("加载 jieba 词频表...")
    jieba.initialize()
    freq = dict(jieba.dt.FREQ)
    print(f"  jieba 词频条目: {len(freq)}")
    return freq


def build_char_freq(char_dict: Dict[str, List[str]], jieba_freq: Dict[str, int]) -> Dict[str, float]:
    """
    构建字频表
    基于 jieba 词频表中的单字频率
    """
    print("构建字频表（基于 jieba）...")
    
    # 收集所有字
    all_chars = set()
    for chars in char_dict.values():
        all_chars.update(chars)
    
    # 从 jieba 获取单字频率
    char_freq_raw = {}
    for char in all_chars:
        freq = jieba_freq.get(char, 0)
        char_freq_raw[char] = freq
    
    # 归一化
    total = sum(char_freq_raw.values()) or 1
    char_freq = {char: freq / total for char, freq in char_freq_raw.items()}
    
    # 为频率为 0 的字设置最低频率
    min_freq = 1e-8
    for char in char_freq:
        if char_freq[char] == 0:
            char_freq[char] = min_freq
    
    # 统计有效频率的字数
    valid_count = sum(1 for f in char_freq.values() if f > min_freq)
    print(f"  字频条目: {len(char_freq)}, 有效频率: {valid_count}")
    return char_freq


def build_word_freq(word_dict: Dict[str, List[str]], jieba_freq: Dict[str, int]) -> Dict[str, float]:
    """
    构建词频表
    基于 jieba 词频表
    """
    print("构建词频表（基于 jieba）...")
    
    # 收集所有词
    all_words = set()
    for words in word_dict.values():
        all_words.update(words)
    
    # 从 jieba 获取词频
    word_freq_raw = {}
    for word in all_words:
        freq = jieba_freq.get(word, 0)
        word_freq_raw[word] = freq
    
    # 归一化
    total = sum(word_freq_raw.values()) or 1
    word_freq = {word: freq / total for word, freq in word_freq_raw.items()}
    
    # 为频率为 0 的词设置基础频率（按词长衰减）
    for word in word_freq:
        if word_freq[word] == 0:
            # 较短的词给稍高的基础频率
            word_freq[word] = 1e-8 / len(word)
    
    # 统计有效频率的词数
    valid_count = sum(1 for w, f in word_freq.items() if f > 1e-8 / len(w))
    print(f"  词频条目: {len(word_freq)}, 有效频率: {valid_count}")
    return word_freq


def build_pinyin_table(char_dict: Dict[str, List[str]]) -> List[str]:
    """提取拼音表"""
    return sorted(char_dict.keys())


def save_json(data, filename: str):
    """保存 JSON 文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"  已保存: {path}")


def save_txt(lines: List[str], filename: str):
    """保存文本文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  已保存: {path}")


def main():
    print("=" * 60)
    print("字典构建（基于 CC-CEDICT + jieba 词频）")
    print("=" * 60)
    
    # 1. 加载数据源
    entries = load_cedict()
    jieba_freq = load_jieba_freq()
    
    # 2. 构建各字典
    char_dict = build_char_dict(entries)
    word_dict = build_word_dict(entries)
    char_freq = build_char_freq(char_dict, jieba_freq)
    word_freq = build_word_freq(word_dict, jieba_freq)
    pinyin_table = build_pinyin_table(char_dict)
    
    # 3. 保存
    print("\n保存文件...")
    save_json(char_dict, 'char_dict.json')
    save_json(word_dict, 'word_dict.json')
    save_json(char_freq, 'char_freq.json')
    save_json(word_freq, 'word_freq.json')
    save_txt(pinyin_table, 'pinyin_table.txt')
    
    print("\n" + "=" * 60)
    print("字典构建完成!")
    print(f"  拼音数: {len(pinyin_table)}")
    print(f"  单字数: {sum(len(v) for v in char_dict.values())}")
    print(f"  词组数: {sum(len(v) for v in word_dict.values())}")
    print("=" * 60)


if __name__ == '__main__':
    main()
