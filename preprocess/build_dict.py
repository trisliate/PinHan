"""
字典构建脚本 v4 - 对话优先 (Dialogue-First)

数据源:
1. SUBTLEX-CH   - 电影字幕词频 (3350万字，最接近口语)
2. CC-CEDICT    - 拼音映射 (权威标准)
3. pypinyin     - 拼音转换 (补充 CEDICT 缺失)

设计理念:
- 彻底抛弃 jieba/THUOCL 的书面语频率
- 词频决定一切: "去" > "区", "和" > "河"
- 字频从词频推断: P(去) = Σ P(含"去"的词)
"""

import gzip
import os
import re
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from pathlib import Path

import pypinyin

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'dicts'

CEDICT_PATH = SCRIPT_DIR / 'cedict.txt.gz'
SUBTLEX_PATH = SCRIPT_DIR / 'SUBTLEX-CH' / 'SUBTLEX-CH-WF'


# ============ 工具函数 ============

def is_chinese(text: str) -> bool:
    """检查是否为纯中文"""
    return all('\u4e00' <= c <= '\u9fff' for c in text)

def word_to_pinyin(word: str) -> List[str]:
    """获取词语的拼音列表 (无声调)"""
    try:
        pys = pypinyin.lazy_pinyin(word, style=pypinyin.Style.NORMAL)
        # 清理: 去除非字母
        result = []
        for py in pys:
            py = py.lower()
            py = re.sub(r'[^a-z]', '', py)
            if py:
                result.append(py)
        return result
    except:
        return []


# ============ 数据加载 ============

def load_subtlex() -> Dict[str, float]:
    """
    加载 SUBTLEX-CH 词频表
    返回: { '我': 50147.83, '你好': 123.45, ... } (每百万词频率)
    """
    print(f"加载 SUBTLEX-CH: {SUBTLEX_PATH}")
    freq_map = {}
    
    if not SUBTLEX_PATH.exists():
        print("⚠ SUBTLEX-CH 未找到！请下载并放置到 preprocess/SUBTLEX-CH/ 目录")
        return freq_map
    
    with open(SUBTLEX_PATH, 'r', encoding='gb2312', errors='ignore') as f:
        lines = f.readlines()
    
    # 跳过前3行 (标题)
    for line in lines[3:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            word = parts[0]
            try:
                # W/million 列 (每百万词频率)
                wpm = float(parts[2])
                if is_chinese(word):
                    freq_map[word] = wpm
            except (ValueError, IndexError):
                pass
    
    print(f"  SUBTLEX-CH 词条: {len(freq_map):,}")
    
    # 显示 Top 10
    top10 = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 10: {[w for w, _ in top10]}")
    
    return freq_map


def load_cedict() -> Dict[str, List[str]]:
    """
    加载 CC-CEDICT
    返回: { 'ni hao': ['你好'], 'qu': ['去', '区', '取', ...], ... }
    """
    print(f"加载 CC-CEDICT: {CEDICT_PATH}")
    mapping = defaultdict(list)
    
    if not CEDICT_PATH.exists():
        print("⚠ CC-CEDICT 未找到！")
        return mapping
    
    with gzip.open(CEDICT_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            match = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]', line)
            if not match:
                continue
            
            simplified = match.group(2)
            pinyin_raw = match.group(3)
            
            # 标准化拼音
            pinyins = []
            for py in pinyin_raw.split():
                py = py.lower()
                py = re.sub(r'[1-5]', '', py)  # 去声调
                py = py.replace('ü', 'v').replace('u:', 'v')
                if py.isalpha():
                    pinyins.append(py)
            
            if pinyins and is_chinese(simplified):
                key = ' '.join(pinyins)
                if simplified not in mapping[key]:
                    mapping[key].append(simplified)
    
    print(f"  CC-CEDICT 拼音组合: {len(mapping):,}")
    return mapping


# ============ 核心构建逻辑 ============

def build_dictionaries():
    """构建所有字典文件"""
    
    # 1. 加载数据源
    subtlex_freq = load_subtlex()
    cedict_map = load_cedict()
    
    if not subtlex_freq:
        print("❌ 无法继续: SUBTLEX-CH 数据缺失")
        return
    
    # 2. 构建词频表 (归一化到概率)
    print("\n构建词频表...")
    total_freq = sum(subtlex_freq.values())
    word_freq = {}
    
    for word, wpm in subtlex_freq.items():
        # wpm = 每百万词出现次数, 转换为概率
        prob = wpm / 1_000_000
        word_freq[word] = prob
    
    print(f"  词频条目: {len(word_freq):,}")
    
    # 3. 构建字频表 (从词频推断)
    # 逻辑: P(字) = Σ P(含该字的词) * 权重
    # 单字词权重更高, 多字词中的字权重递减
    print("构建字频表 (从词频推断)...")
    char_freq_raw = defaultdict(float)
    
    for word, prob in word_freq.items():
        n = len(word)
        for i, char in enumerate(word):
            # 单字词: 权重 1.0
            # 多字词: 权重 1/n (平均分配)
            weight = 1.0 if n == 1 else (1.0 / n)
            char_freq_raw[char] += prob * weight
    
    # 归一化
    total_char = sum(char_freq_raw.values())
    char_freq = {c: p / total_char for c, p in char_freq_raw.items()}
    
    print(f"  字频条目: {len(char_freq):,}")
    
    # 验证: 检查 "去" vs "区"
    qu_chars = [(c, char_freq.get(c, 0)) for c in ['去', '区', '取', '曲']]
    qu_chars.sort(key=lambda x: x[1], reverse=True)
    print(f"  验证 'qu': {qu_chars}")
    
    he_chars = [(c, char_freq.get(c, 0)) for c in ['和', '河', '合', '何']]
    he_chars.sort(key=lambda x: x[1], reverse=True)
    print(f"  验证 'he': {he_chars}")
    
    # 4. 构建词典 (拼音 -> 词/字 列表, 按频率排序)
    print("\n构建词典...")
    
    word_dict = defaultdict(list)
    char_dict = defaultdict(list)
    
    # 4.1 构建 word -> pinyin 映射
    word_to_py = {}
    
    # 从 CEDICT 获取拼音
    for py_key, words in cedict_map.items():
        for w in words:
            if w not in word_to_py:
                word_to_py[w] = py_key
    
    # 用 pypinyin 补充缺失
    for word in word_freq.keys():
        if word not in word_to_py:
            pys = word_to_pinyin(word)
            if pys:
                word_to_py[word] = ' '.join(pys)
    
    # 4.2 填充词典 (按频率排序)
    # 先收集每个拼音下的所有词
    py_to_words = defaultdict(list)
    for word, py_key in word_to_py.items():
        if word in word_freq and len(word) >= 2:
            py_to_words[py_key].append((word, word_freq[word]))
    
    # 排序并填充
    for py_key, items in py_to_words.items():
        items.sort(key=lambda x: x[1], reverse=True)
        word_dict[py_key] = [w for w, _ in items]
    
    # 4.3 填充单字典 (按频率排序)
    py_to_chars = defaultdict(list)
    
    for char, freq in char_freq.items():
        pys = word_to_pinyin(char)
        if pys and len(pys) == 1:
            py_to_chars[pys[0]].append((char, freq))
    
    # 从 CEDICT 补充 (处理多音字)
    for py_key, words in cedict_map.items():
        if ' ' not in py_key:  # 单音节
            for w in words:
                if len(w) == 1:
                    freq = char_freq.get(w, 1e-10)
                    if (w, freq) not in py_to_chars[py_key]:
                        # 检查是否已存在
                        existing = [c for c, _ in py_to_chars[py_key]]
                        if w not in existing:
                            py_to_chars[py_key].append((w, freq))
    
    # 排序并填充
    for py, items in py_to_chars.items():
        items.sort(key=lambda x: x[1], reverse=True)
        char_dict[py] = [c for c, _ in items]
    
    # 5. 补充 CEDICT 中有但 SUBTLEX 没有的词 (作为低频兜底)
    print("补充 CEDICT 词条...")
    added = 0
    for py_key, words in cedict_map.items():
        if ' ' in py_key:  # 多音节词
            for w in words:
                if w not in word_freq:
                    word_freq[w] = 1e-10  # 极低频率
                    if py_key not in word_dict:
                        word_dict[py_key] = []
                    if w not in word_dict[py_key]:
                        word_dict[py_key].append(w)
                        added += 1
    print(f"  补充词条: {added:,}")
    
    # 6. 保存
    print("\n保存字典...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # JSON 保存
    with open(OUTPUT_DIR / 'word_dict.json', 'w', encoding='utf-8') as f:
        json.dump(dict(word_dict), f, ensure_ascii=False)
    
    with open(OUTPUT_DIR / 'char_dict.json', 'w', encoding='utf-8') as f:
        json.dump(dict(char_dict), f, ensure_ascii=False)
    
    with open(OUTPUT_DIR / 'word_freq.json', 'w', encoding='utf-8') as f:
        json.dump(word_freq, f, ensure_ascii=False)
    
    with open(OUTPUT_DIR / 'char_freq.json', 'w', encoding='utf-8') as f:
        json.dump(char_freq, f, ensure_ascii=False)
    
    # 拼音表
    all_pinyins = set(char_dict.keys())
    with open(OUTPUT_DIR / 'pinyin_table.txt', 'w', encoding='utf-8') as f:
        for py in sorted(all_pinyins):
            f.write(py + '\n')
    
    # 7. 验证
    print("\n" + "=" * 50)
    print("验证常用词...")
    
    test_cases = [
        ('qu', '去', 'char'),
        ('he', '和', 'char'),
        ('peng you', '朋友', 'word'),
        ('yi qi', '一起', 'word'),
        ('wo men', '我们', 'word'),
        ('shen me', '什么', 'word'),
    ]
    
    for py, expected, dtype in test_cases:
        if dtype == 'char':
            candidates = char_dict.get(py, [])
        else:
            candidates = word_dict.get(py, [])
        
        if candidates:
            rank = candidates.index(expected) + 1 if expected in candidates else -1
            status = "✓" if rank == 1 else f"✗ (排名#{rank})"
            print(f"  {py} → {expected}: {status}, Top3: {candidates[:3]}")
        else:
            print(f"  {py} → {expected}: ✗ 未找到")
    
    print("=" * 50)
    print(f"构建完成!")
    print(f"  词组数: {len(word_dict):,}")
    print(f"  单字拼音: {len(char_dict):,}")
    print(f"  词频条目: {len(word_freq):,}")
    print(f"  字频条目: {len(char_freq):,}")


if __name__ == '__main__':
    build_dictionaries()
