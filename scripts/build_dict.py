"""
字典构建脚本 v4 - 可扩展词频融合架构

数据源 (按优先级):
1. 用户自定义词频 (patches/*.txt) - 最高优先级，用于修正
2. SUBTLEX-CH - 电影字幕词频 (口语核心)
3. 扩展词库 (sources/*.txt) - 补充覆盖
4. CC-CEDICT - 拼音映射 (兜底)

扩展方法:
- 在 data/patches/ 下添加 .txt 文件 (格式: 词语 频率)
- 在 data/sources/ 下添加 .txt 文件 (格式: 词语 频率)
"""

import gzip
import re
import orjson
from collections import defaultdict
from typing import Dict, List, Callable
from pathlib import Path
from abc import ABC, abstractmethod

import pypinyin

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = DATA_DIR / 'dicts'

CEDICT_PATH = DATA_DIR / 'sources' / 'cedict.txt.gz'
SUBTLEX_PATH = DATA_DIR / 'sources' / 'SUBTLEX-CH' / 'SUBTLEX-CH-WF'
PATCHES_DIR = DATA_DIR / 'patches'   # 高优先级修正
SOURCES_DIR = DATA_DIR / 'sources'   # 扩展词库


# ============ 工具函数 ============

def is_chinese(text: str) -> bool:
    """检查是否为纯中文"""
    return all('\u4e00' <= c <= '\u9fff' for c in text)

def word_to_pinyin(word: str) -> List[str]:
    """获取词语的拼音列表 (无声调)"""
    try:
        pys = pypinyin.lazy_pinyin(word, style=pypinyin.Style.NORMAL)
        result = []
        for py in pys:
            py = py.lower()
            py = re.sub(r'[^a-z]', '', py)
            if py:
                result.append(py)
        return result
    except:
        return []


# ============ 词频源抽象 ============

class FreqSource(ABC):
    """词频数据源抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """优先级 (越高越优先)"""
        pass
    
    @abstractmethod
    def load(self) -> Dict[str, float]:
        """加载词频数据, 返回 {词: 频率}"""
        pass


class SubtlexSource(FreqSource):
    """SUBTLEX-CH 电影字幕词频"""
    
    @property
    def name(self) -> str:
        return "SUBTLEX-CH"
    
    @property
    def priority(self) -> int:
        return 50  # 中等优先级
    
    def load(self) -> Dict[str, float]:
        print(f"  加载 {self.name}...")
        freq_map = {}
        
        if not SUBTLEX_PATH.exists():
            print(f"    ⚠ {SUBTLEX_PATH} 未找到")
            return freq_map
        
        with open(SUBTLEX_PATH, 'r', encoding='gb2312', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines[3:]:  # 跳过前3行标题
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word = parts[0]
                try:
                    wpm = float(parts[2])  # W/million
                    if is_chinese(word):
                        freq_map[word] = wpm
                except (ValueError, IndexError):
                    pass
        
        print(f"    词条: {len(freq_map):,}")
        return freq_map


class TextFileSource(FreqSource):
    """通用文本文件词频源"""
    
    def __init__(self, path: Path, source_name: str, source_priority: int):
        self.path = path
        self._name = source_name
        self._priority = source_priority
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def priority(self) -> int:
        return self._priority
    
    def load(self) -> Dict[str, float]:
        print(f"  加载 {self.name}: {self.path.name}")
        freq_map = {}
        
        if not self.path.exists():
            return freq_map
        
        # 尝试多种编码
        for encoding in ['utf-8', 'gb2312', 'gbk']:
            try:
                with open(self.path, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            word = parts[0]
                            try:
                                freq = float(parts[1])
                                if is_chinese(word):
                                    freq_map[word] = freq
                            except ValueError:
                                pass
                break
            except UnicodeDecodeError:
                continue
        
        print(f"    词条: {len(freq_map):,}")
        return freq_map


# ============ 词频融合器 ============

class FreqMerger:
    """词频融合器 - 管理多个数据源"""
    
    def __init__(self):
        self.sources: List[FreqSource] = []
    
    def add_source(self, source: FreqSource):
        """添加数据源"""
        self.sources.append(source)
    
    def merge(self) -> Dict[str, float]:
        """
        融合所有数据源
        策略: 高优先级覆盖低优先级
        """
        print("\n融合词频数据源...")
        
        # 按优先级排序 (低优先级先加载, 高优先级后覆盖)
        sorted_sources = sorted(self.sources, key=lambda s: s.priority)
        
        merged = {}
        for source in sorted_sources:
            data = source.load()
            # 高优先级直接覆盖
            for word, freq in data.items():
                merged[word] = freq  # 后加载的覆盖先加载的
        
        print(f"  融合后词条: {len(merged):,}")
        return merged


# ============ CC-CEDICT 加载 ============

def load_cedict() -> Dict[str, List[str]]:
    """加载 CC-CEDICT 拼音映射"""
    print(f"加载 CC-CEDICT: {CEDICT_PATH}")
    mapping = defaultdict(list)
    
    if not CEDICT_PATH.exists():
        print("  ⚠ CC-CEDICT 未找到！")
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
            
            pinyins = []
            for py in pinyin_raw.split():
                py = py.lower()
                py = re.sub(r'[1-5]', '', py)
                py = py.replace('ü', 'v').replace('u:', 'v')
                if py.isalpha():
                    pinyins.append(py)
            
            if pinyins and is_chinese(simplified):
                key = ' '.join(pinyins)
                if simplified not in mapping[key]:
                    mapping[key].append(simplified)
    
    print(f"  拼音组合: {len(mapping):,}")
    return mapping


# ============ 自动发现数据源 ============

def discover_sources() -> List[FreqSource]:
    """自动发现所有数据源"""
    sources = []
    
    # 1. SUBTLEX-CH (核心)
    sources.append(SubtlexSource())
    
    # 2. 扩展词库 (sources/*.txt) - 优先级 30
    if SOURCES_DIR.exists():
        for path in SOURCES_DIR.glob('*.txt'):
            sources.append(TextFileSource(path, f"扩展/{path.stem}", 30))
    
    # 3. 补丁词库 (patches/*.txt) - 优先级 100 (最高)
    if PATCHES_DIR.exists():
        for path in PATCHES_DIR.glob('*.txt'):
            sources.append(TextFileSource(path, f"补丁/{path.stem}", 100))
    
    return sources


# ============ 核心构建逻辑 ============

def build_dictionaries():
    """构建所有字典文件"""
    
    # 1. 发现并融合词频
    merger = FreqMerger()
    for source in discover_sources():
        merger.add_source(source)
    
    raw_freq = merger.merge()
    
    if not raw_freq:
        print("❌ 无法继续: 没有可用的词频数据")
        return
    
    # 2. 加载拼音映射
    cedict_map = load_cedict()
    
    # 3. 归一化词频
    print("\n构建词频表...")
    total_freq = sum(raw_freq.values())
    word_freq = {}
    for word, freq in raw_freq.items():
        word_freq[word] = freq / total_freq
    print(f"  词频条目: {len(word_freq):,}")
    
    # 4. 构建字频 (从词频推断)
    print("构建字频表...")
    char_freq_raw = defaultdict(float)
    for word, prob in word_freq.items():
        n = len(word)
        for char in word:
            weight = 1.0 if n == 1 else (1.0 / n)
            char_freq_raw[char] += prob * weight
    
    total_char = sum(char_freq_raw.values())
    char_freq = {c: p / total_char for c, p in char_freq_raw.items()}
    print(f"  字频条目: {len(char_freq):,}")
    
    # 5. 构建词典
    print("\n构建词典...")
    word_dict = defaultdict(list)
    char_dict = defaultdict(list)
    
    # word -> pinyin 映射
    word_to_py = {}
    for py_key, words in cedict_map.items():
        for w in words:
            if w not in word_to_py:
                word_to_py[w] = py_key
    
    for word in word_freq.keys():
        if word not in word_to_py:
            pys = word_to_pinyin(word)
            if pys:
                word_to_py[word] = ' '.join(pys)
    
    # 填充词典
    py_to_words = defaultdict(list)
    for word, py_key in word_to_py.items():
        if word in word_freq and len(word) >= 2:
            py_to_words[py_key].append((word, word_freq[word]))
    
    for py_key, items in py_to_words.items():
        items.sort(key=lambda x: x[1], reverse=True)
        word_dict[py_key] = [w for w, _ in items]
    
    # 填充单字典
    py_to_chars = defaultdict(list)
    for char, freq in char_freq.items():
        pys = word_to_pinyin(char)
        if pys and len(pys) == 1:
            py_to_chars[pys[0]].append((char, freq))
    
    for py_key, words in cedict_map.items():
        if ' ' not in py_key:
            for w in words:
                if len(w) == 1:
                    freq = char_freq.get(w, 1e-10)
                    existing = [c for c, _ in py_to_chars[py_key]]
                    if w not in existing:
                        py_to_chars[py_key].append((w, freq))
    
    for py, items in py_to_chars.items():
        items.sort(key=lambda x: x[1], reverse=True)
        char_dict[py] = [c for c, _ in items]
    
    # 6. 补充 CEDICT
    print("补充 CEDICT 词条...")
    added = 0
    for py_key, words in cedict_map.items():
        if ' ' in py_key:
            for w in words:
                if w not in word_freq:
                    word_freq[w] = 1e-10
                    if w not in word_dict.get(py_key, []):
                        if py_key not in word_dict:
                            word_dict[py_key] = []
                        word_dict[py_key].append(w)
                        added += 1
    print(f"  补充词条: {added:,}")
    
    # 7. 保存
    print("\n保存字典...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    (OUTPUT_DIR / 'word_dict.json').write_bytes(orjson.dumps(dict(word_dict)))
    (OUTPUT_DIR / 'char_dict.json').write_bytes(orjson.dumps(dict(char_dict)))
    (OUTPUT_DIR / 'word_freq.json').write_bytes(orjson.dumps(word_freq))
    (OUTPUT_DIR / 'char_freq.json').write_bytes(orjson.dumps(char_freq))
    
    all_pinyins = set(char_dict.keys())
    with open(OUTPUT_DIR / 'pinyin_table.txt', 'w', encoding='utf-8') as f:
        for py in sorted(all_pinyins):
            f.write(py + '\n')
    
    # 8. 验证
    print("\n" + "=" * 50)
    print("验证常用词...")
    test_cases = [
        ('qu', '去', 'char'),
        ('he', '和', 'char'),
        ('ji', '几', 'char'),
        ('qian', '钱', 'char'),
        ('peng you', '朋友', 'word'),
        ('yi qi', '一起', 'word'),
        ('ji dian', '几点', 'word'),
    ]
    
    for py, expected, dtype in test_cases:
        candidates = char_dict.get(py, []) if dtype == 'char' else word_dict.get(py, [])
        if candidates:
            rank = candidates.index(expected) + 1 if expected in candidates else -1
            status = "✓" if rank == 1 else f"✗ (#{rank})"
            print(f"  {py} → {expected}: {status}, Top3: {candidates[:3]}")
        else:
            print(f"  {py} → {expected}: ✗ 未找到")
    
    print("=" * 50)
    print(f"构建完成!")
    print(f"  词组数: {len(word_dict):,}")
    print(f"  单字拼音: {len(char_dict):,}")


if __name__ == '__main__':
    build_dictionaries()
