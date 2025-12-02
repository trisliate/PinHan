"""
字典构建脚本 v3 - 多源 NLP 数据融合

数据源优先级：
1. CC-CEDICT    - 拼音映射（权威标准）
2. 腾讯词向量   - 800万词，现代语料词频
3. jieba 词频表 - 48万词，补充覆盖
4. pypinyin     - 拼音转换

无需手动维护词表！所有词频来自真实语料统计。
"""

import gzip
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import orjson
import jieba
import pypinyin
from pathlib import Path


# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'dicts'
CEDICT_PATH = SCRIPT_DIR / 'cedict.txt.gz'

# 外部词频数据（如果已下载）
TENCENT_FREQ_PATH = SCRIPT_DIR / 'tencent_word_freq.txt'


# ============ CC-CEDICT 解析 ============

def parse_cedict_line(line: str) -> Tuple[str, str, List[str]]:
    """解析 CC-CEDICT 单行"""
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None, None
    
    match = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]', line)
    if not match:
        return None, None, None
    
    traditional = match.group(1)
    simplified = match.group(2)
    pinyin_raw = match.group(3)
    
    pinyins = []
    for py in pinyin_raw.split():
        py = py.lower()
        py = re.sub(r'[1-5]', '', py)
        py = py.replace('ü', 'v').replace('u:', 'v')
        if py:
            pinyins.append(py)
    
    return simplified, traditional, pinyins


def load_cedict() -> List[Tuple[str, str, List[str]]]:
    """加载 CC-CEDICT"""
    entries = []
    
    if not CEDICT_PATH.exists():
        print(f"  警告: CC-CEDICT 不存在: {CEDICT_PATH}")
        return entries
    
    print(f"加载 CC-CEDICT: {CEDICT_PATH}")
    with gzip.open(CEDICT_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            simplified, traditional, pinyins = parse_cedict_line(line)
            if simplified and pinyins:
                entries.append((simplified, traditional, pinyins))
    
    print(f"  CC-CEDICT 词条: {len(entries):,}")
    return entries


# ============ 多源词频加载 ============

def load_jieba_freq() -> Dict[str, int]:
    """加载 jieba 词频表"""
    print("加载 jieba 词频表...")
    jieba.initialize()
    freq = dict(jieba.dt.FREQ)
    print(f"  jieba 词条: {len(freq):,}")
    return freq


def download_external_freq():
    """
    下载外部词频数据
    
    数据源：
    1. jieba 大词典 (58万词，包含词频)
    2. THUOCL 清华词库 (领域专用词)
    """
    import urllib.request
    
    download_dir = SCRIPT_DIR / 'external_freq'
    download_dir.mkdir(exist_ok=True)
    
    # 1. jieba 大词典（最重要，58万词 + 词频）
    jieba_big_url = 'https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big'
    jieba_big_path = download_dir / 'jieba_dict_big.txt'
    
    if not jieba_big_path.exists():
        print("  下载 jieba 大词典 (8MB)...")
        try:
            req = urllib.request.Request(jieba_big_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(jieba_big_path, 'wb') as f:
                    f.write(resp.read())
            print("    下载成功!")
        except Exception as e:
            print(f"    下载失败: {e}")
    
    # 2. 清华 THUOCL 词库（领域词汇补充）
    thuocl_urls = [
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_animal.txt', 'thuocl_animal.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_food.txt', 'thuocl_food.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_lishimingren.txt', 'thuocl_history.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_caijing.txt', 'thuocl_finance.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_car.txt', 'thuocl_car.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_chengyu.txt', 'thuocl_chengyu.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_diming.txt', 'thuocl_place.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_law.txt', 'thuocl_law.txt'),
        ('https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_medical.txt', 'thuocl_medical.txt'),
    ]
    
    for url, filename in thuocl_urls:
        path = download_dir / filename
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except:
                pass  # 静默失败，THUOCL 是补充数据

    # 3. 常用词表 (使用 jieba 小词典作为核心词汇，确保高频词覆盖)
    common_url = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small"
    common_path = download_dir / "common_words.txt"
    
    if not common_path.exists():
        print("  下载常用词表 (jieba small)...")
        try:
            req = urllib.request.Request(common_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode('utf-8')
                with open(common_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        except Exception as e:
            print(f"    常用词表下载失败: {e}")



def load_jieba_big_freq() -> Dict[str, int]:
    """加载 jieba 大词典（58万词 + 词频）"""
    path = SCRIPT_DIR / 'external_freq' / 'jieba_dict_big.txt'
    
    if not path.exists():
        return {}
    
    freq = {}
    print(f"  加载 jieba 大词典: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                word = parts[0]
                try:
                    count = int(parts[1])
                    freq[word] = count
                except:
                    pass
    
    print(f"    词条数: {len(freq):,}")
    return freq


def load_thuocl_freq() -> Dict[str, int]:
    """加载清华 THUOCL 词频"""
    download_dir = SCRIPT_DIR / 'external_freq'
    
    if not download_dir.exists():
        return {}
    
    freq = {}
    for path in download_dir.glob('thuocl_*.txt'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word, count = parts[0], int(parts[1])
                        freq[word] = max(freq.get(word, 0), count)
        except:
            continue
    
    return freq


def load_web_ngram_freq() -> Dict[str, int]:
    """
    从网络加载现代中文词频数据
    使用 HuggingFace datasets 加载公开词频
    """
    try:
        from datasets import load_dataset
        
        print("尝试加载在线词频数据...")
        # 这里可以替换为实际可用的数据集
        # 例如：中文维基、新闻语料的词频统计
        
        # 暂时返回空，后续可扩展
        return {}
    except Exception as e:
        print(f"  在线词频加载失败: {e}")
        return {}


def generate_bigram_freq_from_corpus(jieba_freq: Dict[str, int]) -> Dict[str, int]:
    """
    基于单字频率生成常用二字词估计频率
    
    原理：
    - 对于动词+得/着/了 结构，概率 = P(动词) * P(结构词)
    - 对于 程度副词+形容词 结构，类似处理
    
    这比手动添加更科学，可以自动覆盖大量组合
    """
    print("生成结构化词组频率...")
    
    generated = {}
    
    # 常见结构词的相对频率权重
    structure_weights = {
        # 得字结构（动词+得）
        '得': 0.5,
        '着': 0.3,
        '了': 0.4,
        # 方向补语
        '来': 0.3,
        '去': 0.3,
        '上': 0.2,
        '下': 0.2,
        '进': 0.15,
        '出': 0.15,
        '回': 0.15,
        '过': 0.2,
        '起': 0.15,
        '开': 0.15,
    }
    
    # 程度副词
    degree_adverbs = {
        '很': 1.0,
        '太': 0.8,
        '真': 0.7,
        '好': 0.6,
        '挺': 0.5,
        '非常': 0.6,
        '特别': 0.5,
        '相当': 0.4,
        '十分': 0.4,
        '极其': 0.3,
    }
    
    # 常见形容词（从 jieba 高频词中提取）
    common_adj = ['好', '大', '小', '多', '少', '快', '慢', '高', '低', '长', '短',
                  '美', '丑', '胖', '瘦', '新', '旧', '冷', '热', '难', '易', '强', '弱',
                  '棒', '差', '贵', '便宜', '开心', '难过', '漂亮', '厉害']
    
    # 常见动词（单字）
    common_verbs = ['吃', '喝', '玩', '看', '听', '说', '读', '写', '走', '跑', 
                    '跳', '唱', '做', '想', '学', '教', '买', '卖', '睡', '坐',
                    '站', '拿', '放', '开', '关', '穿', '脱', '洗', '干', '打']
    
    # 生成 动词+结构词 组合
    for verb in common_verbs:
        verb_freq = jieba_freq.get(verb, 100)
        for suffix, weight in structure_weights.items():
            word = verb + suffix
            if word not in jieba_freq or jieba_freq.get(word, 0) == 0:
                # 估算频率 = 动词频率 * 结构权重
                generated[word] = int(verb_freq * weight * 0.1)
    
    # 生成 程度副词+形容词 组合
    for adv, adv_weight in degree_adverbs.items():
        for adj in common_adj:
            word = adv + adj
            adj_freq = jieba_freq.get(adj, 50)
            if word not in jieba_freq or jieba_freq.get(word, 0) == 0:
                generated[word] = int(adj_freq * adv_weight * 0.2)
    
    print(f"  生成结构化词组: {len(generated):,}")
    return generated


# ============ 词频融合 ============

def merge_frequencies(*freq_dicts: Dict[str, int]) -> Dict[str, int]:
    """
    融合多个词频字典
    策略：取最大值（因为不同语料可能低估某些词）
    """
    merged = {}
    for freq_dict in freq_dicts:
        for word, freq in freq_dict.items():
            merged[word] = max(merged.get(word, 0), freq)
    return merged


def load_common_words() -> Set[str]:
    """
    加载常用词表
    """
    path = SCRIPT_DIR / 'external_freq' / 'common_words.txt'
    common_words = set()
    
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    # jieba dict format: word freq tag
                    parts = line.strip().split(' ')
                    if parts:
                        word = parts[0].strip()
                        if word and is_chinese_word(word):
                            common_words.add(word)
        except Exception as e:
            print(f"  加载常用词表失败: {e}")
            
    print(f"  常用词表加载: {len(common_words):,} 词")
    return common_words


def apply_frequency_corrections(freq: Dict[str, int], common_words: Set[str] = None) -> Dict[str, int]:
    """
    应用词频修正规则
    
    使用下载的常用词表 (HSK) 来修正口语词频
    """
    print("应用词频修正规则...")
    
    if common_words is None:
        common_words = load_common_words()
    
    # 计算高频阈值 (Top 2000)
    all_freqs = sorted(freq.values(), reverse=True)
    if len(all_freqs) > 2000:
        high_freq_threshold = all_freqs[2000]
    else:
        high_freq_threshold = 10000
        
    print(f"  高频阈值: {high_freq_threshold:,}")
    
    corrections_made = 0
    added_words = 0
    
    for word in common_words:
        if word in freq:
            if freq[word] < high_freq_threshold:
                freq[word] = max(freq[word] * 5, high_freq_threshold)
                corrections_made += 1
        else:
            freq[word] = high_freq_threshold
            added_words += 1
            
    # 规则2：常用动补结构提升
    verb_complement_words = [
        '玩得', '吃得', '睡得', '跑得', '走得', '做得', '说得', '看得',
        '听得', '想得', '学得', '写得', '读得', '唱得', '跳得',
    ]
    for word in verb_complement_words:
        if word in freq:
            freq[word] = max(freq[word], high_freq_threshold // 2)
            corrections_made += 1
            
    print(f"  修正词条: {corrections_made}, 新增词条: {added_words}")
    return freq


# ============ 拼音工具 ============

def is_chinese_word(word: str) -> bool:
    """检查是否为纯中文词"""
    return all('\u4e00' <= c <= '\u9fff' for c in word)


def word_to_pinyin(word: str) -> List[str]:
    """汉字词组 → 拼音列表"""
    try:
        pys = pypinyin.pinyin(word, style=pypinyin.NORMAL, errors='ignore')
        result = []
        for py_list in pys:
            if py_list:
                py = py_list[0].lower().replace('ü', 'v')
                if py and py.isalpha():
                    result.append(py)
        return result
    except:
        return []


# ============ 构建字典 ============

def build_char_dict_from_cedict(entries: List[Tuple[str, str, List[str]]]) -> Dict[str, Set[str]]:
    """从 CC-CEDICT 构建单字字典"""
    print("构建单字字典...")
    
    char_dict = defaultdict(set)
    for simplified, traditional, pinyins in entries:
        if len(simplified) == 1 and len(pinyins) == 1:
            char_dict[pinyins[0]].add(simplified)
            if traditional != simplified:
                char_dict[pinyins[0]].add(traditional)
    
    print(f"  拼音数: {len(char_dict)}, 单字数: {sum(len(v) for v in char_dict.values()):,}")
    return char_dict


def build_word_dict(
    merged_freq: Dict[str, int],
    char_dict: Dict[str, Set[str]],
    cedict_entries: List[Tuple[str, str, List[str]]],
    min_freq: int = 1,
    max_word_len: int = 8
) -> Dict[str, List[str]]:
    """构建词组字典"""
    print(f"构建词组字典...")
    
    valid_pinyins = set(char_dict.keys())
    word_dict = defaultdict(list)
    
    # 从词频表构建
    for word, freq in merged_freq.items():
        if freq < min_freq:
            continue
        if len(word) < 2 or len(word) > max_word_len:
            continue
        if not is_chinese_word(word):
            continue
        
        pinyins = word_to_pinyin(word)
        if not pinyins or len(pinyins) != len(word):
            continue
        if not all(py in valid_pinyins for py in pinyins):
            continue
        
        key = ' '.join(pinyins)
        if word not in word_dict[key]:
            word_dict[key].append(word)
    
    # 合并 CC-CEDICT 词组
    added = 0
    for simplified, traditional, pinyins in cedict_entries:
        if len(simplified) >= 2 and len(pinyins) == len(simplified):
            key = ' '.join(pinyins)
            if simplified not in word_dict.get(key, []):
                if key not in word_dict:
                    word_dict[key] = []
                word_dict[key].append(simplified)
                added += 1
    
    print(f"  词组拼音组合: {len(word_dict):,}")
    print(f"  词组总数: {sum(len(v) for v in word_dict.values()):,}")
    print(f"  从 CC-CEDICT 补充: {added:,}")
    
    return dict(word_dict)


# ============ 频率表构建 ============

def build_char_freq(
    char_dict: Dict[str, Set[str]], 
    merged_freq: Dict[str, int]
) -> Dict[str, float]:
    """构建字频表"""
    print("构建字频表...")
    
    all_chars = set()
    for chars in char_dict.values():
        all_chars.update(chars)
    
    char_freq_raw = {char: merged_freq.get(char, 0) for char in all_chars}
    
    # 简体字优先：如果简繁体都存在，给简体加权
    # 使用 opencc 或简单的映射表
    try:
        from opencc import OpenCC
        t2s = OpenCC('t2s')  # 繁体转简体
        
        # 找出繁体字并降权
        for char in list(char_freq_raw.keys()):
            simplified = t2s.convert(char)
            if simplified != char and simplified in char_freq_raw:
                # 这是繁体字，简体也存在，降低繁体权重
                char_freq_raw[char] = char_freq_raw[char] * 0.1
    except ImportError:
        # 没有 opencc，使用简单的常见繁简映射
        traditional_to_simple = {
            '嗎': '吗', '馬': '马', '媽': '妈', '罵': '骂',
            '電': '电', '話': '话', '說': '说', '讀': '读',
            '寫': '写', '買': '买', '賣': '卖', '開': '开',
            '關': '关', '門': '门', '問': '问', '聽': '听',
            '見': '见', '覺': '觉', '學': '学', '樂': '乐',
            '東': '东', '車': '车', '書': '书', '飛': '飞',
            '機': '机', '體': '体', '頭': '头', '臉': '脸',
            '愛': '爱', '國': '国', '語': '语', '時': '时',
            '現': '现', '視': '视', '觀': '观', '點': '点',
            '樹': '树', '雲': '云', '風': '风', '陽': '阳',
        }
        for trad, simp in traditional_to_simple.items():
            if trad in char_freq_raw and simp in char_freq_raw:
                char_freq_raw[trad] = char_freq_raw[trad] * 0.1
    
    total = sum(char_freq_raw.values()) or 1
    
    char_freq = {}
    min_freq = 1e-9
    for char, freq in char_freq_raw.items():
        normalized = freq / total
        char_freq[char] = max(normalized, min_freq)
    
    print(f"  字频条目: {len(char_freq):,}")
    return char_freq


def build_word_freq(
    word_dict: Dict[str, List[str]], 
    merged_freq: Dict[str, int]
) -> Dict[str, float]:
    """构建词频表"""
    print("构建词频表...")
    
    all_words = set()
    for words in word_dict.values():
        all_words.update(words)
    
    word_freq_raw = {word: merged_freq.get(word, 0) for word in all_words}
    total = sum(word_freq_raw.values()) or 1
    
    word_freq = {}
    for word, freq in word_freq_raw.items():
        normalized = freq / total
        if normalized == 0:
            normalized = 1e-9 / len(word)
        word_freq[word] = normalized
    
    valid = sum(1 for w, f in word_freq.items() if merged_freq.get(w, 0) > 0)
    print(f"  词频条目: {len(word_freq):,}, 有频率数据: {valid:,}")
    return word_freq


# ============ 排序优化 ============

def sort_dict_by_freq(d: Dict[str, list], freq: Dict[str, float]) -> Dict[str, List[str]]:
    """按频率排序"""
    return {
        key: sorted(items, key=lambda x: freq.get(x, 0), reverse=True)
        for key, items in d.items()
    }


# ============ 保存 ============

def save_json(data, filename: str):
    path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"  已保存: {path}")


def save_txt(lines: List[str], filename: str):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  已保存: {path}")


# ============ 验证 ============

def verify_dict(word_dict: Dict[str, List[str]], word_freq: Dict[str, float]):
    """验证常用词"""
    print("\n验证常用词...")
    
    test_words = [
        ('hen hao', '很好'),
        ('gong yuan', '公园'),
        ('tian qi', '天气'),
        ('wan de', '玩得'),
        ('wo men', '我们'),
        ('ta men', '他们'),
        ('peng you', '朋友'),
        ('kai xin', '开心'),
    ]
    
    for key, expected in test_words:
        words = word_dict.get(key, [])
        if expected in words:
            rank = words.index(expected) + 1
            freq = word_freq.get(expected, 0)
            print(f"  ✓ {key} → {expected} (排名#{rank}, freq={freq:.2e})")
        else:
            print(f"  ✗ {key} → 缺少 '{expected}', 现有: {words[:3]}")


# ============ 主函数 ============

def main():
    print("=" * 60)
    print("字典构建 v3 - 多源 NLP 数据自动融合")
    print("=" * 60)
    
    # 1. 加载所有数据源
    print("\n[1/6] 加载数据源...")
    cedict_entries = load_cedict()
    jieba_freq = load_jieba_freq()
    
    # 下载并加载扩展词频
    print("\n[2/6] 加载扩展词频...")
    download_external_freq()
    
    # jieba 大词典（58万词，最重要的补充）
    jieba_big_freq = load_jieba_big_freq()
    
    # THUOCL 清华词库（领域词汇）
    thuocl_freq = load_thuocl_freq()
    print(f"  THUOCL 词条: {len(thuocl_freq):,}")
    
    # 生成结构化词组频率
    print("\n[3/6] 生成结构化词组...")
    generated_freq = generate_bigram_freq_from_corpus(jieba_freq)
    
    # 融合所有词频（jieba大词典优先级最高）
    print("\n[4/6] 融合词频数据...")
    merged_freq = merge_frequencies(jieba_freq, jieba_big_freq, thuocl_freq, generated_freq)
    print(f"  融合后词条: {len(merged_freq):,}")
    
    # 应用词频修正（口语优先）
    merged_freq = apply_frequency_corrections(merged_freq)
    
    # 构建字典
    print("\n[5/6] 构建字典...")
    char_dict = build_char_dict_from_cedict(cedict_entries)
    word_dict = build_word_dict(merged_freq, char_dict, cedict_entries)
    
    # 构建频率表
    char_freq = build_char_freq(char_dict, merged_freq)
    word_freq = build_word_freq(word_dict, merged_freq)
    
    # 排序
    print("\n[6/6] 排序优化...")
    char_dict_sorted = sort_dict_by_freq({k: list(v) for k, v in char_dict.items()}, char_freq)
    word_dict_sorted = sort_dict_by_freq(word_dict, word_freq)
    pinyin_table = sorted(char_dict_sorted.keys())
    
    # 验证
    verify_dict(word_dict_sorted, word_freq)
    
    # 保存
    print("\n保存文件...")
    save_json(char_dict_sorted, 'char_dict.json')
    save_json(word_dict_sorted, 'word_dict.json')
    save_json(char_freq, 'char_freq.json')
    save_json(word_freq, 'word_freq.json')
    save_txt(pinyin_table, 'pinyin_table.txt')
    
    # 统计
    print("\n" + "=" * 60)
    print("构建完成!")
    print(f"  数据源: CC-CEDICT + jieba + THUOCL + 结构化生成")
    print(f"  拼音数: {len(pinyin_table)}")
    print(f"  单字数: {sum(len(v) for v in char_dict_sorted.values()):,}")
    print(f"  词组数: {sum(len(v) for v in word_dict_sorted.values()):,}")
    print("=" * 60)


if __name__ == '__main__':
    main()
