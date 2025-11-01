"""
拼音规范化和处理工具模块。

功能：
- 拼音标准化（tone mark -> 数字形式 | tone number -> 标准形式）
- 音调提取和拼音分离
- 拼音验证和清理
- 多音字识别（如果相同汉字有多个拼音）
"""
import re
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


# 声调标记符号到数字的映射（用于 tone mark 转 tone number）
# 映射格式：标记字符 -> (字母, 音调号)
TONE_MARK_TO_NUM = {
    'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'),
    'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'),
    'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'),
    'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'),
    'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'),
    'ǖ': ('v', '1'), 'ǘ': ('v', '2'), 'ǚ': ('v', '3'), 'ǜ': ('v', '4'),
    'ń': ('n', '2'), 'ň': ('n', '3'), 'ǹ': ('n', '4'),
    'ḿ': ('m', '2'),
}

# 常见汉字的多音字映射（可扩展）
COMMON_POLYPHONIC_CHARS = {
    '中': ['zhong1', 'zhong4'],
    '长': ['chang2', 'zhang3'],
    '还': ['hai2', 'huan2'],
    '行': ['xing2', 'hang2'],
    '度': ['du4', 'duo2'],
    '重': ['zhong4', 'chong2'],
    '为': ['wei2', 'wei4'],
    '了': ['le', 'liao3'],
    '着': ['zhe', 'zhao2', 'zhu2'],
}


def tone_mark_to_number(pinyin_with_mark: str) -> str:
    """
    将带声调标记的拼音转换为数字音调形式。
    
    例：
        'mā' -> 'ma1'
        'hǎo' -> 'hao3'
    
    Args:
        pinyin_with_mark: 带声调标记的拼音（如 'mā'）
    
    Returns:
        数字音调形式的拼音（如 'ma1'）
        如果无法转换，返回原字符串
    """
    if not pinyin_with_mark:
        return ''
    
    result = []
    tone = ''
    for char in pinyin_with_mark:
        if char in TONE_MARK_TO_NUM:
            letter, tone_num = TONE_MARK_TO_NUM[char]
            result.append(letter)
            tone = tone_num
        else:
            result.append(char)
    
    # 将音调附加到最后
    return ''.join(result) + tone


def normalize_pinyin(pinyin: str) -> str:
    """
    规范化单个拼音。
    
    - 去除首尾空格
    - 转为小写
    - 将 tone mark 转为 tone number（如果存在）
    - 处理 ü -> v 的兼容形式
    
    Args:
        pinyin: 原始拼音字符串
    
    Returns:
        规范化后的拼音
    """
    pinyin = pinyin.strip().lower()
    
    # 转换 tone mark 到 tone number
    pinyin = tone_mark_to_number(pinyin)
    
    # 处理 ü 的多种表示法：ü, u:, v
    pinyin = pinyin.replace('ü', 'v').replace('u:', 'v')
    
    return pinyin


def extract_tone(pinyin: str) -> Tuple[str, Optional[str]]:
    """
    从拼音中提取音调。
    
    Args:
        pinyin: 标准化的拼音（如 'ma1'）
    
    Returns:
        (拼音不含音调部分, 音调号)，例如 ('ma', '1')
        如果没有音调则返回 (拼音, None)
    """
    match = re.match(r'^([a-z\d]*?)(\d?)$', pinyin)
    if match:
        base, tone = match.groups()
        return base, tone if tone else None
    return pinyin, None


def normalize_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> str:
    """
    规范化拼音序列。
    
    Args:
        pinyin_str: 拼音序列字符串（用分隔符分开的多个拼音）
        separator: 拼音分隔符，默认为空格
    
    Returns:
        规范化后的拼音序列（同样用分隔符分开）
    """
    pinyins = pinyin_str.split(separator)
    normalized = [normalize_pinyin(p) for p in pinyins if p.strip()]
    return separator.join(normalized)


def validate_pinyin(pinyin: str) -> bool:
    """
    验证拼音是否有效。
    
    有效拼音应该：
    - 包含至少一个字母
    - 可能包含数字音调（0-4）
    - 不应该包含其他特殊字符
    
    Args:
        pinyin: 规范化后的拼音
    
    Returns:
        是否为有效拼音
    """
    # 允许：a-z, v (代表ü), 数字 0-4
    pattern = r'^[a-z\d]+$'
    if not re.match(pattern, pinyin):
        return False
    
    # 检查是否至少有一个字母
    if not any(c.isalpha() for c in pinyin):
        return False
    
    return True


def validate_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> bool:
    """
    验证拼音序列是否有效。
    
    Args:
        pinyin_str: 拼音序列
        separator: 拼音分隔符
    
    Returns:
        序列中所有拼音是否都有效
    """
    pinyins = pinyin_str.split(separator)
    return all(validate_pinyin(p.strip()) for p in pinyins if p.strip())


def split_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> List[str]:
    """
    拆分拼音序列为单个拼音列表。
    
    Args:
        pinyin_str: 拼音序列
        separator: 拼音分隔符
    
    Returns:
        拼音列表
    """
    return [p.strip() for p in pinyin_str.split(separator) if p.strip()]


def join_pinyin_sequence(pinyins: List[str], separator: str = ' ') -> str:
    """
    将拼音列表合并为序列字符串。
    
    Args:
        pinyins: 拼音列表
        separator: 拼音分隔符
    
    Returns:
        拼音序列字符串
    """
    return separator.join(pinyins)


def is_polyphonic_char(hanzi: str) -> bool:
    """
    判断汉字是否为常见多音字。
    
    Args:
        hanzi: 单个汉字
    
    Returns:
        是否为多音字
    """
    return hanzi in COMMON_POLYPHONIC_CHARS


def get_possible_pinyins(hanzi: str) -> Optional[List[str]]:
    """
    获取多音字的可能拼音列表。
    
    Args:
        hanzi: 单个汉字
    
    Returns:
        拼音列表，如果不是多音字返回 None
    """
    return COMMON_POLYPHONIC_CHARS.get(hanzi)


class PinyinStatistics:
    """统计拼音和汉字的分布情况。"""
    
    def __init__(self):
        self.pinyin_freq = defaultdict(int)
        self.hanzi_freq = defaultdict(int)
        self.pinyin_hanzi_pairs = defaultdict(set)  # pinyin -> set of hanzi
        self.hanzi_pinyins = defaultdict(set)  # hanzi -> set of pinyin
        self.total_pairs = 0
    
    def update_from_data(self, hanzi_str: str, pinyin_str: str, separator: str = ' '):
        """
        从数据对更新统计信息。
        
        Args:
            hanzi_str: 汉字字符串
            pinyin_str: 拼音序列
            separator: 拼音分隔符
        """
        pinyins = split_pinyin_sequence(pinyin_str, separator)
        hanzis = list(hanzi_str)
        
        # 按字符逐一关联
        for i, (hanzi, pinyin) in enumerate(zip(hanzis, pinyins)):
            self.pinyin_freq[pinyin] += 1
            self.hanzi_freq[hanzi] += 1
            self.pinyin_hanzi_pairs[pinyin].add(hanzi)
            self.hanzi_pinyins[hanzi].add(pinyin)
            self.total_pairs += 1
    
    def get_pinyin_frequency(self, pinyin: str) -> int:
        """获取拼音出现频率。"""
        return self.pinyin_freq[pinyin]
    
    def get_hanzi_frequency(self, hanzi: str) -> int:
        """获取汉字出现频率。"""
        return self.hanzi_freq[hanzi]
    
    def get_polyphonic_hanzis(self) -> Dict[str, set]:
        """获取有多个拼音的汉字（多音字）。"""
        return {h: p for h, p in self.hanzi_pinyins.items() if len(p) > 1}
    
    def get_homophonic_hanzis(self) -> Dict[str, set]:
        """获取多个汉字共享同一个拼音的情况（同音字）。"""
        return {p: h for p, h in self.pinyin_hanzi_pairs.items() if len(h) > 1}
    
    def print_summary(self):
        """打印统计摘要。"""
        polyphonic = self.get_polyphonic_hanzis()
        homophonic = self.get_homophonic_hanzis()
        print(f"总拼音-汉字对数: {self.total_pairs}")
        print(f"唯一拼音数: {len(self.pinyin_freq)}")
        print(f"唯一汉字数: {len(self.hanzi_freq)}")
        print(f"多音字数: {len(polyphonic)}")
        print(f"同音字组数: {len(homophonic)}")
        
        # 打印多音字示例
        if polyphonic:
            print("\n多音字示例（前10个）:")
            for hanzi, pinyins in list(polyphonic.items())[:10]:
                print(f"  {hanzi}: {', '.join(sorted(pinyins))}")


if __name__ == '__main__':
    # 测试示例
    print("=== 拼音规范化测试 ===")
    test_cases = [
        'ma1',
        'hǎo',
        'BEIJING',
        'zhōng',
    ]
    for case in test_cases:
        normalized = normalize_pinyin(case)
        print(f"{case} -> {normalized}")
    
    print("\n=== 拼音提取测试 ===")
    test_pinyins = ['ma1', 'hao', 'zhang3']
    for p in test_pinyins:
        base, tone = extract_tone(p)
        print(f"{p} -> base='{base}', tone='{tone}'")
    
    print("\n=== 拼音验证测试 ===")
    test_validations = ['ma1', 'h3o', 'xyz', 'ma', 'ma5']
    for p in test_validations:
        is_valid = validate_pinyin(p)
        print(f"{p} -> valid={is_valid}")
    
    print("\n=== 多音字检查 ===")
    test_hanzis = ['中', '长', '行', '好']
    for h in test_hanzis:
        is_poly = is_polyphonic_char(h)
        pinyins = get_possible_pinyins(h)
        print(f"{h} -> polyphonic={is_poly}, pinyins={pinyins}")
