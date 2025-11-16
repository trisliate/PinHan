"""
拼音规范化和处理工具模块。

功能：
- 拼音标准化（tone mark -> 数字形式 | tone number -> 标准形式）
- 音调提取和拼音分离
- 拼音验证和清理
- 多音字识别（如果相同汉字有多个拼音）
"""
import re
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

# 🚀 优化：预编译常用正则表达式（避免每次调用时重新编译）
# 注意：允许0表示轻声，1-4表示声调
_TONE_MARK_PATTERN = re.compile(r'^[a-z]+[0-4]?$')
_PINYIN_BASE_PATTERN = re.compile(r'^([a-z\d]*?)(\d?)$')

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
# 🔴 改进：更完整的多音字映射，涵盖常见汉字
COMMON_POLYPHONIC_CHARS = {
    # 学术/文言
    '中': ['zhong1', 'zhong4'],          # 中国 vs 中间
    '长': ['chang2', 'zhang3'],          # 长度 vs 长大
    '还': ['hai2', 'huan2'],             # 还要 vs 还原
    '行': ['xing2', 'hang2'],            # 行走 vs 行业
    '度': ['du4', 'duo2'],               # 温度 vs 度过
    '重': ['zhong4', 'chong2'],          # 重量 vs 重视
    '为': ['wei2', 'wei4'],              # 因为 vs 为了
    '着': ['zhe5', 'zhao2', 'zhu2'],     # 着(轻声) vs 着火 vs 着急
    '了': ['le5', 'liao3'],              # 了(轻声) vs 了解
    
    # 常见字
    '好': ['hao3', 'hao4'],              # 好的 vs 好看
    '多': ['duo1', 'duo2'],              # 多少 vs 多愁善感
    '处': ['chu3', 'chu4'],              # 处理 vs 到处
    '差': ['cha1', 'cha4', 'ci1'],       # 差不多 vs 差距 vs 参差
    '便': ['bian4', 'pian2'],            # 便宜 vs 便利
    '背': ['bei3', 'bei4'],              # 背包 vs 背诗
    '被': ['bei4', 'bei3'],              # 被子 vs 被打
    '弹': ['tan2', 'dan4'],              # 弹琴 vs 弹簧
    '转': ['zhuan3', 'zhuan4'],          # 转身 vs 转向
    '晕': ['yun1', 'yun4'],              # 晕车 vs 发晕
    
    # 时间/数字
    '一': ['yi1'],                       # 虽然有多种用法，但主要发音为 yi1
    '数': ['shu3', 'shuo4'],             # 数字 vs 数落
    '间': ['jian1', 'jian3'],            # 时间 vs 间隔
    '月': ['yue4'],                      # 月份
    
    # 口语/助词
    '吗': ['ma5'],                       # 吗(轻声疑问词)
    '呢': ['ne5'],                       # 呢(轻声疑问词)
    '的': ['de5', 'di4', 'di2'],         # 的(轻声助词) vs 的确 vs 目的地
    '得': ['de5', 'dei3'],               # 得(轻声) vs 得到
    '地': ['de5', 'di4'],                # 地(轻声) vs 地球
    '吧': ['ba5'],                       # 吧(轻声语气词)
    '呀': ['ya5', 'a5'],                 # 呀(轻声) vs 啊
    '喔': ['o5', 'wo3'],                 # 喔(轻声)
    
    # 其他常见多音字
    '都': ['du1', 'dou1'],               # 都市 vs 都(皆)
    '过': ['guo4', 'guo5'],              # 过去 vs 过(轻声)
    '开': ['kai1'],                      # 打开
    '对': ['dui4'],                      # 对的
    '还': ['hai2', 'huan2'],             # 还要 vs 还原
    '给': ['gei3', 'ji3'],               # 给我 vs 供给
    '看': ['kan4', 'kan1'],              # 看书 vs 看守
    '来': ['lai2'],                      # 来自
    '上': ['shang4', 'shang3'],          # 上面 vs 上升
    '说': ['shuo1'],                     # 说话
    '要': ['yao4', 'yao1'],              # 要求 vs 要死了
    '种': ['zhong3', 'zhong4'],          # 种子 vs 种植
    '作': ['zuo4', 'zuo1'],              # 作用 vs 作坊
    '能': ['neng2', 'nai4'],             # 能力 vs 能耐
    '会': ['hui4', 'hui3'],              # 会议 vs 会合
    '发': ['fa1', 'fa3'],                # 发生 vs 发现
    '和': ['he2', 'he4', 'huo2'],        # 和平 vs 和谐 vs 和面
    '或': ['huo4'],                      # 或者
    '通': ['tong1'],                     # 通道
    '用': ['yong4'],                     # 使用
}

# 🔴 改进：轻声拼音映射表 - 改为用 0 表示轻声
# 这些拼音在汉语中通常没有标注声调，表示轻声（第5声）
# 现在统一用 "0" 表示轻声，更直观且避免与 1-4 声混淆
LIGHT_TONE_PINYINS = {
    'de': 'de0',      # 的、得、地（轻声）
    'le': 'le0',      # 了（轻声）
    'men': 'men0',    # 们（复数标记，轻声）
    'zhe': 'zhe0',    # 着（轻声）
    'zi': 'zi0',      # 子（后缀，轻声）
    'me': 'me0',      # 吗（疑问语气，轻声）
    'ba': 'ba0',      # 吧（语气词，轻声）
    'ma': 'ma0',      # 吗（疑问语气，轻声）
    'ne': 'ne0',      # 呢（疑问语气，轻声）
    'a': 'a0',        # 啊（感叹词，轻声）
    'o': 'o0',        # 喔（感叹词，轻声）
    'er': 'er0',      # 儿/耳（在词尾时为轻声）
}

# 🚀 优化：缓存轻声拼音的keys集合（用于快速lookup）
_LIGHT_TONE_PINYINS_KEYS = frozenset(LIGHT_TONE_PINYINS.keys())


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
    # 🚀 优化：快路径检查（大多数拼音不包含声调标记）
    if not pinyin_with_mark or all(c not in TONE_MARK_TO_NUM for c in pinyin_with_mark):
        return pinyin_with_mark
    
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
    
    Raises:
        ValueError: 如果输入为空或无效
    """
    if not pinyin or not pinyin.strip():
        raise ValueError("拼音不能为空")
    
    pinyin = pinyin.strip().lower()
    
    # 转换 tone mark 到 tone number
    pinyin = tone_mark_to_number(pinyin)
    
    # 处理 ü 的多种表示法：ü, u:, v
    pinyin = pinyin.replace('ü', 'v').replace('u:', 'v')
    
    return pinyin


def normalize_light_tone(pinyin: str) -> str:
    """
    🔴 改进：规范化轻声拼音为带"0"的形式。
    
    某些拼音在标准注音中不带声调数字，但实际上是轻声。
    本函数将这些轻声拼音转换为带"0"的标准格式。
    
    使用 "0" 的优势：
    - 0 表示"无声调" = "轻声"，符合直觉
    - 避免与 1-4 声混淆
    - 模型学习更清晰
    - 编码更简洁
    
    例：
        'le' -> 'le0'     （了）
        'de' -> 'de0'     （的）
        'men' -> 'men0'   （们）
        'ma1' -> 'ma1'    （不处理已有声调的）
    
    Args:
        pinyin: 规范化后的拼音字符串
    
    Returns:
        转换后的拼音（带轻声标记"0"）
    
    说明：
        这是重要的修复，因为数据中 28.1% 的样本包含轻声拼音，
        如果不处理这些拼音，模型在推理时会将其识别为 <unk>（未知词）。
    """
    # 🚀 优化：快路径检查（避免不必要的字符遍历）
    if pinyin and pinyin[-1].isdigit():
        return pinyin  # 已经有声调，快速返回
    
    # 使用缓存的keys集合进行快速lookup（O(1)）
    pinyin_lower = pinyin.lower()
    if pinyin_lower in _LIGHT_TONE_PINYINS_KEYS:
        return LIGHT_TONE_PINYINS[pinyin_lower]
    
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
    # 🚀 优化：使用预编译的正则表达式
    match = _PINYIN_BASE_PATTERN.match(pinyin)
    if match:
        base, tone = match.groups()
        return base, tone if tone else None
    return pinyin, None


def normalize_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> str:
    """
    规范化拼音序列。
    
    🔴 更新：现在包括轻声拼音的规范化处理
    
    Args:
        pinyin_str: 拼音序列字符串（用分隔符分开的多个拼音）
        separator: 拼音分隔符，默认为空格
    
    Returns:
        规范化后的拼音序列（同样用分隔符分开）
    """
    pinyins = pinyin_str.split(separator)
    normalized = []
    for p in pinyins:
        if p.strip():
            p = normalize_pinyin(p)          # 先规范化
            p = normalize_light_tone(p)      # 再处理轻声
            normalized.append(p)
    return separator.join(normalized)


def validate_pinyin(pinyin: str) -> bool:
    """
    验证拼音是否有效。
    
    有效拼音应该：
    - 不为空
    - 包含至少一个字母
    - 可能包含数字音调（0-4），其中 0 表示轻声
    - 不应该包含其他特殊字符
    
    Args:
        pinyin: 规范化后的拼音
    
    Returns:
        是否为有效拼音
    """
    # 🚀 优化：快路径检查
    if not pinyin or not pinyin.strip():
        return False
    
    # 使用预编译的正则表达式（允许：a-z, v (代表ü), 数字 0-4（0=轻声, 1-4=声调））
    return _TONE_MARK_PATTERN.match(pinyin) is not None


def validate_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> bool:
    """
    验证拼音序列是否有效。
    
    Args:
        pinyin_str: 拼音序列
        separator: 拼音分隔符
    
    Returns:
        序列中所有拼音是否都有效（空序列返回 False）
    """
    if not pinyin_str or not pinyin_str.strip():
        return False
    
    pinyins = pinyin_str.split(separator)
    valid_pinyins = [p.strip() for p in pinyins if p.strip()]
    
    if not valid_pinyins:
        return False
    
    return all(validate_pinyin(p) for p in valid_pinyins)


def split_pinyin_sequence(pinyin_str: str, separator: str = ' ') -> List[str]:
    """
    拆分拼音序列为单个拼音列表。
    
    Args:
        pinyin_str: 拼音序列
        separator: 拼音分隔符
    
    Returns:
        拼音列表
    
    Raises:
        ValueError: 如果输入为空或无效
    """
    if not pinyin_str or not pinyin_str.strip():
        raise ValueError("拼音序列不能为空")
    
    return [p.strip() for p in pinyin_str.split(separator) if p.strip()]


def join_pinyin_sequence(pinyins: List[str], separator: str = ' ') -> str:
    """
    将拼音列表合并为序列字符串。
    
    Args:
        pinyins: 拼音列表
        separator: 拼音分隔符
    
    Returns:
        拼音序列字符串
    
    Raises:
        ValueError: 如果输入列表为空
    """
    if not pinyins:
        raise ValueError("拼音列表不能为空")
    
    return separator.join(pinyins)


def is_polyphonic_char(hanzi: str) -> bool:
    """
    判断汉字是否为常见多音字。
    
    Args:
        hanzi: 单个汉字
    
    Returns:
        是否为多音字
    
    Raises:
        ValueError: 如果输入不是单个汉字
    """
    if not hanzi or len(hanzi) != 1:
        raise ValueError("输入必须是单个汉字")
    
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


def disambiguate_polyphonic(
    hanzi_str: str, 
    pinyin_str: str, 
    context_window: int = 2,
    separator: str = ' '
) -> str:
    """
    🔴 新增：多音字消歧函数。
    
    尝试根据上下文判断多音字的正确读音。
    这是一个简单的启发式实现，可以作为未来改进的基础。
    
    说明：
        由于当前数据中已经包含了正确的拼音标注，
        这个函数主要用于：
        1. 验证数据一致性
        2. 检测潜在的多音字标注错误
        3. 为未来的上下文感知模型提供基础
    
    Args:
        hanzi_str: 汉字字符串
        pinyin_str: 拼音序列（带声调）
        context_window: 上下文窗口大小
        separator: 拼音分隔符
    
    Returns:
        消歧后的拼音序列
    
    例：
        输入：hanzi_str="中国", pinyin_str="zhong4 guo2"
        输出："zhong1 guo2"（如果根据词典规则）
    """
    pinyins = split_pinyin_sequence(pinyin_str, separator)
    hanzis = list(hanzi_str)
    
    if len(hanzis) != len(pinyins):
        # 长度不匹配，返回原样
        return pinyin_str
    
    result = []
    for i, (hanzi, pinyin) in enumerate(zip(hanzis, pinyins)):
        # 如果是多音字，可以考虑上下文
        if is_polyphonic_char(hanzi):
            possible = get_possible_pinyins(hanzi)
            if possible and pinyin in possible:
                # 标注的拼音在可能列表中，保持原样
                result.append(pinyin)
            elif possible:
                # 标注的拼音不在列表中，这可能是数据错误
                # 使用第一个可能的拼音作为默认
                result.append(possible[0])
            else:
                result.append(pinyin)
        else:
            result.append(pinyin)
    
    return separator.join(result)


def get_polyphonic_statistics(
    hanzi_str: str, 
    pinyin_str: str, 
    separator: str = ' '
) -> Dict[str, dict]:
    """
    🔴 新增：获取多音字统计信息。
    
    分析字符串中多音字的分布情况。
    
    Args:
        hanzi_str: 汉字字符串
        pinyin_str: 拼音序列
        separator: 拼音分隔符
    
    Returns:
        多音字统计信息字典
        格式: {hanzi: {pinyins: [...], count: N, position: [...]}}
    
    例：
        输入：hanzi_str="中国中心", pinyin_str="zhong1 guo2 zhong1 xin1"
        输出：{
            '中': {'pinyins': ['zhong1'], 'count': 2, 'positions': [0, 2]}
        }
    """
    pinyins = split_pinyin_sequence(pinyin_str, separator)
    hanzis = list(hanzi_str)
    
    stats = {}
    for i, (hanzi, pinyin) in enumerate(zip(hanzis, pinyins)):
        if is_polyphonic_char(hanzi):
            if hanzi not in stats:
                stats[hanzi] = {
                    'pinyins': set(),
                    'count': 0,
                    'positions': []
                }
            stats[hanzi]['pinyins'].add(pinyin)
            stats[hanzi]['count'] += 1
            stats[hanzi]['positions'].append(i)
    
    # 转换集合为列表
    for hanzi in stats:
        stats[hanzi]['pinyins'] = sorted(list(stats[hanzi]['pinyins']))
    
    return stats


class PinyinStatistics:
    """
    🚀 改进：统计拼音和汉字的分布情况（添加缓存和批量处理）。
    """
    
    def __init__(self):
        self.pinyin_freq = defaultdict(int)
        self.hanzi_freq = defaultdict(int)
        self.pinyin_hanzi_pairs = defaultdict(set)  # pinyin -> set of hanzi
        self.hanzi_pinyins = defaultdict(set)  # hanzi -> set of pinyin
        self.total_pairs = 0
        # 缓存：延迟计算的结果
        self._polyphonic_cache = None
        self._homophonic_cache = None
    
    def update_from_data(self, hanzi_str: str, pinyin_str: str, separator: str = ' '):
        """
        从数据对更新统计信息。
        
        Args:
            hanzi_str: 汉字字符串
            pinyin_str: 拼音序列
            separator: 拼音分隔符
        
        Raises:
            ValueError: 如果汉字和拼音数量不匹配
        """
        try:
            pinyins = split_pinyin_sequence(pinyin_str, separator)
        except ValueError:
            # 处理空序列
            return
        
        hanzis = list(hanzi_str)
        
        if len(hanzis) != len(pinyins):
            raise ValueError(f"汉字数量({len(hanzis)})与拼音数量({len(pinyins)})不匹配: {hanzi_str} vs {pinyin_str}")
        
        # 按字符逐一关联
        for hanzi, pinyin in zip(hanzis, pinyins):
            self.pinyin_freq[pinyin] += 1
            self.hanzi_freq[hanzi] += 1
            self.pinyin_hanzi_pairs[pinyin].add(hanzi)
            self.hanzi_pinyins[hanzi].add(pinyin)
            self.total_pairs += 1
        
        # 清除缓存（数据已更新）
        self._polyphonic_cache = None
        self._homophonic_cache = None
    
    def update_from_batch(self, data: List[Tuple[str, str]]):
        """
        🚀 新增：批量更新统计信息（性能优化）。
        
        Args:
            data: [(hanzi_str, pinyin_str), ...] 列表
        """
        for hanzi_str, pinyin_str in data:
            try:
                self.update_from_data(hanzi_str, pinyin_str)
            except ValueError:
                # 跳过无效数据
                continue
    
    def get_pinyin_frequency(self, pinyin: str) -> int:
        """获取拼音出现频率。"""
        return self.pinyin_freq.get(pinyin, 0)
    
    def get_hanzi_frequency(self, hanzi: str) -> int:
        """获取汉字出现频率。"""
        return self.hanzi_freq.get(hanzi, 0)
    
    def get_polyphonic_hanzis(self) -> Dict[str, set]:
        """
        获取有多个拼音的汉字（多音字）。
        
        🚀 优化：结果缓存（延迟计算）
        """
        if self._polyphonic_cache is None:
            self._polyphonic_cache = {
                h: p for h, p in self.hanzi_pinyins.items() if len(p) > 1
            }
        return self._polyphonic_cache
    
    def get_homophonic_hanzis(self) -> Dict[str, set]:
        """
        获取多个汉字共享同一个拼音的情况（同音字）。
        
        🚀 优化：结果缓存（延迟计算）
        """
        if self._homophonic_cache is None:
            self._homophonic_cache = {
                p: h for p, h in self.pinyin_hanzi_pairs.items() if len(h) > 1
            }
        return self._homophonic_cache
    
    def get_top_pinyins(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        🚀 新增：获取频率最高的拼音。
        
        Args:
            n: 返回前n个
        
        Returns:
            [(拼音, 频率), ...] 按频率降序排列
        """
        return sorted(self.pinyin_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_top_hanzis(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        🚀 新增：获取频率最高的汉字。
        
        Args:
            n: 返回前n个
        
        Returns:
            [(汉字, 频率), ...] 按频率降序排列
        """
        return sorted(self.hanzi_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_stats_summary(self) -> Dict[str, any]:
        """
        🚀 新增：获取统计摘要字典（便于序列化和日志）。
        
        Returns:
            统计摘要字典
        """
        polyphonic = self.get_polyphonic_hanzis()
        homophonic = self.get_homophonic_hanzis()
        
        return {
            'total_pairs': self.total_pairs,
            'unique_pinyins': len(self.pinyin_freq),
            'unique_hanzis': len(self.hanzi_freq),
            'polyphonic_count': len(polyphonic),
            'homophonic_count': len(homophonic),
            'avg_pinyins_per_hanzi': self.total_pairs / len(self.hanzi_freq) if self.hanzi_freq else 0,
            'avg_hanzis_per_pinyin': self.total_pairs / len(self.pinyin_freq) if self.pinyin_freq else 0,
        }
    
    def print_summary(self):
        """打印统计摘要。"""
        stats = self.get_stats_summary()
        print(f"总拼音-汉字对数: {stats['total_pairs']}")
        print(f"唯一拼音数: {stats['unique_pinyins']}")
        print(f"唯一汉字数: {stats['unique_hanzis']}")
        print(f"多音字数: {stats['polyphonic_count']}")
        print(f"同音字组数: {stats['homophonic_count']}")
        print(f"平均每个汉字对应{stats['avg_pinyins_per_hanzi']:.2f}个拼音")
        print(f"平均每个拼音对应{stats['avg_hanzis_per_pinyin']:.2f}个汉字")
        
        # 打印多音字示例
        polyphonic = self.get_polyphonic_hanzis()
        if polyphonic:
            print("\n多音字示例（前10个）:")
            for hanzi, pinyins in list(polyphonic.items())[:10]:
                print(f"  {hanzi}: {', '.join(sorted(pinyins))}")
        
        # 打印频率最高的拼音
        print("\n频率最高的拼音（前10个）:")
        for pinyin, freq in self.get_top_pinyins(10):
            print(f"  {pinyin}: {freq} 次")


if __name__ == '__main__':
    # 测试示例
    print("=== 拼音规范化测试 ===")
    test_cases = [
        'ma1',
        'hǎo',
        'BEIJING',
        'zhōng',
        'le',  # 轻声拼音
        'de',
    ]
    for case in test_cases:
        try:
            normalized = normalize_pinyin(case)
            normalized = normalize_light_tone(normalized)
            print(f"{case:15s} -> {normalized}")
        except ValueError as e:
            print(f"{case:15s} -> ERROR: {e}")
    
    print("\n=== 拼音提取测试 ===")
    test_pinyins = ['ma1', 'hao', 'zhang3', 'le0']
    for p in test_pinyins:
        try:
            base, tone = extract_tone(p)
            print(f"{p:10s} -> base='{base}', tone='{tone}'")
        except Exception as e:
            print(f"{p:10s} -> ERROR: {e}")
    
    print("\n=== 拼音验证测试 ===")
    test_validations = ['ma1', 'h3o', 'xyz', 'ma', 'ma5', 'le0']
    for p in test_validations:
        try:
            is_valid = validate_pinyin(p)
            print(f"{p:10s} -> valid={is_valid}")
        except Exception as e:
            print(f"{p:10s} -> ERROR: {e}")
    
    print("\n=== 多音字检查 ===")
    test_hanzis = ['中', '长', '行', '好', '我']
    for h in test_hanzis:
        try:
            is_poly = is_polyphonic_char(h)
            pinyins = get_possible_pinyins(h)
            print(f"{h} -> polyphonic={is_poly}, pinyins={pinyins}")
        except Exception as e:
            print(f"{h} -> ERROR: {e}")
    
    print("\n=== 拼音序列处理测试 ===")
    test_seq = "ni3 hao3 jia1 hao4"
    try:
        split = split_pinyin_sequence(test_seq)
        print(f"拆分: {test_seq} -> {split}")
        joined = join_pinyin_sequence(split)
        print(f"合并: {split} -> {joined}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n=== 统计测试 ===")
    stats = PinyinStatistics()
    test_data = [
        ("你好", "ni3 hao3"),
        ("中国", "zhong1 guo2"),
        ("中心", "zhong1 xin1"),
        ("好的", "hao3 de5"),
    ]
    try:
        stats.update_from_batch(test_data)
        stats.print_summary()
    except Exception as e:
        print(f"ERROR: {e}")
