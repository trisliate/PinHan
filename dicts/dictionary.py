"""
字典服务模块

提供拼音-汉字映射、频率查询等核心功能
"""

import orjson
import os
import re
from typing import Dict, List, Optional, Tuple
from functools import lru_cache


# 声调字符映射表（字符音标 → 数字音标）
TONE_MARKS = {
    'ā': ('a', 1), 'á': ('a', 2), 'ǎ': ('a', 3), 'à': ('a', 4),
    'ē': ('e', 1), 'é': ('e', 2), 'ě': ('e', 3), 'è': ('e', 4),
    'ī': ('i', 1), 'í': ('i', 2), 'ǐ': ('i', 3), 'ì': ('i', 4),
    'ō': ('o', 1), 'ó': ('o', 2), 'ǒ': ('o', 3), 'ò': ('o', 4),
    'ū': ('u', 1), 'ú': ('u', 2), 'ǔ': ('u', 3), 'ù': ('u', 4),
    'ǖ': ('v', 1), 'ǘ': ('v', 2), 'ǚ': ('v', 3), 'ǜ': ('v', 4),
    'ü': ('v', 0), 'ń': ('n', 2), 'ň': ('n', 3), 'ǹ': ('n', 4),
}


class PinyinUtils:
    """拼音工具类"""
    
    @staticmethod
    def normalize_pinyin(pinyin: str) -> str:
        """
        将拼音标准化为无调拼音
        支持：字符音标(nǐ)、数字音标(ni3)、无调(ni)
        """
        pinyin = pinyin.lower().strip()
        
        # 处理字符音标
        result = []
        for char in pinyin:
            if char in TONE_MARKS:
                result.append(TONE_MARKS[char][0])
            else:
                result.append(char)
        pinyin = ''.join(result)
        
        # 移除数字声调
        pinyin = re.sub(r'[1-5]', '', pinyin)
        
        # ü → v 统一处理
        pinyin = pinyin.replace('ü', 'v')
        
        return pinyin
    
    @staticmethod
    def extract_tone(pinyin: str) -> Tuple[str, int]:
        """
        提取拼音的声调
        返回：(无调拼音, 声调数字)
        声调：1-4 为一到四声，5 为轻声，0 为无声调
        """
        pinyin = pinyin.lower().strip()
        tone = 0
        
        # 检查字符音标
        for char in pinyin:
            if char in TONE_MARKS:
                _, tone = TONE_MARKS[char]
                break
        
        # 检查数字音标
        if tone == 0:
            match = re.search(r'([1-5])$', pinyin)
            if match:
                tone = int(match.group(1))
        
        # 获取无调拼音
        base = PinyinUtils.normalize_pinyin(pinyin)
        
        return base, tone


class DictionaryService:
    """字典服务主类"""
    
    def __init__(self, dict_dir: str = "dicts"):
        self.dict_dir = dict_dir
        
        # 数据存储
        self._pinyin_set: set = set()           # 合法拼音集合
        self._char_dict: Dict[str, List[str]] = {}  # 拼音 → 字列表
        self._word_dict: Dict[str, List[str]] = {}  # 拼音 → 词列表
        self._char_freq: Dict[str, float] = {}  # 字 → 频率
        self._word_freq: Dict[str, float] = {}  # 词 → 频率
        
        # 加载数据
        self._load_all()
    
    def _get_path(self, filename: str) -> str:
        """获取字典文件路径"""
        return os.path.join(self.dict_dir, filename)
    
    def _load_json(self, filename: str) -> dict:
        """加载 JSON 文件"""
        path = self._get_path(filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return orjson.loads(f.read())
        return {}
    
    def _load_all(self):
        """加载所有字典数据"""
        # 加载拼音表
        pinyin_path = self._get_path('pinyin_table.txt')
        if os.path.exists(pinyin_path):
            with open(pinyin_path, 'r', encoding='utf-8') as f:
                self._pinyin_set = set(line.strip() for line in f if line.strip())
        
        # 加载字典
        self._char_dict = self._load_json('char_dict.json')
        self._word_dict = self._load_json('word_dict.json')
        self._char_freq = self._load_json('char_freq.json')
        self._word_freq = self._load_json('word_freq.json')
        
        # 如果拼音表为空，从 char_dict 提取
        if not self._pinyin_set and self._char_dict:
            self._pinyin_set = set(self._char_dict.keys())
    
    def is_valid_pinyin(self, pinyin: str) -> bool:
        """判断是否为合法拼音音节"""
        normalized = PinyinUtils.normalize_pinyin(pinyin)
        return normalized in self._pinyin_set
    
    def get_all_pinyin(self) -> List[str]:
        """获取所有合法拼音列表"""
        return sorted(self._pinyin_set)
    
    def get_chars(self, pinyin: str) -> List[str]:
        """
        拼音 → 单字列表（按频率排序）
        支持带调拼音，优先返回匹配声调的字
        """
        base, tone = PinyinUtils.extract_tone(pinyin)
        chars = self._char_dict.get(base, [])
        
        # 按频率排序
        chars_with_freq = [(c, self._char_freq.get(c, 0.0)) for c in chars]
        chars_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in chars_with_freq]
    
    def get_words(self, pinyin_list: List[str]) -> List[str]:
        """
        拼音序列 → 词组列表
        """
        # 将拼音序列转为查询键
        normalized = [PinyinUtils.normalize_pinyin(p) for p in pinyin_list]
        key = ' '.join(normalized)
        return self._word_dict.get(key, [])
    
    def get_char_freq(self, char: str) -> float:
        """获取单字频率"""
        return self._char_freq.get(char, 0.0)
    
    def get_word_freq(self, word: str) -> float:
        """获取词组频率"""
        return self._word_freq.get(word, 0.0)
    
    def get_candidates(
        self, 
        pinyin_list: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        综合查询：返回候选及评分
        
        Args:
            pinyin_list: 拼音序列，如 ["ni", "hao"]
            top_k: 返回数量
        
        Returns:
            [(候选文本, 评分), ...]
        """
        candidates = []
        
        # 1. 查询词组
        words = self.get_words(pinyin_list)
        for word in words:
            freq = self.get_word_freq(word)
            # 词组优先，给予基础分加成
            score = freq + 0.5
            candidates.append((word, score))
        
        # 2. 如果只有一个拼音，查询单字
        if len(pinyin_list) == 1:
            chars = self.get_chars(pinyin_list[0])
            for char in chars:
                freq = self.get_char_freq(char)
                candidates.append((char, freq))
        
        # 3. 组合单字（简单笛卡尔积，仅用于短拼音序列）
        if len(pinyin_list) <= 4 and not words:
            char_lists = [self.get_chars(p) for p in pinyin_list]
            if all(char_lists):
                # 限制组合数量
                from itertools import product
                count = 0
                for combo in product(*char_lists):
                    if count >= 100:
                        break
                    text = ''.join(combo)
                    # 计算组合评分
                    score = sum(self.get_char_freq(c) for c in combo) / len(combo)
                    candidates.append((text, score))
                    count += 1
        
        # 去重并排序
        seen = set()
        unique = []
        for text, score in candidates:
            if text not in seen:
                seen.add(text)
                unique.append((text, score))
        
        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:top_k]
    
    def reload(self):
        """重新加载字典"""
        self._load_all()


# 全局单例
_dict_service: Optional[DictionaryService] = None


def get_dict_service(dict_dir: str = "dicts") -> DictionaryService:
    """获取字典服务单例"""
    global _dict_service
    if _dict_service is None:
        _dict_service = DictionaryService(dict_dir)
    return _dict_service
