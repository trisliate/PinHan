"""
拼音纠错模块

功能：
1. 编辑距离纠错（拼写错误）
2. 键盘距离纠错（相邻键误触）
3. 音近纠错（模糊音：z/zh, c/ch, s/sh, n/l, f/h, l/r, an/ang, en/eng, in/ing）
"""

from typing import List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class CorrectionCandidate:
    """纠错候选"""
    pinyin: str
    score: float  # 0-1，越高越好
    reason: str   # 纠错原因


# QWERTY 键盘布局，用于计算键盘距离
KEYBOARD_LAYOUT = [
    'qwertyuiop',
    'asdfghjkl',
    'zxcvbnm'
]

# 构建键位坐标映射
KEY_POSITIONS = {}
for row_idx, row in enumerate(KEYBOARD_LAYOUT):
    for col_idx, key in enumerate(row):
        KEY_POSITIONS[key] = (row_idx, col_idx)


# 模糊音规则：(原音, 替换音, 权重)
# 权重越高表示越容易混淆
FUZZY_RULES = [
    # 声母模糊
    ('zh', 'z', 0.9),
    ('ch', 'c', 0.9),
    ('sh', 's', 0.9),
    ('z', 'zh', 0.9),
    ('c', 'ch', 0.9),
    ('s', 'sh', 0.9),
    ('n', 'l', 0.8),
    ('l', 'n', 0.8),
    ('f', 'h', 0.7),
    ('h', 'f', 0.7),
    ('r', 'l', 0.6),
    ('l', 'r', 0.6),
    
    # 韵母模糊
    ('an', 'ang', 0.9),
    ('ang', 'an', 0.9),
    ('en', 'eng', 0.9),
    ('eng', 'en', 0.9),
    ('in', 'ing', 0.9),
    ('ing', 'in', 0.9),
    ('ian', 'iang', 0.8),
    ('iang', 'ian', 0.8),
    ('uan', 'uang', 0.8),
    ('uang', 'uan', 0.8),
]


class PinyinCorrector:
    """拼音纠错器"""
    
    def __init__(self, valid_pinyins: Set[str]):
        """
        初始化纠错器
        
        Args:
            valid_pinyins: 有效拼音集合（从词典加载）
        """
        self.valid_pinyins = valid_pinyins
        # 预计算模糊音映射
        self._build_fuzzy_map()
    
    def _build_fuzzy_map(self):
        """构建模糊音快速查找表"""
        self.fuzzy_map = {}  # pinyin -> [(variant, score), ...]
        
        for pinyin in self.valid_pinyins:
            variants = self._generate_fuzzy_variants(pinyin)
            for variant, score in variants:
                if variant not in self.fuzzy_map:
                    self.fuzzy_map[variant] = []
                self.fuzzy_map[variant].append((pinyin, score))
    
    def _generate_fuzzy_variants(self, pinyin: str) -> List[Tuple[str, float]]:
        """生成拼音的模糊变体"""
        variants = []
        
        for pattern, replacement, weight in FUZZY_RULES:
            if pattern in pinyin:
                variant = pinyin.replace(pattern, replacement, 1)
                variants.append((variant, weight))
        
        return variants
    
    def correct(self, pinyin: str, top_k: int = 5) -> List[CorrectionCandidate]:
        """
        纠正拼音
        
        Args:
            pinyin: 待纠正的拼音
            top_k: 返回候选数量
        
        Returns:
            纠错候选列表，按分数降序
        """
        candidates = []
        
        # 1. 如果已经是有效拼音，直接返回
        if pinyin in self.valid_pinyins:
            candidates.append(CorrectionCandidate(
                pinyin=pinyin,
                score=1.0,
                reason="exact"
            ))
            return candidates
        
        # 2. 模糊音纠错
        if pinyin in self.fuzzy_map:
            for correct_py, score in self.fuzzy_map[pinyin]:
                candidates.append(CorrectionCandidate(
                    pinyin=correct_py,
                    score=score,
                    reason="fuzzy"
                ))
        
        # 3. 编辑距离纠错（距离1）
        for valid_py in self.valid_pinyins:
            dist = self._edit_distance(pinyin, valid_py)
            if dist == 1:
                # 检查是否是键盘相邻键
                kb_score = self._keyboard_similarity(pinyin, valid_py)
                score = 0.7 + 0.2 * kb_score  # 基础0.7，键盘相似加成
                candidates.append(CorrectionCandidate(
                    pinyin=valid_py,
                    score=score,
                    reason="edit_dist_1"
                ))
            elif dist == 2 and len(pinyin) > 3:
                candidates.append(CorrectionCandidate(
                    pinyin=valid_py,
                    score=0.5,
                    reason="edit_dist_2"
                ))
        
        # 4. 去重并排序
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.pinyin not in seen:
                seen.add(c.pinyin)
                unique_candidates.append(c)
        
        unique_candidates.sort(key=lambda x: x.score, reverse=True)
        return unique_candidates[:top_k]
    
    def correct_sequence(self, pinyins: List[str]) -> List[List[CorrectionCandidate]]:
        """
        纠正拼音序列
        
        Args:
            pinyins: 拼音列表
        
        Returns:
            每个拼音的候选列表
        """
        return [self.correct(py) for py in pinyins]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离（Levenshtein）"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def _keyboard_similarity(self, s1: str, s2: str) -> float:
        """
        计算键盘相似度
        
        Returns:
            0-1，1表示非常相似（相邻键）
        """
        if len(s1) != len(s2):
            return 0.0
        
        total_dist = 0.0
        diff_count = 0
        
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                diff_count += 1
                dist = self._key_distance(c1, c2)
                total_dist += dist
        
        if diff_count == 0:
            return 1.0
        
        # 平均键盘距离，归一化到0-1
        avg_dist = total_dist / diff_count
        # 距离1（相邻）-> 1.0，距离越大越低
        return max(0, 1.0 - (avg_dist - 1) * 0.3)
    
    def _key_distance(self, k1: str, k2: str) -> float:
        """计算两个键的距离"""
        if k1 not in KEY_POSITIONS or k2 not in KEY_POSITIONS:
            return 5.0  # 未知键，返回大距离
        
        r1, c1 = KEY_POSITIONS[k1]
        r2, c2 = KEY_POSITIONS[k2]
        
        # 欧几里得距离
        return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def create_corrector_from_dict(dict_path: str) -> PinyinCorrector:
    """
    从词典文件创建纠错器
    
    Args:
        dict_path: char_dict.json 或类似文件路径
    
    Returns:
        PinyinCorrector 实例
    """
    import orjson
    
    with open(dict_path, 'rb') as f:
        char_dict = orjson.loads(f.read())
    
    # 提取所有有效拼音
    valid_pinyins = set(char_dict.keys())
    
    return PinyinCorrector(valid_pinyins)


# 标准拼音表（用于快速验证）
VALID_INITIALS = {
    '', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 
    'g', 'k', 'h', 'j', 'q', 'x', 
    'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w'
}

VALID_FINALS = {
    'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ao', 'ou',
    'an', 'en', 'ang', 'eng', 'ong', 'er',
    'ia', 'ie', 'iao', 'iu', 'ian', 'in', 'iang', 'ing', 'iong',
    'ua', 'uo', 'uai', 'ui', 'uan', 'un', 'uang', 'ueng',
    'va', 've', 'van', 'vn'
}


if __name__ == '__main__':
    # 测试
    import os
    
    dict_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'dicts', 'char_dict.json'
    )
    
    if os.path.exists(dict_path):
        corrector = create_corrector_from_dict(dict_path)
        
        test_cases = [
            'ni',      # 正确
            'nii',     # 多打一个i
            'zhonh',   # zhong 打错
            'zhongguo',  # 正确
            'zon',     # 模糊音 zong
            'xang',    # 模糊音 xiang
            'lv',      # 正确
        ]
        
        print("拼音纠错测试：")
        for py in test_cases:
            results = corrector.correct(py, top_k=3)
            print(f"\n  {py}:")
            for r in results:
                print(f"    -> {r.pinyin} (score={r.score:.2f}, {r.reason})")
    else:
        print(f"词典不存在: {dict_path}")
