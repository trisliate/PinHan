"""
拼音切分模块

功能：
1. 将连续拼音字符串切分为单个拼音
2. 支持多种切分方案（歧义处理）
3. 使用动态规划找最优切分
"""

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class SegmentResult:
    """切分结果"""
    segments: List[str]  # 切分后的拼音列表
    score: float         # 切分得分（越高越好）
    
    def __str__(self):
        return " ".join(self.segments)


class PinyinSegmenter:
    """拼音切分器"""
    
    # 所有有效拼音（无声调）
    VALID_PINYINS = {
        # 单韵母
        'a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er',
        
        # b
        'ba', 'bo', 'bi', 'bu', 'bai', 'bei', 'bao', 'ban', 'ben', 'bang', 'beng',
        'bie', 'biao', 'bian', 'bin', 'bing',
        
        # p
        'pa', 'po', 'pi', 'pu', 'pai', 'pei', 'pao', 'pou', 'pan', 'pen', 'pang', 'peng',
        'pie', 'piao', 'pian', 'pin', 'ping',
        
        # m
        'ma', 'mo', 'me', 'mi', 'mu', 'mai', 'mei', 'mao', 'mou', 'man', 'men', 'mang', 'meng',
        'mie', 'miao', 'miu', 'mian', 'min', 'ming',
        
        # f
        'fa', 'fo', 'fu', 'fei', 'fou', 'fan', 'fen', 'fang', 'feng',
        
        # d
        'da', 'de', 'di', 'du', 'dai', 'dei', 'dao', 'dou', 'dan', 'den', 'dang', 'deng', 'dong',
        'die', 'diao', 'diu', 'dian', 'ding',
        'duo', 'dui', 'duan', 'dun',
        
        # t
        'ta', 'te', 'ti', 'tu', 'tai', 'tei', 'tao', 'tou', 'tan', 'tang', 'teng', 'tong',
        'tie', 'tiao', 'tian', 'ting',
        'tuo', 'tui', 'tuan', 'tun',
        
        # n
        'na', 'ne', 'ni', 'nu', 'nv', 'nai', 'nei', 'nao', 'nou', 'nan', 'nen', 'nang', 'neng', 'nong',
        'nie', 'niao', 'niu', 'nian', 'nin', 'niang', 'ning',
        'nuo', 'nuan', 'nve',
        
        # l
        'la', 'le', 'li', 'lu', 'lv', 'lai', 'lei', 'lao', 'lou', 'lan', 'lang', 'leng', 'long',
        'lia', 'lie', 'liao', 'liu', 'lian', 'lin', 'liang', 'ling',
        'luo', 'luan', 'lun', 'lve',
        
        # g
        'ga', 'ge', 'gu', 'gai', 'gei', 'gao', 'gou', 'gan', 'gen', 'gang', 'geng', 'gong',
        'gua', 'guo', 'guai', 'gui', 'guan', 'gun', 'guang',
        
        # k
        'ka', 'ke', 'ku', 'kai', 'kei', 'kao', 'kou', 'kan', 'ken', 'kang', 'keng', 'kong',
        'kua', 'kuo', 'kuai', 'kui', 'kuan', 'kun', 'kuang',
        
        # h
        'ha', 'he', 'hu', 'hai', 'hei', 'hao', 'hou', 'han', 'hen', 'hang', 'heng', 'hong',
        'hua', 'huo', 'huai', 'hui', 'huan', 'hun', 'huang',
        
        # j
        'ji', 'jia', 'jie', 'jiao', 'jiu', 'jian', 'jin', 'jiang', 'jing', 'jiong',
        'ju', 'jue', 'juan', 'jun',
        
        # q
        'qi', 'qia', 'qie', 'qiao', 'qiu', 'qian', 'qin', 'qiang', 'qing', 'qiong',
        'qu', 'que', 'quan', 'qun',
        
        # x
        'xi', 'xia', 'xie', 'xiao', 'xiu', 'xian', 'xin', 'xiang', 'xing', 'xiong',
        'xu', 'xue', 'xuan', 'xun',
        
        # zh
        'zha', 'zhe', 'zhi', 'zhu', 'zhai', 'zhei', 'zhao', 'zhou', 'zhan', 'zhen', 'zhang', 'zheng', 'zhong',
        'zhua', 'zhuo', 'zhuai', 'zhui', 'zhuan', 'zhun', 'zhuang',
        
        # ch
        'cha', 'che', 'chi', 'chu', 'chai', 'chao', 'chou', 'chan', 'chen', 'chang', 'cheng', 'chong',
        'chua', 'chuo', 'chuai', 'chui', 'chuan', 'chun', 'chuang',
        
        # sh
        'sha', 'she', 'shi', 'shu', 'shai', 'shei', 'shao', 'shou', 'shan', 'shen', 'shang', 'sheng',
        'shua', 'shuo', 'shuai', 'shui', 'shuan', 'shun', 'shuang',
        
        # r
        'ri', 're', 'ru', 'rao', 'rou', 'ran', 'ren', 'rang', 'reng', 'rong',
        'rua', 'ruo', 'rui', 'ruan', 'run',
        
        # z
        'za', 'ze', 'zi', 'zu', 'zai', 'zei', 'zao', 'zou', 'zan', 'zen', 'zang', 'zeng', 'zong',
        'zuo', 'zui', 'zuan', 'zun',
        
        # c
        'ca', 'ce', 'ci', 'cu', 'cai', 'cao', 'cou', 'can', 'cen', 'cang', 'ceng', 'cong',
        'cuo', 'cui', 'cuan', 'cun',
        
        # s
        'sa', 'se', 'si', 'su', 'sai', 'sao', 'sou', 'san', 'sen', 'sang', 'seng', 'song',
        'suo', 'sui', 'suan', 'sun',
        
        # y
        'ya', 'yo', 'ye', 'yi', 'yu', 'yai', 'yao', 'you', 'yan', 'yin', 'yang', 'ying', 'yong',
        'yue', 'yuan', 'yun',
        
        # w
        'wa', 'wo', 'wu', 'wai', 'wei', 'wan', 'wen', 'wang', 'weng',
    }
    
    def __init__(self, pinyin_freq: dict = None):
        """
        初始化切分器
        
        Args:
            pinyin_freq: 拼音频率字典，用于计算切分得分
        """
        self.valid_pinyins = self.VALID_PINYINS
        self.pinyin_freq = pinyin_freq or {}
        self.max_pinyin_len = max(len(p) for p in self.valid_pinyins)
    
    def segment(self, text: str, top_k: int = 5) -> List[SegmentResult]:
        """
        切分拼音字符串
        
        Args:
            text: 连续拼音字符串（如 "nihaoma"）
            top_k: 返回最多 top_k 个切分方案
        
        Returns:
            切分结果列表，按得分降序
        """
        text = text.lower().strip()
        if not text:
            return []
        
        n = len(text)
        
        # dp[i] = [(segments, score), ...] 表示 text[:i] 的所有切分方案
        dp: List[List[Tuple[List[str], float]]] = [[] for _ in range(n + 1)]
        dp[0] = [([], 1.0)]
        
        for i in range(1, n + 1):
            candidates = []
            
            # 尝试所有可能的最后一个拼音
            for j in range(max(0, i - self.max_pinyin_len), i):
                last_pinyin = text[j:i]
                
                if last_pinyin in self.valid_pinyins:
                    # 计算这个拼音的得分
                    py_score = self._pinyin_score(last_pinyin)
                    
                    # 继承之前的切分
                    for prev_segments, prev_score in dp[j]:
                        new_segments = prev_segments + [last_pinyin]
                        new_score = prev_score * py_score
                        candidates.append((new_segments, new_score))
            
            # 保留 top_k 个最好的
            candidates.sort(key=lambda x: x[1], reverse=True)
            dp[i] = candidates[:top_k * 2]  # 多保留一些以应对后续筛选
        
        # 返回最终结果
        results = []
        for segments, score in dp[n]:
            results.append(SegmentResult(segments=segments, score=score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def segment_best(self, text: str) -> Optional[SegmentResult]:
        """返回最佳切分方案"""
        results = self.segment(text, top_k=1)
        return results[0] if results else None
    
    def _pinyin_score(self, pinyin: str) -> float:
        """计算单个拼音的得分"""
        # 基础分
        base_score = 0.9
        
        # 长度偏好：更长的拼音更具体，优先选择
        # 这样 xue(3) 会优于 xu(2) + e(1)
        length_bonus = len(pinyin) * 0.05
        
        # 频率加成
        if self.pinyin_freq:
            freq = self.pinyin_freq.get(pinyin, 0)
            # 归一化
            freq_score = min(freq / 1000, 0.05)
            return base_score + length_bonus + freq_score
        
        return base_score + length_bonus
    
    def is_valid_sequence(self, pinyins: List[str]) -> bool:
        """检查拼音序列是否全部有效"""
        return all(py in self.valid_pinyins for py in pinyins)


def create_segmenter_from_dict(dict_path: str = None) -> PinyinSegmenter:
    """
    从词典创建切分器
    
    Args:
        dict_path: 拼音频率文件路径
    
    Returns:
        PinyinSegmenter 实例
    """
    pinyin_freq = {}
    
    if dict_path and os.path.exists(dict_path):
        import orjson
        with open(dict_path, 'rb') as f:
            char_dict = orjson.loads(f.read())
        
        # 统计每个拼音出现的次数
        for pinyin, chars in char_dict.items():
            pinyin_freq[pinyin] = len(chars)
    
    return PinyinSegmenter(pinyin_freq)


if __name__ == '__main__':
    # 测试
    segmenter = PinyinSegmenter()
    
    test_cases = [
        'nihao',
        'nihaoma',
        'zhongguoren',
        'xian',       # 西安 vs 先
        'fangan',     # 方案 vs 翻干
        'xianzai',
        'womenshizuihaodepengyou',
    ]
    
    print("拼音切分测试：")
    for text in test_cases:
        results = segmenter.segment(text, top_k=3)
        print(f"\n  {text}:")
        for i, r in enumerate(results):
            print(f"    {i+1}. {r} (score={r.score:.4f})")
