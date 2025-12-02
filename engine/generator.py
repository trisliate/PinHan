import math
import itertools
from typing import List, Tuple, Set
from .config import CandidateResult, EngineConfig

class CandidateGenerator:
    """候选生成器 (负责词典召回和 Beam Search)"""
    
    def __init__(self, dict_service, corrector, config: EngineConfig):
        self.dict_service = dict_service
        self.corrector = corrector
        self.config = config
        
    def generate(self, segments: List[str], limit: int = None) -> List[CandidateResult]:
        """生成候选"""
        limit = limit or self.config.top_k * 3
        candidates = []
        seen = set()
        
        # 根据配置决定是否启用模糊音
        if self.config.use_fuzzy:
            combos = self._expand_fuzzy(segments)
        else:
            combos = [segments]  # 只用原始拼音
        
        for pinyins in combos:
            is_original = (pinyins == segments)
            
            # 词组匹配 - 只在多拼音时查询
            # 单拼音查词组会错误匹配多音节词（如 xian → 西安）
            if len(pinyins) > 1:
                words = self.dict_service.get_words(pinyins)
                for i, word in enumerate(words[:20 if is_original else 5]):
                    if word in seen:
                        continue
                    seen.add(word)
                    
                    freq = self.dict_service.get_word_freq(word)
                    # 直接使用词频作为分数
                    # 加上排名惩罚，确保词典排序优先
                    rank_bonus = 1.0 / (i + 1) * 0.1
                    score = freq + rank_bonus
                    if not is_original:
                        score *= 0.8  # 模糊音降权
                    
                    candidates.append(CandidateResult(
                        text=word,
                        score=score,
                        source="dict" if is_original else "fuzzy"
                    ))
        
        # 单字候选
        if len(segments) == 1:
            chars = self.dict_service.get_chars(segments[0])
            for char in chars[:15]:
                if char not in seen:
                    seen.add(char)
                    candidates.append(CandidateResult(
                        text=char,
                        score=self.dict_service.get_char_freq(char),
                        source="char"
                    ))
        
        # Beam Search 组合（多字时生成多个候选）
        if len(segments) > 1:
            beam_results = self._beam_search(segments, beam_width=5)
            for text, score in beam_results:
                if text not in seen:
                    seen.add(text)
                    candidates.append(CandidateResult(text=text, score=score, source="beam"))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]

    def _beam_search(self, segments: List[str], beam_width: int = 5) -> List[Tuple[str, float]]:
        """
        Beam Search 生成多个候选组合 (Viterbi-style)
        确保在每一步都只比较相同长度的路径
        """
        n = len(segments)
        if n == 0:
            return []
        
        # dp[i] 存储到达第 i 个位置的候选列表: List[Tuple[text, score]]
        dp = [[] for _ in range(n + 1)]
        dp[0] = [("", 0.0)]
        
        for i in range(n):
            if not dp[i]:
                continue
            
            # 剪枝：当前位置只保留分数最高的 beam_width * 2 个路径
            # 避免路径爆炸
            current_candidates = sorted(dp[i], key=lambda x: x[1], reverse=True)[:beam_width * 2]
            
            for curr_text, curr_score in current_candidates:
                # 1. 单字 (始终尝试)
                # 移动步长: 1
                char_pinyin = segments[i]
                chars = self.dict_service.get_chars(char_pinyin)
                
                # 如果没有候选字，保留原拼音（作为兜底）
                if not chars:
                    next_score = curr_score - 10.0  # 惩罚
                    dp[i + 1].append((curr_text + char_pinyin, next_score))
                else:
                    for char in chars[:beam_width]:
                        freq = self.dict_service.get_char_freq(char)
                        # log(freq)
                        char_score = math.log(max(freq, 1e-10))
                        dp[i + 1].append((curr_text + char, curr_score + char_score))
                
                # 2. 多字词组 (尝试不同长度)
                # 移动步长: length (2 to 4)
                max_len = min(4, n - i)
                for length in range(2, max_len + 1):
                    sub_segments = segments[i : i + length]
                    words = self.dict_service.get_words(sub_segments)
                    
                    if words:
                        for word in words[:beam_width]:
                            freq = self.dict_service.get_word_freq(word)
                            word_score = math.log(max(freq, 1e-10))
                            
                            # 词组奖励：鼓励长词
                            # 长度越长，奖励越大，抵消 log 概率的累积下降
                            # 经验值：每个额外字符奖励 1.5 分 (相当于 log prob 增加 1.5)
                            bonus = (length - 1) * 1.5
                            
                            dp[i + length].append((curr_text + word, curr_score + word_score + bonus))
        
        # 收集最终结果 (到达位置 n 的路径)
        final_results = dp[n]
        
        # 转换回概率分数 (exp) 并排序
        results = []
        seen = set()
        
        # 按 log_prob 排序
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        for text, log_prob in final_results:
            if text not in seen:
                seen.add(text)
                # 归一化分数，避免过小
                # 这里返回 exp(log_prob) 可能会非常小，但只要相对顺序对就行
                # 为了显示好看，可以不 exp，或者 exp 后乘个系数
                # 保持原接口返回 float score
                results.append((text, math.exp(log_prob)))
        
        return results[:beam_width]

    def _expand_fuzzy(self, segments: List[str]) -> List[List[str]]:
        """模糊音扩展 - 保守策略"""
        # 短输入不做模糊音扩展（避免误扩展）
        if len(segments) <= 2:
            return [segments]
        
        if len(segments) > 3:
            return [segments]
        
        options_list = []
        for seg in segments:
            options = [seg]
            corrs = self.corrector.correct(seg, top_k=3)
            for c in corrs:
                # 只接受高置信度的纠错（0.95以上）
                if c.pinyin != seg and c.score >= 0.95:
                    options.append(c.pinyin)
            options_list.append(options[:2])  # 最多2个选项
        
        combos = [segments]
        for combo in itertools.product(*options_list):
            c = list(combo)
            if c != segments:
                combos.append(c)
        
        return combos
