"""
IME-SLM 引擎主入口

职责：协调各子模块，完成 拼音 -> 汉字候选 的完整流程
"""

import os
import logging
from typing import List, Optional
import torch

from config import EngineConfig, EngineOutput, CandidateResult, DEFAULT_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMEEngine:
    """输入法引擎主类"""
    
    def __init__(self, config: Optional[EngineConfig] = None, model_dir: str = None):
        self.config = config or DEFAULT_CONFIG
        self.model_dir = model_dir or os.path.dirname(__file__)
        
        # 子模块
        self.dict_service = None   # 词典服务
        self.corrector = None      # 模块4: PinyinCorrector
        self.segmenter = None      # 模块5: PinyinSegmenter
        self.p2h_model = None      # 模块6: P2HModel
        self.p2h_vocab = None      # P2H 词表
        self.slm_model = None      # 模块7: SLModel
        self.slm_vocab = None      # SLM 词表
        self.reranker = None       # 重排序器
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化各子模块"""
        # 词典服务
        from dicts import DictionaryService
        dicts_dir = os.path.join(self.model_dir, 'dicts')
        self.dict_service = DictionaryService(dicts_dir)
        
        # 拼音纠错
        from corrector import create_corrector_from_dict
        char_dict_path = os.path.join(dicts_dir, 'char_dict.json')
        self.corrector = create_corrector_from_dict(char_dict_path)
        
        # 拼音切分
        from segmenter import create_segmenter_from_dict
        self.segmenter = create_segmenter_from_dict(char_dict_path)
        
        # P2H 模型（如果存在检查点）
        p2h_checkpoint = os.path.join(self.model_dir, 'checkpoints', 'p2h', 'best_model.pt')
        p2h_vocab_path = os.path.join(self.model_dir, 'checkpoints', 'p2h', 'vocab.json')
        
        if os.path.exists(p2h_checkpoint) and os.path.exists(p2h_vocab_path):
            from p2h import P2HModel, P2HVocab
            
            # 加载词表
            self.p2h_vocab = P2HVocab()
            self.p2h_vocab.load(p2h_vocab_path)
            
            # 加载模型
            checkpoint = torch.load(p2h_checkpoint, map_location=self.device)
            self.p2h_model = P2HModel(checkpoint['config']).to(self.device)
            self.p2h_model.load_state_dict(checkpoint['model_state_dict'])
            self.p2h_model.eval()
            logger.info("✓ P2H 模型加载成功")
        else:
            logger.warning("⚠ P2H 模型未找到，将使用词典回退")
            logger.warning(f"  检查点路径: {p2h_checkpoint}")
        
        # SLM 模型（如果存在检查点）
        slm_checkpoint = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_best.pt')
        slm_vocab_path = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_vocab.json')
        
        if os.path.exists(slm_checkpoint) and os.path.exists(slm_vocab_path):
            from slm import SLModel, SLMVocab, CandidateReranker
            
            # 加载词表
            self.slm_vocab = SLMVocab()
            self.slm_vocab.load(slm_vocab_path)
            
            # 加载模型
            checkpoint = torch.load(slm_checkpoint, map_location=self.device)
            self.slm_model = SLModel(checkpoint['config']).to(self.device)
            self.slm_model.load_state_dict(checkpoint['model_state_dict'])
            self.slm_model.eval()
            
            # 重排序器
            self.reranker = CandidateReranker(self.slm_model, self.slm_vocab, self.device)
            logger.info("✓ SLM 模型加载成功")
        else:
            logger.warning("⚠ SLM 模型未找到，将禁用语义重排")
            logger.warning(f"  检查点路径: {slm_checkpoint}")
        
        # 输出模块状态汇总
        self._log_module_status()
    
    def _log_module_status(self):
        """输出模块加载状态汇总"""
        logger.info("=" * 50)
        logger.info("IME-SLM 引擎模块状态:")
        logger.info(f"  词典服务: {'✓ 已加载' if self.dict_service else '✗ 未加载'}")
        logger.info(f"  拼音纠错: {'✓ 已加载' if self.corrector else '✗ 未加载'}")
        logger.info(f"  拼音切分: {'✓ 已加载' if self.segmenter else '✗ 未加载'}")
        logger.info(f"  P2H 模型: {'✓ 已加载' if self.p2h_model else '✗ 未加载 (使用词典回退)'}")
        logger.info(f"  SLM 模型: {'✓ 已加载' if self.slm_model else '✗ 未加载 (禁用重排)'}")
        logger.info(f"  设备: {self.device}")
        logger.info("=" * 50)

    def process(self, pinyin: str, context: str = "") -> EngineOutput:
        """
        主处理流程
        
        Args:
            pinyin: 用户输入的原始拼音（可能含错误、无分隔）
            context: 上下文已确认文本（用于语义重排）
        
        Returns:
            EngineOutput: 包含候选列表及中间结果
        """
        output = EngineOutput(raw_pinyin=pinyin)
        
        # Step 1: 拼音切分
        segments = []
        if ' ' in pinyin:
            # 有空格分隔，对每个部分分别切分
            parts = pinyin.split()
            for part in parts:
                if part in self.segmenter.valid_pinyins:
                    segments.append(part)
                else:
                    # 尝试切分
                    result = self.segmenter.segment_best(part)
                    if result:
                        segments.extend(result.segments)
                    else:
                        segments.append(part)
        else:
            # 无空格，整体切分
            segment_result = self.segmenter.segment_best(pinyin)
            if segment_result:
                segments = segment_result.segments
            else:
                segments = [pinyin]
        
        # Step 2: 拼音纠错
        corrected_segments = []
        for seg in segments:
            corrections = self.corrector.correct(seg, top_k=1)
            if corrections:
                corrected_segments.append(corrections[0].pinyin)
            else:
                corrected_segments.append(seg)
        
        output.corrected_pinyin = ' '.join(corrected_segments)
        output.segmented_pinyin = corrected_segments
        
        # Step 3: P2H 转换
        candidates = self._pinyin_to_hanzi(corrected_segments)
        
        # Step 4: SLM 语义重排
        if self.config.enable_slm_rerank and self.reranker and len(candidates) > 1:
            candidate_texts = [c.text for c in candidates]
            candidate_scores = [c.score for c in candidates]
            
            reranked = self.reranker.rerank(candidate_texts, candidate_scores, alpha=0.7)
            
            candidates = [
                CandidateResult(text=text, score=score, source="reranked")
                for text, score in reranked
            ]
        
        # Step 5: 截断并输出
        output.candidates = candidates[:self.config.top_k]
        
        return output
    
    def _pinyin_to_hanzi(self, segments: List[str]) -> List[CandidateResult]:
        """拼音转汉字"""
        candidates = []
        
        # 方法1: 使用 P2H 模型（如果已加载）
        if self.p2h_model is not None and self.p2h_vocab is not None:
            try:
                model_candidates = self._p2h_model_predict(segments)
                candidates.extend(model_candidates)
            except Exception as e:
                pass  # 静默失败，使用词典回退
        
        # 方法2: 词典查询（回退方案，或补充候选）
        dict_candidates = self._dict_lookup(segments)
        
        # 合并候选，去重
        seen = set()
        merged = []
        for c in candidates + dict_candidates:
            if c.text not in seen:
                seen.add(c.text)
                merged.append(c)
        
        # 按分数排序
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged[:self.config.top_k * 2]  # 多返回一些给重排序
    
    def _p2h_model_predict(self, segments: List[str]) -> List[CandidateResult]:
        """使用 P2H 模型预测"""
        # 编码拼音
        pinyin_ids = self.p2h_vocab.encode_pinyin(segments)
        pinyin_tensor = torch.tensor([pinyin_ids], dtype=torch.long, device=self.device)
        
        # Beam Search
        results = self.p2h_model.beam_search(pinyin_tensor, beam_size=self.config.top_k)
        
        candidates = []
        for token_ids, score in results:
            text = self.p2h_vocab.decode_hanzi(token_ids[0].tolist())
            if text:
                candidates.append(CandidateResult(
                    text=text,
                    score=float(score),
                    source="p2h_model"
                ))
        
        return candidates
    
    def _dict_lookup(self, segments: List[str]) -> List[CandidateResult]:
        """词典查询方式生成候选"""
        candidates = []
        
        if len(segments) == 1:
            # 单字/单词查询
            pinyin = segments[0]
            
            # 查字
            chars = self.dict_service.get_chars(pinyin)
            for char in chars[:10]:
                freq = self.dict_service.get_char_freq(char)
                candidates.append(CandidateResult(
                    text=char,
                    score=freq,
                    source="dict_char"
                ))
            
            # 查词（单拼音作为列表传入）
            words = self.dict_service.get_words([pinyin])
            for word in words[:10]:
                freq = self.dict_service.get_word_freq(word)
                candidates.append(CandidateResult(
                    text=word,
                    score=freq + 0.1,  # 词略优先
                    source="dict_word"
                ))
        else:
            # 多音节查询
            # 1. 尝试整体作为词查询
            words = self.dict_service.get_words(segments)
            for word in words[:5]:
                freq = self.dict_service.get_word_freq(word)
                candidates.append(CandidateResult(
                    text=word,
                    score=freq + 0.5,  # 完整词组优先
                    source="dict_word"
                ))
            
            # 2. 尝试分段词组匹配（如 "jin tian" + "tian qi" + ...）
            candidates.extend(self._segment_word_lookup(segments))
            
            # 3. 逐字查询组合（生成多个候选）
            if len(segments) <= 8:  # 限制长度避免组合爆炸
                candidates.extend(self._char_combination_lookup(segments))
        
        return candidates
    
    def _segment_word_lookup(self, segments: List[str]) -> List[CandidateResult]:
        """分段词组匹配：尝试将拼音序列分成词组"""
        candidates = []
        n = len(segments)
        
        # 使用动态规划找最佳分词
        # dp[i] = [(text, score), ...] 表示 segments[:i] 的最佳组合
        dp = [[] for _ in range(n + 1)]
        dp[0] = [("", 1.0)]
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 4), i):  # 最长4字词
                sub_segments = segments[j:i]
                
                # 尝试作为词查询
                words = self.dict_service.get_words(sub_segments)
                if words:
                    word = words[0]
                    freq = self.dict_service.get_word_freq(word) + 0.3
                    for prev_text, prev_score in dp[j][:3]:
                        dp[i].append((prev_text + word, prev_score * freq))
                
                # 尝试逐字
                if len(sub_segments) == 1:
                    chars = self.dict_service.get_chars(sub_segments[0])
                    if chars:
                        char = chars[0]
                        freq = self.dict_service.get_char_freq(char)
                        for prev_text, prev_score in dp[j][:3]:
                            dp[i].append((prev_text + char, prev_score * max(freq, 0.1)))
            
            # 保留前几个
            dp[i].sort(key=lambda x: x[1], reverse=True)
            dp[i] = dp[i][:5]
        
        for text, score in dp[n]:
            if text:
                candidates.append(CandidateResult(
                    text=text,
                    score=score,
                    source="dict_segment"
                ))
        
        return candidates
    
    def _char_combination_lookup(self, segments: List[str]) -> List[CandidateResult]:
        """字符组合查询：生成多个候选组合"""
        candidates = []
        
        # 获取每个拼音的候选字（取前3个）
        char_options = []
        for seg in segments:
            chars = self.dict_service.get_chars(seg)
            if chars:
                char_options.append(chars[:3])
            else:
                char_options.append(['?'])
        
        # 生成组合（限制数量）
        from itertools import product
        
        # 限制组合数量
        max_combinations = 10
        count = 0
        
        for combo in product(*char_options):
            if count >= max_combinations:
                break
            
            text = ''.join(combo)
            if '?' not in text:
                # 计算组合分数
                score = 0.3
                for char in combo:
                    score *= max(self.dict_service.get_char_freq(char), 0.1)
                
                candidates.append(CandidateResult(
                    text=text,
                    score=score,
                    source="dict_combo"
                ))
                count += 1
        
        return candidates


def create_engine(config: Optional[EngineConfig] = None, model_dir: str = None) -> IMEEngine:
    """工厂函数：创建引擎实例"""
    return IMEEngine(config, model_dir)


# 简单测试
if __name__ == "__main__":
    engine = create_engine()
    result = engine.process("nihao")
    
    print("原始拼音:", result.raw_pinyin)
    print("纠正后:", result.corrected_pinyin)
    print("切分后:", result.segmented_pinyin)
    print("候选数:", len(result.candidates))
    for c in result.candidates:
        print(f"  {c.text} (score={c.score}, source={c.source})")
