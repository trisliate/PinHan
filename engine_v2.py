"""
IME-SLM 引擎 v2 - 参考华为/搜狗分层策略

架构:
  Level 1: 热词缓存 (<1ms) - LRU缓存 + 用户词库
  Level 2: 词典匹配 (<10ms) - 短输入快速路径
  Level 3: 模型推理 (<100ms) - 长句/复杂输入
"""

import os
import time
import logging
from typing import List, Optional, Dict, Tuple
from functools import lru_cache
from collections import OrderedDict
import torch

from config import EngineConfig, EngineOutput, CandidateResult, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """LRU 缓存实现"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[str, List[CandidateResult]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[List[CandidateResult]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: List[CandidateResult]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IMEEngineV2:
    """
    输入法引擎 v2 - 分层策略
    
    设计原则:
    1. 短输入走快速路径，不触发模型
    2. 热词缓存减少重复计算
    3. 模型推理仅用于复杂场景
    4. 增量输入优化（前缀复用）
    """
    
    # 策略阈值
    SHORT_INPUT_THRESHOLD = 6   # ≤6字符(2个汉字)走词典快速路径
    CACHE_CAPACITY = 2000       # 缓存容量
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # 置信度阈值，超过则跳过模型
    
    def __init__(self, config: Optional[EngineConfig] = None, model_dir: str = None):
        self.config = config or DEFAULT_CONFIG
        self.model_dir = model_dir or os.path.dirname(__file__)
        
        # 缓存
        self.result_cache = LRUCache(self.CACHE_CAPACITY)
        self.prefix_cache: Dict[str, List[str]] = {}  # 前缀 -> 切分结果
        
        # 统计
        self.stats = {
            'level1_hits': 0,  # 缓存命中
            'level2_hits': 0,  # 词典快速路径
            'level3_hits': 0,  # 模型推理
            'total_requests': 0,
            'total_time_ms': 0,
        }
        
        # 子模块（延迟加载）
        self.dict_service = None
        self.corrector = None
        self.segmenter = None
        self.p2h_model = None
        self.p2h_vocab = None
        self.slm_model = None
        self.slm_vocab = None
        self.reranker = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化模块（同 v1）"""
        from dicts import DictionaryService
        from corrector import create_corrector_from_dict
        from segmenter import create_segmenter_from_dict
        
        dicts_dir = os.path.join(self.model_dir, 'dicts')
        char_dict_path = os.path.join(dicts_dir, 'char_dict.json')
        
        self.dict_service = DictionaryService(dicts_dir)
        self.corrector = create_corrector_from_dict(char_dict_path)
        self.segmenter = create_segmenter_from_dict(char_dict_path)
        
        # P2H 模型
        p2h_checkpoint = os.path.join(self.model_dir, 'checkpoints', 'p2h', 'best_model.pt')
        p2h_vocab_path = os.path.join(self.model_dir, 'checkpoints', 'p2h', 'vocab.json')
        
        if os.path.exists(p2h_checkpoint) and os.path.exists(p2h_vocab_path):
            from p2h import P2HModel, P2HVocab
            self.p2h_vocab = P2HVocab()
            self.p2h_vocab.load(p2h_vocab_path)
            checkpoint = torch.load(p2h_checkpoint, map_location=self.device, weights_only=False)
            self.p2h_model = P2HModel(checkpoint['config']).to(self.device)
            self.p2h_model.load_state_dict(checkpoint['model_state_dict'])
            self.p2h_model.eval()
            logger.info("✓ P2H 模型加载成功")
        else:
            logger.warning("⚠ P2H 模型未找到")
        
        # SLM 模型
        slm_checkpoint = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_best.pt')
        slm_vocab_path = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_vocab.json')
        
        if os.path.exists(slm_checkpoint) and os.path.exists(slm_vocab_path):
            from slm import SLModel, SLMVocab, CandidateReranker
            self.slm_vocab = SLMVocab()
            self.slm_vocab.load(slm_vocab_path)
            checkpoint = torch.load(slm_checkpoint, map_location=self.device, weights_only=False)
            self.slm_model = SLModel(checkpoint['config']).to(self.device)
            self.slm_model.load_state_dict(checkpoint['model_state_dict'])
            self.slm_model.eval()
            self.reranker = CandidateReranker(self.slm_model, self.slm_vocab, self.device)
            logger.info("✓ SLM 模型加载成功")
        else:
            logger.warning("⚠ SLM 模型未找到")
        
        self._log_status()
    
    def _log_status(self):
        """输出状态"""
        logger.info("=" * 50)
        logger.info("IME-SLM 引擎 v2 (分层策略)")
        logger.info(f"  词典服务: {'✓' if self.dict_service else '✗'}")
        logger.info(f"  P2H 模型: {'✓' if self.p2h_model else '✗'}")
        logger.info(f"  SLM 模型: {'✓' if self.slm_model else '✗'}")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  短输入阈值: ≤{self.SHORT_INPUT_THRESHOLD} 字符")
        logger.info(f"  缓存容量: {self.CACHE_CAPACITY}")
        logger.info("=" * 50)
    
    def process(self, pinyin: str, context: str = "") -> EngineOutput:
        """
        主入口 - 分层处理策略
        """
        start_time = time.perf_counter()
        self.stats['total_requests'] += 1
        
        pinyin = pinyin.strip().lower()
        cache_key = f"{pinyin}|{context[:20]}"  # 缓存键
        
        # ========== Level 1: 缓存命中 ==========
        cached = self.result_cache.get(cache_key)
        if cached:
            self.stats['level1_hits'] += 1
            elapsed = (time.perf_counter() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed
            return self._build_output(pinyin, cached, elapsed, level=1)
        
        # ========== 切分 + 纠错 ==========
        segments = self._segment(pinyin)
        corrected = self._correct(segments)
        
        # ========== Level 2: 短输入快速路径 ==========
        raw_len = len(pinyin.replace(' ', ''))
        if raw_len <= self.SHORT_INPUT_THRESHOLD:
            candidates = self._dict_lookup_fast(corrected)
            self.stats['level2_hits'] += 1
            
            # 缓存结果
            self.result_cache.put(cache_key, candidates)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed
            return self._build_output(pinyin, candidates, elapsed, level=2, segments=corrected)
        
        # ========== 检查词典置信度 ==========
        dict_candidates = self._dict_lookup_fast(corrected)
        if dict_candidates and dict_candidates[0].score >= self.HIGH_CONFIDENCE_THRESHOLD:
            # 词典结果置信度高，跳过模型
            self.stats['level2_hits'] += 1
            self.result_cache.put(cache_key, dict_candidates)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed
            return self._build_output(pinyin, dict_candidates, elapsed, level=2, segments=corrected)
        
        # ========== Level 3: 模型推理 ==========
        candidates = self._model_inference(corrected, dict_candidates)
        self.stats['level3_hits'] += 1
        
        # SLM 重排
        if self.reranker and len(candidates) > 1:
            candidates = self._rerank(candidates)
        
        # 缓存结果
        self.result_cache.put(cache_key, candidates)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed
        return self._build_output(pinyin, candidates, elapsed, level=3, segments=corrected)
    
    def _segment(self, pinyin: str) -> List[str]:
        """切分拼音（带前缀缓存）"""
        # 检查前缀缓存
        if pinyin in self.prefix_cache:
            return self.prefix_cache[pinyin]
        
        segments = []
        if ' ' in pinyin:
            parts = pinyin.split()
            for part in parts:
                if part in self.segmenter.valid_pinyins:
                    segments.append(part)
                else:
                    result = self.segmenter.segment_best(part)
                    if result:
                        segments.extend(result.segments)
                    else:
                        segments.append(part)
        else:
            result = self.segmenter.segment_best(pinyin)
            if result:
                segments = result.segments
            else:
                segments = [pinyin]
        
        # 缓存
        self.prefix_cache[pinyin] = segments
        return segments
    
    def _correct(self, segments: List[str]) -> List[str]:
        """拼音纠错"""
        corrected = []
        for seg in segments:
            corrections = self.corrector.correct(seg, top_k=1)
            if corrections:
                corrected.append(corrections[0].pinyin)
            else:
                corrected.append(seg)
        return corrected
    
    def _dict_lookup_fast(self, segments: List[str]) -> List[CandidateResult]:
        """快速词典查询"""
        candidates = []
        
        # 1. 完整词组匹配
        words = self.dict_service.get_words(segments)
        for word in words[:5]:
            freq = self.dict_service.get_word_freq(word)
            candidates.append(CandidateResult(
                text=word,
                score=min(freq + 0.5, 1.0),
                source="dict_word"
            ))
        
        # 2. 逐字高频组合
        if len(segments) <= 6:
            text = ""
            score = 1.0
            for seg in segments:
                chars = self.dict_service.get_chars(seg)
                if chars:
                    text += chars[0]
                    score *= max(self.dict_service.get_char_freq(chars[0]), 0.1)
                else:
                    text += "?"
                    score *= 0.01
            
            if "?" not in text:
                candidates.append(CandidateResult(
                    text=text,
                    score=score,
                    source="dict_combo"
                ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:self.config.top_k]
    
    def _model_inference(self, segments: List[str], dict_candidates: List[CandidateResult]) -> List[CandidateResult]:
        """模型推理"""
        candidates = list(dict_candidates)  # 保留词典结果
        
        if self.p2h_model and self.p2h_vocab:
            try:
                pinyin_ids = self.p2h_vocab.encode_pinyin(segments)
                pinyin_tensor = torch.tensor([pinyin_ids], dtype=torch.long, device=self.device)
                
                # 限制生成长度 = 拼音音节数 + 1 (EOS)
                max_len = len(segments) + 1
                
                with torch.no_grad():
                    results = self.p2h_model.beam_search(
                        pinyin_tensor, 
                        beam_size=self.config.top_k,
                        max_len=max_len
                    )
                
                for token_ids, score in results:
                    text = self.p2h_vocab.decode_hanzi(token_ids[0].tolist())
                    # 只保留长度合理的结果 (与拼音音节数接近)
                    if text and abs(len(text) - len(segments)) <= 1:
                        candidates.append(CandidateResult(
                            text=text,
                            score=float(score),
                            source="p2h_model"
                        ))
            except Exception as e:
                logger.debug(f"P2H 推理失败: {e}")
        
        # 去重
        seen = set()
        unique = []
        for c in candidates:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)
        
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique[:self.config.top_k * 2]
    
    def _rerank(self, candidates: List[CandidateResult]) -> List[CandidateResult]:
        """SLM 重排序"""
        texts = [c.text for c in candidates]
        scores = [c.score for c in candidates]
        
        reranked = self.reranker.rerank(texts, scores, alpha=0.6)
        
        return [
            CandidateResult(text=text, score=score, source="reranked")
            for text, score in reranked
        ]
    
    def _build_output(
        self, 
        pinyin: str, 
        candidates: List[CandidateResult], 
        elapsed_ms: float,
        level: int,
        segments: List[str] = None
    ) -> EngineOutput:
        """构建输出"""
        output = EngineOutput(raw_pinyin=pinyin)
        output.candidates = candidates[:self.config.top_k]
        output.segmented_pinyin = segments or []
        output.corrected_pinyin = ' '.join(segments) if segments else pinyin
        
        # 添加性能信息
        output.metadata = {
            'level': level,
            'elapsed_ms': round(elapsed_ms, 2),
            'cache_hit_rate': round(self.result_cache.hit_rate, 3),
        }
        
        return output
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_requests']
        return {
            'total_requests': total,
            'level1_rate': self.stats['level1_hits'] / total if total else 0,
            'level2_rate': self.stats['level2_hits'] / total if total else 0,
            'level3_rate': self.stats['level3_hits'] / total if total else 0,
            'avg_time_ms': self.stats['total_time_ms'] / total if total else 0,
            'cache_hit_rate': self.result_cache.hit_rate,
        }


def create_engine_v2(config: Optional[EngineConfig] = None, model_dir: str = None) -> IMEEngineV2:
    """创建 v2 引擎"""
    return IMEEngineV2(config, model_dir)


if __name__ == "__main__":
    engine = create_engine_v2()
    
    # 测试
    tests = [
        "nihao",          # 短输入 -> Level 2
        "ni hao",         # 短输入 -> Level 2
        "nihao",          # 重复 -> Level 1 (缓存)
        "jintiantianqihenhao",  # 长输入 -> Level 3
        "woaini",         # 短输入 -> Level 2
    ]
    
    print("\n" + "=" * 60)
    print("分层策略测试")
    print("=" * 60)
    
    for pinyin in tests:
        result = engine.process(pinyin)
        meta = result.metadata
        print(f"\n输入: {pinyin}")
        print(f"  Level: {meta['level']} | 耗时: {meta['elapsed_ms']:.2f}ms")
        print(f"  候选: {[c.text for c in result.candidates[:3]]}")
    
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    stats = engine.get_stats()
    print(f"  总请求: {stats['total_requests']}")
    print(f"  Level 1 (缓存): {stats['level1_rate']*100:.1f}%")
    print(f"  Level 2 (词典): {stats['level2_rate']*100:.1f}%")
    print(f"  Level 3 (模型): {stats['level3_rate']*100:.1f}%")
    print(f"  平均耗时: {stats['avg_time_ms']:.2f}ms")
