"""
IME Engine v3 - 精简版

架构: 词典召回 + SLM 重排序
目标: 单模型，低延迟，高准确率
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """引擎配置"""
    top_k: int = 10
    use_slm: bool = True
    cache_size: int = 2000


@dataclass
class CandidateResult:
    """候选结果"""
    text: str
    score: float
    source: str = "dict"


@dataclass
class EngineOutput:
    """引擎输出"""
    raw_pinyin: str = ""
    candidates: List[CandidateResult] = field(default_factory=list)
    segmented_pinyin: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class LRUCache:
    """简单 LRU 缓存"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str):
        if key in self.cache:
            self.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IMEEngineV3:
    """
    IME 引擎 v3 - 精简版
    
    核心思路：词典为主，SLM 辅助
    - 短输入 (<=6字符): 纯词典，无模型
    - 长输入: 词典召回 + SLM 重排序
    """
    
    SHORT_THRESHOLD = 6  # 短输入阈值
    
    def __init__(self, config: EngineConfig = None, model_dir: str = None):
        self.config = config or EngineConfig()
        self.model_dir = model_dir or os.path.dirname(__file__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 缓存
        self.cache = LRUCache(self.config.cache_size)
        
        # 模块
        self.dict_service = None
        self.corrector = None
        self.segmenter = None
        self.slm_model = None
        self.slm_vocab = None
        self.reranker = None
        
        # 统计
        self.stats = {'total': 0, 'cache_hits': 0, 'slm_calls': 0, 'total_ms': 0}
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化模块"""
        from dicts import DictionaryService
        from corrector import create_corrector_from_dict
        from segmenter import create_segmenter_from_dict
        
        dicts_dir = os.path.join(self.model_dir, 'dicts')
        char_dict_path = os.path.join(dicts_dir, 'char_dict.json')
        
        self.dict_service = DictionaryService(dicts_dir)
        self.corrector = create_corrector_from_dict(char_dict_path)
        self.segmenter = create_segmenter_from_dict(char_dict_path)
        
        # SLM Lite 模型 (优先加载轻量版)
        slm_lite_path = os.path.join(self.model_dir, 'checkpoints', 'slm_lite', 'best.pt')
        slm_lite_vocab = os.path.join(self.model_dir, 'checkpoints', 'slm_lite', 'vocab.json')
        
        # 回退到旧版 SLM
        slm_path = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_best.pt')
        slm_vocab_path = os.path.join(self.model_dir, 'checkpoints', 'slm', 'slm_vocab.json')
        
        if os.path.exists(slm_lite_path) and os.path.exists(slm_lite_vocab):
            self._load_slm(slm_lite_path, slm_lite_vocab, "SLM Lite")
        elif os.path.exists(slm_path) and os.path.exists(slm_vocab_path):
            self._load_slm(slm_path, slm_vocab_path, "SLM")
        else:
            logger.warning("⚠ SLM 模型未找到，仅使用词典")
        
        self._log_status()
    
    def _load_slm(self, checkpoint_path: str, vocab_path: str, name: str):
        """加载 SLM 模型"""
        try:
            from slm import SLModel, SLMVocab, CandidateReranker
            
            self.slm_vocab = SLMVocab()
            self.slm_vocab.load(vocab_path)
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.slm_model = SLModel(checkpoint['config']).to(self.device)
            self.slm_model.load_state_dict(checkpoint['model_state_dict'])
            self.slm_model.eval()
            
            self.reranker = CandidateReranker(self.slm_model, self.slm_vocab, self.device)
            
            params = sum(p.numel() for p in self.slm_model.parameters())
            logger.info(f"✓ {name} 加载成功 ({params:,} 参数)")
        except Exception as e:
            logger.error(f"SLM 加载失败: {e}")
    
    def _log_status(self):
        """输出状态"""
        logger.info("=" * 50)
        logger.info("IME 引擎 v3 (词典 + SLM)")
        logger.info(f"  词典: {'✓' if self.dict_service else '✗'}")
        logger.info(f"  SLM:  {'✓' if self.slm_model else '✗'}")
        logger.info(f"  设备: {self.device}")
        logger.info("=" * 50)
    
    def process(self, pinyin: str, context: str = "") -> EngineOutput:
        """主处理入口"""
        start = time.perf_counter()
        self.stats['total'] += 1
        
        pinyin = pinyin.strip().lower()
        cache_key = f"{pinyin}|{context[-10:]}"
        
        # 缓存命中
        cached = self.cache.get(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            elapsed = (time.perf_counter() - start) * 1000
            return self._build_output(pinyin, cached, elapsed, cached=True)
        
        # 分词 + 纠错
        segments = self._segment(pinyin)
        corrected = self._correct(segments)
        
        # 词典召回
        candidates = self._dict_lookup(corrected)
        
        # 长输入或有上下文时，使用 SLM 重排序
        # 无上下文的短/中输入完全依赖词频
        use_slm = (
            self.reranker and 
            len(candidates) > 1 and
            context  # 只在有上下文时使用 SLM
        )
        
        if use_slm:
            self.stats['slm_calls'] += 1
            candidates = self._rerank(candidates, context)
        
        # 缓存
        self.cache.put(cache_key, candidates)
        
        elapsed = (time.perf_counter() - start) * 1000
        self.stats['total_ms'] += elapsed
        
        return self._build_output(pinyin, candidates, elapsed, segments=corrected)
    
    def _segment(self, pinyin: str) -> List[str]:
        """分词"""
        if ' ' in pinyin:
            parts = pinyin.split()
            segments = []
            for part in parts:
                if part in self.segmenter.valid_pinyins:
                    segments.append(part)
                else:
                    result = self.segmenter.segment_best(part)
                    segments.extend(result.segments if result else [part])
            return segments
        else:
            result = self.segmenter.segment_best(pinyin)
            return result.segments if result else [pinyin]
    
    def _correct(self, segments: List[str]) -> List[str]:
        """纠错 (取 top1)"""
        corrected = []
        for seg in segments:
            corrections = self.corrector.correct(seg, top_k=1)
            corrected.append(corrections[0].pinyin if corrections else seg)
        return corrected
    
    def _dict_lookup(self, segments: List[str], limit: int = None) -> List[CandidateResult]:
        """词典查询 (含模糊音扩展)"""
        limit = limit or self.config.top_k * 3
        candidates = []
        seen = set()
        
        # 生成拼音组合 (模糊音)
        combos = self._expand_fuzzy(segments)
        
        for pinyins in combos:
            is_original = (pinyins == segments)
            
            # 词组匹配
            words = self.dict_service.get_words(pinyins)
            for word in words[:20 if is_original else 5]:
                if word in seen:
                    continue
                seen.add(word)
                
                freq = self.dict_service.get_word_freq(word)
                score = min(freq + 0.5, 1.0) * (1.0 if is_original else 0.8)
                
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
        
        # 逐字组合
        if 1 < len(segments) <= 4:
            text, score = "", 1.0
            for seg in segments:
                chars = self.dict_service.get_chars(seg)
                if chars:
                    text += chars[0]
                    score *= max(self.dict_service.get_char_freq(chars[0]), 0.1)
                else:
                    text, score = "", 0
                    break
            
            if text and text not in seen:
                candidates.append(CandidateResult(text=text, score=score, source="combo"))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]
    
    def _expand_fuzzy(self, segments: List[str]) -> List[List[str]]:
        """模糊音扩展"""
        if len(segments) > 3:
            return [segments]
        
        import itertools
        
        options_list = []
        for seg in segments:
            options = [seg]
            corrs = self.corrector.correct(seg, top_k=5)
            for c in corrs:
                if c.pinyin != seg and c.score >= 0.85:
                    options.append(c.pinyin)
            options_list.append(options[:3])
        
        combos = [segments]
        for combo in itertools.product(*options_list):
            c = list(combo)
            if c != segments:
                combos.append(c)
        
        return combos
    
    def _rerank(self, candidates: List[CandidateResult], context: str) -> List[CandidateResult]:
        """SLM 重排序"""
        texts = [c.text for c in candidates]
        scores = [c.score for c in candidates]
        
        # 动态 alpha: 有上下文时更依赖 SLM，无上下文时更依赖词频
        alpha = 0.5 if context else 0.2
        
        reranked = self.reranker.rerank(texts, scores, alpha=alpha, context=context[-10:])
        
        return [CandidateResult(text=t, score=s, source="reranked") for t, s in reranked]
    
    def _build_output(
        self, 
        pinyin: str, 
        candidates: List[CandidateResult], 
        elapsed_ms: float,
        segments: List[str] = None,
        cached: bool = False
    ) -> EngineOutput:
        """构建输出"""
        return EngineOutput(
            raw_pinyin=pinyin,
            candidates=candidates[:self.config.top_k],
            segmented_pinyin=segments or [],
            metadata={
                'elapsed_ms': round(elapsed_ms, 2),
                'cached': cached,
                'cache_rate': round(self.cache.hit_rate, 3),
            }
        )
    
    def get_stats(self) -> Dict:
        """获取统计"""
        total = self.stats['total'] or 1
        return {
            'total_requests': self.stats['total'],
            'cache_hit_rate': self.stats['cache_hits'] / total,
            'slm_call_rate': self.stats['slm_calls'] / total,
            'avg_latency_ms': self.stats['total_ms'] / total,
        }


def create_engine_v3(config: EngineConfig = None, model_dir: str = None) -> IMEEngineV3:
    """创建 v3 引擎"""
    return IMEEngineV3(config, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = create_engine_v3()
    
    # 测试
    tests = [
        ("nihao", "", "你好"),
        ("jintian", "", "今天"),
        ("tianqi", "今天", "天气"),
        ("shi", "我做", "事"),
        ("shengme", "", "什么"),
        ("renzhen", "", "认真"),
    ]
    
    print("\n测试结果:")
    print("-" * 50)
    
    # 预热
    engine.process("test")
    
    for pinyin, ctx, expected in tests:
        result = engine.process(pinyin, context=ctx)
        texts = [c.text for c in result.candidates]
        rank = texts.index(expected) + 1 if expected in texts else -1
        status = "✓" if rank > 0 and rank <= 3 else "✗"
        print(f"{status} '{pinyin}' (ctx='{ctx}'): {expected} @ #{rank} | {result.metadata['elapsed_ms']:.1f}ms")
        print(f"   候选: {texts[:5]}")
    
    print("-" * 50)
    print(f"统计: {engine.get_stats()}")
