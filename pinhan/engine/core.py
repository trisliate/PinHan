import os
import time
from typing import List, Dict, Tuple

from .config import EngineConfig, EngineOutput, CandidateResult
from .cache import LRUCache
from .generator import CandidateGenerator
from .dictionary import DictionaryService
from .corrector import create_corrector_from_dict
from .segmenter import create_segmenter_from_dict
from .logging import get_engine_logger

logger = get_engine_logger()

class IMEEngineV3:
    """
    IME 引擎 v3 - 纯词典版
    
    核心思路：词典为主，规则为辅
    - 纯词典召回
    - 规则重排序
    """
    
    # 标点符号集合（中英文）
    PUNCTUATION = set("，。！？、；：""''（）……——·,.:;!?\"'()-_+=[]{}|\\<>/@#$%^&*~`")
    
    def __init__(self, config: EngineConfig = None, model_dir: str = None):
        self.config = config or EngineConfig()
        self.model_dir = model_dir or os.path.dirname(os.path.dirname(__file__)) # engine/..
        
        # 缓存
        self.cache = LRUCache(self.config.cache_size)
        
        # 模块
        self.dict_service = None
        self.corrector = None
        self.segmenter = None
        self.generator = None
        
        # 统计
        self.stats = {'total': 0, 'cache_hits': 0, 'total_ms': 0}
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化模块"""
        # 字典数据位于 data/dicts
        dicts_dir = os.path.join(self.model_dir, 'data', 'dicts')
        char_dict_path = os.path.join(dicts_dir, 'char_dict.json')
        
        self.dict_service = DictionaryService(dicts_dir)
        self.corrector = create_corrector_from_dict(char_dict_path)
        self.segmenter = create_segmenter_from_dict(char_dict_path)
        
        # 初始化生成器
        self.generator = CandidateGenerator(self.dict_service, self.corrector, self.config)
        
        self._log_status()
    
    def _log_status(self):
        """输出状态"""
        logger.info("=" * 50)
        logger.info("PinHan 引擎 (纯词典版)")
        logger.info(f"  词典: {'✓' if self.dict_service else '✗'}")
        logger.info("=" * 50)
    
    def process(self, pinyin: str, context: str = "") -> EngineOutput:
        """主处理入口"""
        start = time.perf_counter()
        self.stats['total'] += 1
        
        raw_input = pinyin.strip()
        
        # 分离拼音和标点符号
        parts = self._split_pinyin_and_punct(raw_input)
        
        # 如果全是标点，直接返回
        if all(p[0] == 'punct' for p in parts):
            punct_text = ''.join(p[1] for p in parts)
            elapsed = (time.perf_counter() - start) * 1000
            return self._build_output(raw_input, [CandidateResult(punct_text, 1.0, "punct")], elapsed)
        
        # 处理每个拼音片段，收集候选
        all_candidates = []
        running_context = context  # 累积上下文
        
        for part_type, part_value in parts:
            if part_type == 'punct':
                # 标点直接作为候选的一部分（后续合并）
                all_candidates.append(('punct', part_value))
                running_context += part_value  # 标点也加入上下文
            else:
                # 拼音片段，走正常流程
                pinyin_lower = part_value.lower()
                cache_key = f"{pinyin_lower}|{running_context[-10:]}"
                
                cached = self.cache.get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    all_candidates.append(('pinyin', cached))
                    # 用首选更新上下文
                    if cached:
                        running_context += cached[0].text
                else:
                    segments = self._segment(pinyin_lower)
                    corrected = self._correct(segments)
                    
                    # 使用生成器生成候选
                    candidates = self.generator.generate(corrected)
                    
                    self.cache.put(cache_key, candidates)
                    all_candidates.append(('pinyin', candidates))
                    # 用首选更新上下文
                    if candidates:
                        running_context += candidates[0].text
        
        # 合并结果：拼音候选 + 标点
        merged = self._merge_candidates_with_punct(all_candidates)
        
        elapsed = (time.perf_counter() - start) * 1000
        self.stats['total_ms'] += elapsed
        
        return self._build_output(raw_input, merged, elapsed)
    
    def _split_pinyin_and_punct(self, text: str) -> List[Tuple[str, str]]:
        """将输入拆分为拼音片段和标点符号"""
        parts = []
        current = ""
        current_type = None
        
        for char in text:
            is_punct = char in self.PUNCTUATION
            char_type = 'punct' if is_punct else 'pinyin'
            
            if current_type is None:
                current_type = char_type
                current = char
            elif char_type == current_type:
                current += char
            else:
                if current:
                    parts.append((current_type, current))
                current_type = char_type
                current = char
        
        if current:
            parts.append((current_type, current))
        
        return parts
    
    def _merge_candidates_with_punct(self, all_candidates: List) -> List[CandidateResult]:
        """合并拼音候选和标点符号"""
        if not all_candidates:
            return []
        
        # 找出所有拼音候选的数量（取最少的）
        pinyin_counts = [len(c[1]) for c in all_candidates if c[0] == 'pinyin' and c[1]]
        if not pinyin_counts:
            # 没有拼音候选，只有标点
            punct_text = ''.join(c[1] for c in all_candidates if c[0] == 'punct')
            return [CandidateResult(punct_text, 1.0, "punct")]
        
        num_candidates = min(pinyin_counts + [self.config.top_k])
        
        merged = []
        for i in range(num_candidates):
            text = ""
            score = 1.0
            for ctype, cvalue in all_candidates:
                if ctype == 'punct':
                    text += cvalue
                else:
                    if i < len(cvalue):
                        text += cvalue[i].text
                        score *= cvalue[i].score
                    elif cvalue:
                        text += cvalue[0].text
                        score *= cvalue[0].score
            
            merged.append(CandidateResult(text, score, "merged"))
        
        return merged
    
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
    
    def _build_output(
        self, 
        pinyin: str, 
        candidates: List[CandidateResult], 
        elapsed_ms: float,
        segments: List[str] = None
    ) -> EngineOutput:
        """构建输出"""
        return EngineOutput(
            raw_pinyin=pinyin,
            candidates=candidates[:self.config.top_k],
            segmented_pinyin=segments or [],
            metadata={
                'elapsed_ms': round(elapsed_ms, 2),
                'cache_rate': round(self.cache.hit_rate, 3),
            }
        )
    
    def get_stats(self) -> Dict:
        """获取统计"""
        total = self.stats['total'] or 1
        return {
            'total_requests': self.stats['total'],
            'cache_hit_rate': self.stats['cache_hits'] / total,
            'avg_latency_ms': self.stats['total_ms'] / total,
        }
