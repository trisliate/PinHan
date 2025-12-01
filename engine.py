"""
IME-SLM 引擎主入口

职责：协调各子模块，完成 拼音 -> 汉字候选 的完整流程
"""

from typing import List, Optional
from config import EngineConfig, EngineOutput, CandidateResult, DEFAULT_CONFIG


class IMEEngine:
    """输入法引擎主类"""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # 子模块占位（后续模块实现后替换）
        self.corrector = None      # 模块4: PinyinCorrector
        self.segmenter = None      # 模块5: PinyinSegmenter
        self.p2h_model = None      # 模块6: P2HModel
        self.slm_reranker = None   # 模块7: SLMReranker
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化各子模块"""
        # TODO: 模块4-7实现后在此初始化
        pass
    
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
        
        # Step 1: 拼音纠错
        corrected = self._correct_pinyin(pinyin)
        output.corrected_pinyin = corrected
        
        # Step 2: 拼音切分
        segments = self._segment_pinyin(corrected)
        output.segmented_pinyin = segments
        
        # Step 3: P2H 转换
        candidates = self._pinyin_to_hanzi(segments)
        
        # Step 4: SLM 语义重排
        if self.config.enable_slm_rerank and context:
            candidates = self._rerank_candidates(candidates, context)
        
        # Step 5: 截断并输出
        output.candidates = candidates[:self.config.top_k]
        
        return output
    
    def _correct_pinyin(self, pinyin: str) -> str:
        """拼音纠错"""
        if self.corrector is None:
            # 模块未实现，直接返回原输入
            return pinyin
        return self.corrector.correct(pinyin)
    
    def _segment_pinyin(self, pinyin: str) -> List[str]:
        """拼音切分"""
        if self.segmenter is None:
            # 模块未实现，简单按空格切分
            return pinyin.split() if " " in pinyin else [pinyin]
        return self.segmenter.segment(pinyin)
    
    def _pinyin_to_hanzi(self, segments: List[str]) -> List[CandidateResult]:
        """拼音转汉字"""
        if self.p2h_model is None:
            # 模块未实现，返回占位结果
            return [CandidateResult(text="[P2H未实现]", score=0.0, source="placeholder")]
        return self.p2h_model.predict(segments)
    
    def _rerank_candidates(
        self, 
        candidates: List[CandidateResult], 
        context: str
    ) -> List[CandidateResult]:
        """语义重排"""
        if self.slm_reranker is None:
            return candidates
        return self.slm_reranker.rerank(candidates, context)


def create_engine(config: Optional[EngineConfig] = None) -> IMEEngine:
    """工厂函数：创建引擎实例"""
    return IMEEngine(config)


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
