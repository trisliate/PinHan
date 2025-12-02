from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EngineConfig:
    """引擎配置"""
    top_k: int = 10
    use_slm: bool = True       # SLM 重排序（已启用）
    use_fuzzy: bool = False     # 模糊音匹配（暂时禁用）
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
