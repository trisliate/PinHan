"""
IME-SLM 全局配置
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EngineConfig:
    """引擎主配置"""
    
    # 候选数量
    top_k: int = 9
    
    # 模块开关
    enable_corrector: bool = True
    enable_slm_rerank: bool = True
    
    # 模型路径
    p2h_model_path: str = "models/p2h"
    slm_model_path: str = "models/slm"
    
    # 字典路径
    dict_dir: str = "dicts"
    
    # 设备配置
    device: str = "cuda"  # cuda / cpu
    
    # 日志级别
    log_level: str = "INFO"


@dataclass
class CandidateResult:
    """单个候选结果"""
    text: str           # 汉字文本
    score: float        # 综合评分
    source: str = ""    # 来源标记 (p2h / dict / slm)


@dataclass
class EngineOutput:
    """引擎输出"""
    candidates: List[CandidateResult] = field(default_factory=list)
    raw_pinyin: str = ""
    corrected_pinyin: str = ""
    segmented_pinyin: List[str] = field(default_factory=list)


# 默认配置实例
DEFAULT_CONFIG = EngineConfig()
