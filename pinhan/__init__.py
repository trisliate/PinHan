"""
PinHan - 轻量级智能拼音输入法引擎

纯词典版，专为嵌入式/MCU 设备优化
"""

__version__ = "0.1.0"

from pinhan.engine import (
    IMEEngineV3,
    create_engine_v3,
    EngineConfig,
    EngineOutput,
    CandidateResult,
    DictionaryService,
    PinyinUtils,
    get_dict_service,
    PinyinCorrector,
    CorrectionCandidate,
    create_corrector_from_dict,
    PinyinSegmenter,
    SegmentResult,
    create_segmenter_from_dict,
)

__all__ = [
    "__version__",
    # 引擎
    "IMEEngineV3",
    "create_engine_v3",
    "EngineConfig",
    "EngineOutput",
    "CandidateResult",
    # 字典
    "DictionaryService",
    "PinyinUtils",
    "get_dict_service",
    # 纠错
    "PinyinCorrector",
    "CorrectionCandidate",
    "create_corrector_from_dict",
    # 切分
    "PinyinSegmenter",
    "SegmentResult",
    "create_segmenter_from_dict",
]
