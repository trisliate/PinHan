from .config import EngineConfig, EngineOutput, CandidateResult
from .core import IMEEngineV3
from .dictionary import DictionaryService, PinyinUtils, get_dict_service
from .corrector import PinyinCorrector, CorrectionCandidate, create_corrector_from_dict
from .segmenter import PinyinSegmenter, SegmentResult, create_segmenter_from_dict
from .logging import setup_logging, get_logger, get_api_logger, get_engine_logger

def create_engine_v3(config: EngineConfig = None, dicts_dir: str = None) -> IMEEngineV3:
    """
    创建 v3 引擎
    
    Args:
        config: 引擎配置
        dicts_dir: 词典目录路径（可选，默认使用项目根目录下的 data/dicts）
    
    Returns:
        IMEEngineV3 实例
    """
    return IMEEngineV3(config, dicts_dir)

__all__ = [
    # 引擎
    'IMEEngineV3',
    'create_engine_v3',
    'EngineConfig',
    'EngineOutput',
    'CandidateResult',
    # 字典
    'DictionaryService',
    'PinyinUtils',
    'get_dict_service',
    # 纠错
    'PinyinCorrector',
    'CorrectionCandidate',
    'create_corrector_from_dict',
    # 切分
    'PinyinSegmenter',
    'SegmentResult',
    'create_segmenter_from_dict',
    # 日志
    'setup_logging',
    'get_logger',
    'get_api_logger',
    'get_engine_logger',
]
