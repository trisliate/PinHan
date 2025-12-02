from .config import EngineConfig, EngineOutput, CandidateResult
from .core import IMEEngineV3

def create_engine_v3(config: EngineConfig = None, model_dir: str = None) -> IMEEngineV3:
    """创建 v3 引擎"""
    return IMEEngineV3(config, model_dir)
