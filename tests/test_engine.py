"""
引擎主入口单元测试
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import IMEEngine, create_engine
from config import EngineConfig, EngineOutput


class TestIMEEngine:
    """引擎基础测试"""
    
    def test_create_engine(self):
        """测试引擎创建"""
        engine = create_engine()
        assert engine is not None
        assert isinstance(engine, IMEEngine)
    
    def test_create_engine_with_config(self):
        """测试带配置创建引擎"""
        config = EngineConfig(top_k=5, enable_corrector=False)
        engine = create_engine(config)
        assert engine.config.top_k == 5
        assert engine.config.enable_corrector is False
    
    def test_process_returns_output(self):
        """测试 process 返回正确类型"""
        engine = create_engine()
        result = engine.process("nihao")
        assert isinstance(result, EngineOutput)
    
    def test_process_preserves_raw_pinyin(self):
        """测试原始拼音保留"""
        engine = create_engine()
        result = engine.process("nihao")
        assert result.raw_pinyin == "nihao"
    
    def test_process_with_context(self):
        """测试带上下文处理"""
        engine = create_engine()
        result = engine.process("nihao", context="你好，")
        assert result is not None


class TestEngineConfig:
    """配置测试"""
    
    def test_default_config(self):
        """测试默认配置值"""
        config = EngineConfig()
        assert config.top_k == 9
        assert config.enable_corrector is True
        assert config.enable_slm_rerank is True
        assert config.device == "cuda"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EngineConfig(
            top_k=5,
            device="cpu",
            enable_slm_rerank=False
        )
        assert config.top_k == 5
        assert config.device == "cpu"
        assert config.enable_slm_rerank is False
