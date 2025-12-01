"""
字典模块单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dictionary import (
    DictionaryService, 
    PinyinUtils,
    get_dict_service
)


class TestPinyinUtils:
    """拼音工具测试"""
    
    def test_normalize_no_tone(self):
        """无调拼音标准化"""
        assert PinyinUtils.normalize_pinyin("ni") == "ni"
        assert PinyinUtils.normalize_pinyin("hao") == "hao"
        assert PinyinUtils.normalize_pinyin("NI") == "ni"
    
    def test_normalize_number_tone(self):
        """数字音标标准化"""
        assert PinyinUtils.normalize_pinyin("ni3") == "ni"
        assert PinyinUtils.normalize_pinyin("hao3") == "hao"
        assert PinyinUtils.normalize_pinyin("zhong1") == "zhong"
    
    def test_normalize_char_tone(self):
        """字符音标标准化"""
        assert PinyinUtils.normalize_pinyin("nǐ") == "ni"
        assert PinyinUtils.normalize_pinyin("hǎo") == "hao"
        assert PinyinUtils.normalize_pinyin("zhōng") == "zhong"
    
    def test_normalize_v(self):
        """ü 处理"""
        assert PinyinUtils.normalize_pinyin("lü") == "lv"
        assert PinyinUtils.normalize_pinyin("nü") == "nv"
        assert PinyinUtils.normalize_pinyin("lǖ") == "lv"
    
    def test_extract_tone_number(self):
        """提取数字声调"""
        base, tone = PinyinUtils.extract_tone("ni3")
        assert base == "ni"
        assert tone == 3
    
    def test_extract_tone_char(self):
        """提取字符声调"""
        base, tone = PinyinUtils.extract_tone("nǐ")
        assert base == "ni"
        assert tone == 3
    
    def test_extract_tone_none(self):
        """无声调"""
        base, tone = PinyinUtils.extract_tone("ni")
        assert base == "ni"
        assert tone == 0


class TestDictionaryService:
    """字典服务测试"""
    
    def test_create_service(self):
        """创建服务"""
        service = DictionaryService()
        assert service is not None
    
    def test_get_singleton(self):
        """单例获取"""
        s1 = get_dict_service()
        s2 = get_dict_service()
        assert s1 is s2
    
    def test_is_valid_pinyin_after_build(self):
        """拼音验证（需先运行 build_dict.py）"""
        service = get_dict_service()
        # 如果字典未构建，跳过
        if not service._pinyin_set:
            return
        assert service.is_valid_pinyin("ni") is True
        assert service.is_valid_pinyin("hao") is True
        assert service.is_valid_pinyin("xxx") is False
    
    def test_get_chars_after_build(self):
        """字符查询（需先运行 build_dict.py）"""
        service = get_dict_service()
        if not service._char_dict:
            return
        chars = service.get_chars("ni")
        assert isinstance(chars, list)
        assert "你" in chars or len(chars) > 0
