"""
PinHan 核心测试：字典加载、解码功能。
"""
import unittest
from pathlib import Path

from model.core import PinyinDict


class TestPinyinDict(unittest.TestCase):
    """测试 PinyinDict 类。"""
    
    @classmethod
    def setUpClass(cls):
        """加载字典（只执行一次）。"""
        dict_path = Path(__file__).parent.parent / 'dicts'
        if dict_path.exists() and (dict_path / 'word_dict.json').exists():
            cls.pdict = PinyinDict.load(dict_path)
            cls.has_dict = True
        else:
            cls.pdict = None
            cls.has_dict = False
    
    def test_dict_loading(self):
        """测试字典加载。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        self.assertIsNotNone(self.pdict)
        self.assertGreater(self.pdict.word_count, 100000)  # 至少10万词
        self.assertGreater(self.pdict.char_count, 1000)    # 至少1000拼音
    
    def test_get_chars(self):
        """测试单字查询。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        # 常见拼音
        self.assertIn('你', self.pdict.get_chars('ni3'))
        self.assertIn('好', self.pdict.get_chars('hao3'))
        self.assertIn('是', self.pdict.get_chars('shi4'))
        
        # 常用字排前面
        self.assertEqual(self.pdict.get_chars('shi4')[0], '是')
        self.assertEqual(self.pdict.get_chars('de5')[0], '的')
    
    def test_get_words(self):
        """测试词语查询。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        self.assertIn('你好', self.pdict.get_words('ni3 hao3'))
        self.assertIn('中国', self.pdict.get_words('zhong1 guo2'))
        self.assertIn('一起', self.pdict.get_words('yi1 qi3'))
        self.assertIn('今天', self.pdict.get_words('jin1 tian1'))
    
    def test_neutral_tone(self):
        """测试轻声(5声)回退。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        # ma5 应回退到其他声调
        chars = self.pdict.get_chars('ma5')
        self.assertGreater(len(chars), 0)
        self.assertTrue(any(c in chars for c in ['么', '吗', '嘛', '妈']))
    
    def test_decode_word(self):
        """测试词语解码。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        self.assertEqual(self.pdict.decode(['ni3', 'hao3']), '你好')
        self.assertEqual(self.pdict.decode(['zhong1', 'guo2']), '中国')
        self.assertEqual(self.pdict.decode(['yi1', 'qi3']), '一起')
    
    def test_decode_sentence(self):
        """测试句子解码。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        tokens = 'jin1 tian1 tian1 qi4 hen3 hao3'.split()
        self.assertEqual(self.pdict.decode(tokens), '今天天气很好')
        
        tokens = 'wo3 ai4 zhong1 guo2'.split()
        self.assertEqual(self.pdict.decode(tokens), '我爱中国')
    
    def test_decode_with_neutral_tone(self):
        """测试带轻声解码。"""
        if not self.has_dict:
            self.skipTest("字典文件不存在")
        
        tokens = 'ni3 hao3 ma5'.split()
        result = self.pdict.decode(tokens)
        self.assertNotIn('?', result)  # 不应有未知字符


class TestPinyinDictEdgeCases(unittest.TestCase):
    """边界情况测试。"""
    
    def test_empty_dict(self):
        """测试空字典。"""
        pdict = PinyinDict({}, {}, {})
        self.assertEqual(pdict.word_count, 0)
        self.assertEqual(pdict.char_count, 0)
        self.assertEqual(pdict.decode(['ni3']), '?')
    
    def test_unknown_pinyin(self):
        """测试未知拼音。"""
        pdict = PinyinDict({}, {'ni3': ['你']}, {})
        self.assertEqual(pdict.decode(['xyz']), '?')
        self.assertEqual(pdict.decode(['ni3']), '你')


if __name__ == '__main__':
    unittest.main()
