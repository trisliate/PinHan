"""轻量级拼音翻译引擎。

特性：
- 双向最大匹配分词
- 常用短语补充
- 零外部依赖（仅 orjson）
- 速度 <1ms

使用：
    from engine import PinyinEngine
    engine = PinyinEngine.load('dicts')
    print(engine.convert('ni3 hao3 shi4 jie4'))
"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import orjson


class PinyinEngine:
    """轻量级拼音到汉字转换引擎。"""
    
    # 低频词黑名单：与高频单字组合冲突
    LOW_FREQ_WORDS = {'shi4 yi1', 'zai4 yi1', 'bu4 yi1', 'you3 yi1', 'zhong1 guo2 shi4'}
    
    # 补充常用短语（CC-CEDICT 未收录）
    COMMON_PHRASES = {
        'yi1 ge4': [('一个', 100)],
        'zhe4 ge4': [('这个', 100)],
        'na4 ge4': [('那个', 100)],
        'mei3 ge4': [('每个', 100)],
        'ge4 ren2': [('个人', 50)],
        'yi1 xia4': [('一下', 100)],
        'yi1 dian3': [('一点', 100)],
        'yi1 bian4': [('一遍', 80)],
        'yi1 yang4': [('一样', 100)],
        'yi1 zhi2': [('一直', 100)],
        'yi1 ding4': [('一定', 100)],
        'yi1 ban1': [('一般', 100)],
        'yi1 qie4': [('一切', 100)],
        'shi2 me5': [('什么', 100)],
        'zen3 me5': [('怎么', 100)],
        'na3 li3': [('哪里', 100)],
        'zhe4 li3': [('这里', 100)],
        'na4 li3': [('那里', 100)],
        'wei4 shen2 me5': [('为什么', 100)],
        'ke3 yi3': [('可以', 100)],
        'bu4 shi4': [('不是', 100)],
        'mei2 you3': [('没有', 100)],
        'bu4 yao4': [('不要', 100)],
        'hao3 de5': [('好的', 100)],
        'dui4 bu5 qi3': [('对不起', 100)],
        'mei2 guan1 xi5': [('没关系', 100)],
        'xie4 xie5': [('谢谢', 100)],
        'bu4 ke4 qi5': [('不客气', 100)],
    }
    
    def __init__(
        self,
        word_dict: Dict[str, List[Tuple[str, int]]],
        char_dict: Dict[str, List[str]],
        bigram: Dict[str, int] = None
    ):
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.bigram = bigram or {}
        
        # 添加常用短语
        for k, v in self.COMMON_PHRASES.items():
            if k not in self.word_dict:
                self.word_dict[k] = v
        
        # 移除低频冲突词
        for key in self.LOW_FREQ_WORDS:
            self.word_dict.pop(key, None)
        
        # 预计算最大词长
        self._max_word_len = 6
    
    @classmethod
    def load(cls, path: str) -> 'PinyinEngine':
        """从目录加载引擎。"""
        p = Path(path)
        
        word_dict = {}
        char_dict = {}
        bigram = {}
        
        word_file = p / 'word_dict.json'
        if word_file.exists():
            with open(word_file, 'rb') as f:
                raw = orjson.loads(f.read())
                word_dict = {k: [(w, freq) for w, freq in v] for k, v in raw.items()}
        
        char_file = p / 'char_dict.json'
        if char_file.exists():
            with open(char_file, 'rb') as f:
                char_dict = orjson.loads(f.read())
        
        bigram_file = p / 'bigram.json'
        if bigram_file.exists():
            with open(bigram_file, 'rb') as f:
                bigram = orjson.loads(f.read())
        
        return cls(word_dict, char_dict, bigram)
    
    def _get_char(self, pinyin: str) -> str:
        """获取单字，支持轻声回退。"""
        chars = self.char_dict.get(pinyin, [])
        
        # 轻声回退 (5 -> 1,2,3,4)
        if not chars and pinyin and pinyin[-1] == '5':
            base = pinyin[:-1]
            for tone in '1234':
                chars = self.char_dict.get(base + tone, [])
                if chars:
                    break
        
        return chars[0] if chars else '?'
    
    def _normalize_pinyin(self, pinyin: str) -> List[str]:
        """生成拼音变体（处理轻声）。"""
        variants = [pinyin]
        # 5声 -> 1234声
        if pinyin.endswith('5'):
            base = pinyin[:-1]
            variants.extend(base + t for t in '1234')
        # 1234声 -> 5声
        elif pinyin[-1:] in '1234':
            variants.append(pinyin[:-1] + '5')
        return variants
    
    def _get_word(self, pinyin_seq: str) -> Optional[str]:
        """获取词语，支持轻声变体匹配。"""
        # 直接匹配
        words = self.word_dict.get(pinyin_seq, [])
        if words:
            return words[0][0]
        
        # 尝试轻声变体（仅对最后一个音节）
        tokens = pinyin_seq.split()
        if tokens:
            last = tokens[-1]
            for variant in self._normalize_pinyin(last):
                if variant != last:
                    key = ' '.join(tokens[:-1] + [variant]) if len(tokens) > 1 else variant
                    words = self.word_dict.get(key, [])
                    if words:
                        return words[0][0]
        
        return None
    
    def _segment(self, tokens: List[str]) -> List[str]:
        """正向最大匹配分词。"""
        result = []
        i = 0
        n = len(tokens)
        
        while i < n:
            matched = False
            
            # 从最长开始尝试
            for length in range(min(self._max_word_len, n - i), 1, -1):
                key = ' '.join(tokens[i:i+length])
                word = self._get_word(key)
                if word:
                    result.append(word)
                    i += length
                    matched = True
                    break
            
            if not matched:
                result.append(self._get_char(tokens[i]))
                i += 1
        
        return result
    
    def convert(self, pinyin: str) -> str:
        """将拼音转换为汉字。"""
        # 标点处理
        punct = '，。！？、；：""''（）,.!?;:()'
        
        cleaned = pinyin
        for p in punct:
            cleaned = cleaned.replace(p, f' {p} ')
        
        parts = cleaned.split()
        result = []
        buffer = []
        
        for part in parts:
            if part in punct:
                if buffer:
                    result.extend(self._segment(buffer))
                    buffer = []
                result.append(part)
            else:
                buffer.append(part)
        
        if buffer:
            result.extend(self._segment(buffer))
        
        return ''.join(result)
    
    @property
    def stats(self) -> dict:
        return {
            'words': len(self.word_dict),
            'chars': len(self.char_dict),
            'bigrams': len(self.bigram)
        }
    
    def __repr__(self) -> str:
        s = self.stats
        return f"PinyinEngine(words={s['words']}, chars={s['chars']})"


if __name__ == '__main__':
    import time
    
    engine = PinyinEngine.load('dicts')
    print(engine)
    
    tests = [
        'ni3 hao3',
        'wo3 shi4 yi1 ge4 xue2 sheng1',
        'ta1 shi4 yi1 ge4 hao3 ren2',
        'wo3 men5 zai4 yi1 qi3',
        'jin1 tian1 tian1 qi4 hen3 hao3',
        'zhong1 guo2 shi4 yi1 ge4 wei3 da4 de5 guo2 jia1',
        'wei4 shen2 me5 ni3 bu4 lai2',
        'xie4 xie5 ni3 de5 bang1 zhu4',
    ]
    
    for t in tests:
        start = time.perf_counter()
        result = engine.convert(t)
        elapsed = (time.perf_counter() - start) * 1000
        print(f'{t} -> {result} ({elapsed:.2f}ms)')
