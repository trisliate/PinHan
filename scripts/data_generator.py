"""
训练数据生成器

从清洗后的中文语料生成 (拼音, 汉字) 配对数据
"""

import os
import re
import orjson
from collections import defaultdict
from typing import List, Dict, Tuple, Iterator, Optional
from pypinyin import lazy_pinyin, Style
import jieba


# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DICT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dicts')


def hanzi_to_pinyin(text: str) -> List[str]:
    """
    汉字转拼音（无声调）
    
    Args:
        text: 汉字文本
    
    Returns:
        拼音列表，如 ["ni", "hao"]
    """
    # 使用 pypinyin 转换
    pinyins = lazy_pinyin(text, style=Style.NORMAL)
    # 处理 ü → v
    pinyins = [p.replace('ü', 'v') for p in pinyins]
    return pinyins


def generate_training_pair(sentence: str) -> Optional[Dict]:
    """
    从句子生成训练样本
    
    Args:
        sentence: 中文句子
    
    Returns:
        {"pinyin": "ni hao", "hanzi": "你好"} 或 None
    """
    # 过滤非中文字符
    hanzi = re.sub(r'[^\u4e00-\u9fff]', '', sentence)
    if len(hanzi) < 2:
        return None
    
    # 转换拼音
    pinyins = hanzi_to_pinyin(hanzi)
    
    # 验证长度一致
    if len(pinyins) != len(hanzi):
        return None
    
    return {
        "pinyin": " ".join(pinyins),
        "hanzi": hanzi
    }


def extract_words(sentence: str) -> List[str]:
    """使用 jieba 分词"""
    words = jieba.lcut(sentence)
    # 过滤非中文词
    words = [w for w in words if re.match(r'^[\u4e00-\u9fff]+$', w)]
    return words


class DataGenerator:
    """训练数据生成器"""
    
    def __init__(self):
        self.word_count: Dict[str, int] = defaultdict(int)  # 词频统计
        self.bigram_count: Dict[str, int] = defaultdict(int)  # Bigram 统计
        self.word_pinyin: Dict[str, List[str]] = defaultdict(list)  # 词 → 拼音映射
        self.sample_count = 0
        
    def process_sentence(self, sentence: str) -> Optional[Dict]:
        """
        处理单个句子
        
        返回训练样本，同时更新统计信息
        """
        # 生成训练对
        pair = generate_training_pair(sentence)
        if pair is None:
            return None
        
        self.sample_count += 1
        
        # 分词统计
        words = extract_words(sentence)
        for word in words:
            if len(word) >= 2:  # 只统计多字词
                self.word_count[word] += 1
                # 记录词的拼音
                py = " ".join(hanzi_to_pinyin(word))
                if word not in self.word_pinyin.get(py, []):
                    self.word_pinyin[py].append(word)
        
        # Bigram 统计（拼音层面）
        pinyins = pair["pinyin"].split()
        for i in range(len(pinyins) - 1):
            bigram = f"{pinyins[i]}|{pinyins[i+1]}"
            self.bigram_count[bigram] += 1
        
        return pair
    
    def process_sentences(
        self, 
        sentences: Iterator[str],
        output_path: str,
        max_samples: Optional[int] = None
    ) -> int:
        """
        批量处理句子并写入文件
        
        Args:
            sentences: 句子迭代器
            output_path: 输出文件路径（JSONL 格式）
            max_samples: 最大样本数
        
        Returns:
            处理的样本数
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        count = 0
        with open(output_path, 'wb') as f:
            for sentence in sentences:
                if max_samples and count >= max_samples:
                    break
                
                pair = self.process_sentence(sentence)
                if pair:
                    f.write(orjson.dumps(pair) + b'\n')
                    count += 1
                    
                    if count % 50000 == 0:
                        print(f"  已生成: {count} 条训练数据")
        
        print(f"训练数据生成完成: {count} 条")
        return count
    
    def save_word_dict(self, min_freq: int = 2):
        """
        保存词组字典
        
        Args:
            min_freq: 最小词频阈值
        """
        # 过滤低频词
        filtered = {
            py: words for py, words in self.word_pinyin.items()
            if any(self.word_count.get(w, 0) >= min_freq for w in words)
        }
        
        # 按词频排序每个拼音对应的词
        for py in filtered:
            filtered[py] = sorted(
                filtered[py],
                key=lambda w: self.word_count.get(w, 0),
                reverse=True
            )
        
        path = os.path.join(DICT_DIR, 'word_dict.json')
        with open(path, 'wb') as f:
            f.write(orjson.dumps(filtered, option=orjson.OPT_INDENT_2))
        print(f"词组字典已保存: {len(filtered)} 个拼音组合")
    
    def save_word_freq(self, min_freq: int = 2):
        """保存词频表"""
        # 过滤并归一化
        total = sum(self.word_count.values())
        freq = {
            word: count / total
            for word, count in self.word_count.items()
            if count >= min_freq and len(word) >= 2
        }
        
        path = os.path.join(DICT_DIR, 'word_freq.json')
        with open(path, 'wb') as f:
            f.write(orjson.dumps(freq, option=orjson.OPT_INDENT_2))
        print(f"词频表已保存: {len(freq)} 个词")
    
    def save_bigram(self):
        """保存 Bigram 统计"""
        path = os.path.join(DICT_DIR, 'bigram.json')
        with open(path, 'wb') as f:
            f.write(orjson.dumps(dict(self.bigram_count), option=orjson.OPT_INDENT_2))
        print(f"Bigram 统计已保存: {len(self.bigram_count)} 个组合")
    
    def save_all(self, min_freq: int = 2):
        """保存所有统计数据"""
        self.save_word_dict(min_freq)
        self.save_word_freq(min_freq)
        self.save_bigram()


if __name__ == '__main__':
    # 测试
    gen = DataGenerator()
    
    test_sentences = [
        "你好世界",
        "中国人民共和国",
        "今天天气很好",
        "我们一起学习",
    ]
    
    for sent in test_sentences:
        pair = gen.process_sentence(sent)
        if pair:
            print(f"{pair['hanzi']} → {pair['pinyin']}")
    
    print(f"\n词频统计: {dict(gen.word_count)}")
    print(f"Bigram: {dict(list(gen.bigram_count.items())[:5])}")
