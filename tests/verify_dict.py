"""
字典验证脚本

测试字典查询功能，验证各种场景
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.dictionary import DictionaryService, PinyinUtils


def test_single_char():
    """测试单字查询"""
    print("=" * 50)
    print("【测试1】单字查询")
    print("=" * 50)
    
    s = DictionaryService()
    
    test_cases = [
        ("ni", "你"),
        ("hao", "好"),
        ("de", "的"),
        ("shi", "是"),
        ("wo", "我"),
        ("zhong", "中"),
        ("guo", "国"),
    ]
    
    for pinyin, expected in test_cases:
        chars = s.get_chars(pinyin)[:5]
        found = expected in chars
        status = "✓" if found else "✗"
        print(f"  {pinyin} → {chars}  期望'{expected}' {status}")
    
    print()


def test_short_word():
    """测试短词查询"""
    print("=" * 50)
    print("【测试2】短词查询（2-3字）")
    print("=" * 50)
    
    s = DictionaryService()
    
    test_cases = [
        (["ni", "hao"], "你好"),
        (["zhong", "guo"], "中国"),
        (["wo", "men"], "我们"),
        (["shi", "jie"], "世界"),
        (["xie", "xie"], "谢谢"),
        (["dui", "bu", "qi"], "对不起"),
        (["mei", "guan", "xi"], "没关系"),
    ]
    
    for pinyins, expected in test_cases:
        words = s.get_words(pinyins)
        candidates = s.get_candidates(pinyins, top_k=5)
        found = expected in words or any(c[0] == expected for c in candidates)
        status = "✓" if found else "✗"
        print(f"  {' '.join(pinyins)} → 词组:{words[:3]}  候选:{[c[0] for c in candidates[:3]]}  期望'{expected}' {status}")
    
    print()


def test_long_word():
    """测试长词查询"""
    print("=" * 50)
    print("【测试3】长词查询（4字以上）")
    print("=" * 50)
    
    s = DictionaryService()
    
    test_cases = [
        (["zhong", "hua", "ren", "min", "gong", "he", "guo"], "中华人民共和国"),
        (["bei", "jing", "shi"], "北京市"),
        (["da", "xue", "sheng"], "大学生"),
        (["ji", "suan", "ji"], "计算机"),
        (["hu", "lian", "wang"], "互联网"),
    ]
    
    for pinyins, expected in test_cases:
        words = s.get_words(pinyins)
        found = expected in words
        status = "✓" if found else "✗"
        
        # 如果完整词没找到，尝试拆分
        if not found:
            # 检查是否能组合出来
            candidates = s.get_candidates(pinyins, top_k=10)
            found_in_candidates = any(c[0] == expected for c in candidates)
            if found_in_candidates:
                status = "✓(组合)"
            else:
                status = "✗(需切分)"
        
        print(f"  {' '.join(pinyins)} → 词组:{words[:2] if words else '无'}  期望'{expected}' {status}")
    
    print()


def test_frequency_order():
    """测试频率排序"""
    print("=" * 50)
    print("【测试4】频率排序验证")
    print("=" * 50)
    
    s = DictionaryService()
    
    # 测试同音字排序
    test_cases = [
        ("de", ["的", "地", "得"]),  # 的 应该最高频
        ("shi", ["是", "时", "事"]),  # 是 应该最高频
        ("ta", ["他", "她", "它"]),  # 他 应该较高频
        ("zhi", ["之", "只", "知"]),
    ]
    
    for pinyin, expected_order in test_cases:
        chars = s.get_chars(pinyin)[:5]
        # 检查期望的字是否按顺序出现
        positions = []
        for exp in expected_order:
            if exp in chars:
                positions.append(chars.index(exp))
            else:
                positions.append(999)
        
        is_ordered = positions == sorted(positions)
        status = "✓" if is_ordered else "✗"
        print(f"  {pinyin} → {chars}  期望顺序:{expected_order} {status}")
    
    print()


def test_word_frequency():
    """测试词频"""
    print("=" * 50)
    print("【测试5】词频验证")
    print("=" * 50)
    
    s = DictionaryService()
    
    high_freq_words = ["中国", "我们", "什么", "没有", "可以", "因为", "所以"]
    
    for word in high_freq_words:
        freq = s.get_word_freq(word)
        status = "✓" if freq > 0 else "✗"
        print(f"  '{word}' 频率: {freq:.6f} {status}")
    
    print()


def test_statistics():
    """统计信息"""
    print("=" * 50)
    print("【统计】字典规模")
    print("=" * 50)
    
    s = DictionaryService()
    
    print(f"  拼音数: {len(s.get_all_pinyin())}")
    print(f"  单字字典: {len(s._char_dict)} 个拼音")
    print(f"  词组字典: {len(s._word_dict)} 个拼音组合")
    print(f"  字频表: {len(s._char_freq)} 条")
    print(f"  词频表: {len(s._word_freq)} 条")
    
    # 统计有效词频
    valid_word_freq = sum(1 for f in s._word_freq.values() if f > 1e-7)
    print(f"  有效词频(>1e-7): {valid_word_freq} 条")
    
    print()


def main():
    print("\n" + "=" * 50)
    print("       字典功能验证")
    print("=" * 50 + "\n")
    
    test_statistics()
    test_single_char()
    test_short_word()
    test_long_word()
    test_frequency_order()
    test_word_frequency()
    
    print("=" * 50)
    print("验证完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
