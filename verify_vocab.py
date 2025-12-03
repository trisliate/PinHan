#!/usr/bin/env python3
"""验证新增词汇"""

import json

# 读取词典
with open('data/dicts/word_dict.json', 'r', encoding='utf-8') as f:
    word_dict = json.load(f)

# 检查教育和 IT 相关词汇
test_words = {
    '教育': 'jiao yu',
    '学位': 'xue wei',
    '初始化': 'chu shi hua',
    '数组': 'shu zu',
    '字符串': 'zi fu chuan',
    '算法': 'suan fa',
    '机器学习': 'ji qi xue xi',
    '深度学习': 'shen du xue xi',
}

print("=== 教育和 IT 词汇检查 ===")
found_count = 0
for word, expected_py in test_words.items():
    if expected_py in word_dict:
        chars = word_dict[expected_py]
        if word in chars:
            print(f"✓ {word} ({expected_py}): 已包含")
            found_count += 1
        else:
            print(f"△ {expected_py}: 拼音存在但 {word} 不在 Top3: {chars[:3]}")
    else:
        print(f"✗ {word} ({expected_py}): 未找到拼音")

print(f"\n=== 统计信息 ===")
print(f"找到的词汇: {found_count}/{len(test_words)}")
print(f"词典拼音组合数: {len(word_dict):,}")
print(f"词典总词条数: {sum(len(v) for v in word_dict.values()):,}")

# 显示来自 THUOCL 的 IT 词汇示例
print(f"\n=== THUOCL IT 词汇示例 ===")
it_keywords = ['编程', '数据库', '网络', '操作系统', '软件', '硬件']
for kw in it_keywords[:5]:
    # 简单的拼音转换
    import pypinyin
    py = ' '.join(pypinyin.lazy_pinyin(kw, style=pypinyin.Style.NORMAL))
    if py in word_dict and kw in word_dict[py]:
        print(f"✓ {kw}")
    else:
        print(f"△ {kw} (可能被其他词覆盖)")
