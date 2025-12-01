"""上下文消歧义测试"""
import sys
sys.path.insert(0, '.')
from engine import IMEEngineV3

engine = IMEEngineV3()

# 上下文测试用例
tests = [
    # (pinyin, context, expected, description)
    ('shi', '我做', '事', '做+事'),
    ('shi', '什么', '事', '什么事'),
    ('shi', '这', '是', '这是'),
    ('shi', '历', '史', '历史'),
    ('shi', '老', '师', '老师'),
    ('de', '我', '的', '我的'),
    ('de', '目', '的', '目的'),
    ('li', '道', '理', '道理'),
    ('li', '美', '丽', '美丽'),
    ('guo', '中', '国', '中国'),
    ('guo', '苹', '果', '苹果'),
    ('ti', '问', '题', '问题'),
]

print('上下文消歧义测试')
print('='*60)

passed = 0
for pinyin, ctx, expected, desc in tests:
    result = engine.process(pinyin, context=ctx)
    texts = [c.text for c in result.candidates]
    
    if expected in texts:
        rank = texts.index(expected) + 1
        status = '✓' if rank <= 3 else '△'
        if rank <= 3:
            passed += 1
    else:
        status = '✗'
        rank = -1
    
    print(f'{status} [{desc}] ctx="{ctx}" + "{pinyin}" -> {expected} (#{rank})')
    print(f'   候选: {texts[:5]}')

print('='*60)
print(f'通过率: {passed}/{len(tests)} ({passed/len(tests)*100:.0f}%)')
