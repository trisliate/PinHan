"""模拟用户连续输入测试"""
import sys
import time
sys.path.insert(0, '.')
from engine import IMEEngineV3

def simulate_typing(engine, sentence_pinyin_pairs, title=""):
    """模拟用户连续输入一个句子"""
    print(f"\n{'='*60}")
    print(f"场景: {title}")
    print('='*60)
    
    context = ""
    result_text = ""
    total_time = 0
    success = True
    
    for pinyin, expected in sentence_pinyin_pairs:
        start = time.perf_counter()
        result = engine.process(pinyin, context=context)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        
        texts = [c.text for c in result.candidates]
        
        if expected in texts:
            rank = texts.index(expected) + 1
            status = "✓" if rank <= 3 else "△"
        else:
            rank = -1
            status = "✗"
            success = False
        
        print(f'{status} "{pinyin}" (ctx="{context[-6:]}...") -> {expected} (#{rank}) [{elapsed:.1f}ms]')
        print(f'   候选: {texts[:5]}')
        
        # 模拟用户选择
        if expected in texts:
            context += expected
            result_text += expected
        else:
            # 如果没找到，用第一个候选
            context += texts[0] if texts else ""
            result_text += f"[{expected}?]"
    
    print(f"\n最终结果: {result_text}")
    print(f"总耗时: {total_time:.1f}ms | 平均: {total_time/len(sentence_pinyin_pairs):.1f}ms/词")
    return success


def main():
    print("初始化引擎...")
    engine = IMEEngineV3()
    
    # 预热
    engine.process("test")
    
    all_passed = True
    
    # 场景1: 简单日常对话
    pairs1 = [
        ("jintian", "今天"),
        ("tianqi", "天气"),
        ("zhen", "真"),
        ("bucuo", "不错"),
    ]
    all_passed &= simulate_typing(engine, pairs1, "今天天气真不错")
    
    # 场景2: 工作场景
    pairs2 = [
        ("wo", "我"),
        ("mingtian", "明天"),
        ("yao", "要"),
        ("kaihui", "开会"),
    ]
    all_passed &= simulate_typing(engine, pairs2, "我明天要开会")
    
    # 场景3: 较长句子
    pairs3 = [
        ("zhege", "这个"),
        ("xiangmu", "项目"),
        ("feichang", "非常"),
        ("zhongyao", "重要"),
    ]
    all_passed &= simulate_typing(engine, pairs3, "这个项目非常重要")
    
    # 场景4: 问句
    pairs4 = [
        ("ni", "你"),
        ("zhidao", "知道"),
        ("zhe", "这"),
        ("shi", "是"),
        ("shengme", "什么"),
        ("yisi", "意思"),
        ("ma", "吗"),
    ]
    all_passed &= simulate_typing(engine, pairs4, "你知道这是什么意思吗")
    
    # 场景5: 常见错误拼音
    pairs5 = [
        ("wo", "我"),
        ("xiang", "想"),
        ("wen", "问"),
        ("ni", "你"),
        ("yige", "一个"),
        ("wenti", "问题"),
    ]
    all_passed &= simulate_typing(engine, pairs5, "我想问你一个问题")
    
    # 场景6: 技术相关
    pairs6 = [
        ("zhe", "这"),
        ("shi", "是"),
        ("yige", "一个"),
        ("pinyin", "拼音"),
        ("shuru", "输入"),
        ("fa", "法"),
    ]
    all_passed &= simulate_typing(engine, pairs6, "这是一个拼音输入法")
    
    print("\n" + "="*60)
    print(f"引擎统计: {engine.get_stats()}")
    print("="*60)
    
    if all_passed:
        print("✓ 所有场景测试通过!")
    else:
        print("△ 部分场景有待改进")


if __name__ == "__main__":
    main()
