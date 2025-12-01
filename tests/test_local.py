"""本地引擎测试 (无需 API 服务)"""
import sys
import time
sys.path.insert(0, '.')

from engine import IMEEngineV3

def main():
    print("初始化引擎...")
    engine = IMEEngineV3()
    print()

    # 预热
    engine.process("nihao")

    test_cases = [
        # (pinyin, context, expected, description)
        ("jintian", "", "今天", "基础词组"),
        ("tianqi", "今天", "天气", "上下文连贯"),
        ("shi", "我做", "事", "上下文消歧义 (做+事)"),
        ("shengme", "", "什么", "模糊音纠错 (sheng->shen)"),
        ("wo", "", "我", "单字"),
        ("renzhen", "", "认真", "双字词"),
    ]

    print("="*60)
    print("测试结果")
    print("="*60)
    
    passed = 0
    total = len(test_cases)
    
    for pinyin, context, expected, desc in test_cases:
        start = time.perf_counter()
        result = engine.process(pinyin, context=context)
        elapsed = (time.perf_counter() - start) * 1000
        
        candidates = result.candidates
        texts = [c.text for c in candidates]
        
        if expected in texts:
            rank = texts.index(expected) + 1
            status = "✓" if rank <= 3 else "△"
            if rank <= 3:
                passed += 1
            print(f"{status} [{desc}] '{pinyin}' (ctx='{context}')")
            print(f"   期望: {expected} | 排名: #{rank} | 耗时: {elapsed:.1f}ms")
            print(f"   候选: {texts[:5]}")
        else:
            print(f"✗ [{desc}] '{pinyin}' (ctx='{context}')")
            print(f"   期望: {expected} | 未找到!")
            print(f"   候选: {texts[:5]}")
        print()
    
    print("="*60)
    print(f"通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
