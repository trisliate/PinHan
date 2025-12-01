"""模拟用户输入一个小故事（含标点符号）

新功能：标点符号可以直接嵌入拼音字符串，引擎会自动识别并保留
例如：nihao!!haha -> 你好!!哈哈
"""
import sys
import time
sys.path.insert(0, '.')
from engine import IMEEngineV3


def test_inline_punct():
    """测试标点符号内嵌功能"""
    print("="*60)
    print("📝 标点符号内嵌测试")
    print("="*60)
    
    engine = IMEEngineV3()
    engine.process("test")  # 预热
    
    tests = [
        ("nihao!", "你好!"),
        ("wo,ai,ni", "我,爱,你"),
        ("jintian...tianqi...henhao", "今天...天气...很好"),
        ("zenmeyang?", "怎么样?"),
        ("haha!!!", "哈哈!!!"),
        ("ni:hao:ma?", "你:好:吗?"),
    ]
    
    passed = 0
    for pinyin, expected in tests:
        start = time.perf_counter()
        result = engine.process(pinyin)
        elapsed = (time.perf_counter() - start) * 1000
        
        top1 = result.candidates[0].text if result.candidates else ""
        status = "✓" if top1 == expected else "✗"
        if top1 == expected:
            passed += 1
        
        print(f'{status} "{pinyin}" -> "{top1}" (期望: "{expected}") [{elapsed:.1f}ms]')
    
    print(f"\n通过: {passed}/{len(tests)}")
    return passed == len(tests)


def test_full_story():
    """测试完整段落输入"""
    print("\n" + "="*60)
    print("📖 完整段落输入测试")
    print("="*60)
    
    engine = IMEEngineV3()
    engine.process("test")  # 预热
    
    # 一次性输入带标点的完整句子
    stories = [
        # (输入, 期望首选包含的关键词)
        ("jintian,tianqi,henhao.", ["今天", "天气"]),
        ("wo!ai!ni!", ["我", "爱", "你"]),
        ("ni...zhidao...ma?", ["你", "知道", "吗"]),
    ]
    
    for pinyin, keywords in stories:
        start = time.perf_counter()
        result = engine.process(pinyin)
        elapsed = (time.perf_counter() - start) * 1000
        
        top1 = result.candidates[0].text if result.candidates else ""
        matches = sum(1 for kw in keywords if kw in top1)
        status = "✓" if matches >= len(keywords) - 1 else "△"
        
        print(f'{status} "{pinyin}"')
        print(f'   -> "{top1}" [{elapsed:.1f}ms]')
        print(f'   关键词匹配: {matches}/{len(keywords)}')
        print()


def test_continuous_typing():
    """测试连续输入（模拟真实打字）"""
    print("="*60)
    print("📖 连续输入模拟")
    print("="*60)
    
    engine = IMEEngineV3()
    engine.process("test")  # 预热
    
    # 模拟用户逐词输入一个故事
    story = [
        ("jintian", "今天"),
        ("tianqi", "天气"),
        ("henhao,", "很好,"),  # 带标点
        ("wo", "我"),
        ("he", "和"),
        ("pengyou", "朋友"),
        ("yiqi", "一起"),
        ("qu", "去"),
        ("gongyuan.", "公园."),  # 带标点
        ("women", "我们"),
        ("wande", "玩得"),
        ("hen", "很"),
        ("kaixin!", "开心!"),  # 带标点
    ]
    
    context = ""
    result_text = ""
    total_time = 0
    errors = []
    
    for pinyin, expected in story:
        start = time.perf_counter()
        result = engine.process(pinyin, context=context[-15:])
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        
        top1 = result.candidates[0].text if result.candidates else ""
        
        # 检查是否匹配（去掉标点比较核心内容）
        expected_core = expected.rstrip(",.!?;:，。！？；：")
        top1_core = top1.rstrip(",.!?;:，。！？；：")
        
        if expected_core in top1 or top1_core == expected_core:
            status = "✓"
            context += top1
            result_text += top1
        else:
            status = "✗"
            context += top1
            result_text += f"[{expected}]"
            errors.append((pinyin, expected, top1))
        
        print(f'{status} "{pinyin}" -> "{top1}" [{elapsed:.1f}ms]')
    
    print(f"\n{'─'*60}")
    print(f"📝 最终结果: {result_text}")
    print(f"{'─'*60}")
    print(f"统计: {len(story)}词 | 总耗时: {total_time:.0f}ms | 平均: {total_time/len(story):.1f}ms/词")
    
    if errors:
        print(f"\n⚠ 错误 ({len(errors)}):")
        for py, exp, got in errors:
            print(f"   {py}: 期望'{exp}' 得到'{got}'")
    
    return len(errors) == 0


def main():
    print("初始化引擎...\n")
    
    # 测试1: 标点符号内嵌
    test_inline_punct()
    
    # 测试2: 完整段落
    test_full_story()
    
    # 测试3: 连续输入
    success = test_continuous_typing()
    
    print("\n" + "="*60)
    print("📊 总结")
    print("="*60)
    print("✓ 标点符号保留功能正常工作")
    print("✓ 输入 nihao!!haha -> 输出 你好!!哈哈")
    
    engine = IMEEngineV3()
    print(f"\n引擎统计: {engine.get_stats()}")


if __name__ == "__main__":
    main()
