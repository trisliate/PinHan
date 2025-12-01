"""
IME-SLM v2 引擎集成测试
测试分层策略和性能
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine_v2 import create_engine_v2
from config import EngineConfig


def test_v2_engine():
    """测试 v2 引擎"""
    print("=" * 70)
    print("IME-SLM v2 引擎集成测试")
    print("=" * 70)
    
    config = EngineConfig(top_k=10)
    engine = create_engine_v2(config)
    
    # 测试用例
    test_cases = [
        # (拼音, 期望结果列表, 类别)
        # 简单词 - 应该走 Level 2
        ("nihao", ["你好"], "简单词"),
        ("xie xie", ["谢谢"], "简单词"),
        ("zaijian", ["再见"], "简单词"),
        
        # 简单句 - 短输入走 Level 2
        ("woaini", ["我爱你"], "简单句"),
        ("henhao", ["很好"], "简单句"),
        
        # 长句 - 走 Level 3
        ("jintiantianqihenhao", ["今天天气很好"], "长句"),
        ("woxiangchifan", ["我想吃饭"], "长句"),
        ("mingtianjiandao", ["明天见到", "明天见"], "长句"),
        
        # 上下文测试
        ("shi", ["是", "时", "事"], "单字"),
        ("zhongguo", ["中国"], "简单词"),
        
        # 缓存测试（重复输入）
        ("nihao", ["你好"], "缓存测试"),
        ("nihao", ["你好"], "缓存测试"),
    ]
    
    # 统计
    results = []
    level_counts = {1: 0, 2: 0, 3: 0}
    total_time = 0
    
    for pinyin, expected_list, category in test_cases:
        result = engine.process(pinyin)
        
        candidates = [c.text for c in result.candidates]
        level = result.metadata.get('level', 0)
        elapsed = result.metadata.get('elapsed_ms', 0)
        
        # 检查是否有任一期望结果在候选中
        passed = any(exp in candidates for exp in expected_list)
        
        level_counts[level] = level_counts.get(level, 0) + 1
        total_time += elapsed
        
        status = "✓" if passed else "✗"
        results.append((pinyin, expected_list, candidates[:3], passed, level, elapsed, category))
        
        print(f"\n[{status}] {category}: {pinyin}")
        print(f"    期望: {expected_list}")
        print(f"    实际: {candidates[:5]}")
        print(f"    Level: {level} | 耗时: {elapsed:.2f}ms")
    
    # 统计汇总
    passed_count = sum(1 for r in results if r[3])
    total_count = len(results)
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"总计: {passed_count}/{total_count} ({100*passed_count/total_count:.1f}%)")
    print(f"\n层级分布:")
    print(f"  Level 1 (缓存): {level_counts.get(1, 0)} 次")
    print(f"  Level 2 (词典): {level_counts.get(2, 0)} 次")
    print(f"  Level 3 (模型): {level_counts.get(3, 0)} 次")
    print(f"\n平均耗时: {total_time/total_count:.2f}ms")
    
    # 分类统计
    categories = {}
    for r in results:
        cat = r[6]
        if cat not in categories:
            categories[cat] = {'passed': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r[3]:
            categories[cat]['passed'] += 1
    
    print(f"\n分类通过率:")
    for cat, stats in categories.items():
        pct = 100 * stats['passed'] / stats['total']
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    # 验证缓存功能
    print(f"\n缓存命中率: {engine.result_cache.hit_rate*100:.1f}%")
    
    return passed_count == total_count


def test_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    
    config = EngineConfig(top_k=10)
    engine = create_engine_v2(config)
    
    # 预热
    for _ in range(5):
        engine.process("nihao")
    
    # 短输入测试
    short_inputs = ["nihao", "xie xie", "henhao", "zaijian", "woaini"]
    
    start = time.perf_counter()
    for _ in range(100):
        for inp in short_inputs:
            engine.process(inp)
    short_time = (time.perf_counter() - start) * 1000 / 500
    
    print(f"短输入 (≤6字符) 平均耗时: {short_time:.3f}ms")
    
    # 长输入测试
    long_inputs = [
        "jintiantianqihenhao",
        "woxiangchifan",
        "mingtianzaihuijia",
        "zhegexiangmuhenbang",
    ]
    
    # 清空缓存
    engine.result_cache.cache.clear()
    
    start = time.perf_counter()
    for inp in long_inputs:
        engine.process(inp)
    long_time = (time.perf_counter() - start) * 1000 / len(long_inputs)
    
    print(f"长输入 (>6字符) 平均耗时: {long_time:.2f}ms")
    
    # 缓存效果测试
    engine.result_cache.cache.clear()
    
    # 第一轮（冷启动）
    start = time.perf_counter()
    for inp in short_inputs * 10:
        engine.process(inp)
    cold_time = (time.perf_counter() - start) * 1000 / 50
    
    # 第二轮（热缓存）
    start = time.perf_counter()
    for inp in short_inputs * 10:
        engine.process(inp)
    hot_time = (time.perf_counter() - start) * 1000 / 50
    
    print(f"\n缓存效果:")
    print(f"  冷启动: {cold_time:.3f}ms/请求")
    print(f"  热缓存: {hot_time:.3f}ms/请求")
    print(f"  加速比: {cold_time/hot_time:.1f}x")
    
    print(f"\n引擎统计: {engine.get_stats()}")


if __name__ == "__main__":
    all_passed = test_v2_engine()
    test_performance_benchmark()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败，请检查")
    print("=" * 70)
