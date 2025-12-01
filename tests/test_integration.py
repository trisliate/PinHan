"""
IME-SLM 集成测试

测试引擎在各种场景下的表现
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from engine import create_engine


def run_tests():
    """运行测试用例"""
    print("=" * 60)
    print("IME-SLM 集成测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 初始化引擎
    print("\n正在初始化引擎...")
    engine = create_engine()
    print()
    
    # 测试用例
    test_cases = {
        "简单词": [
            ("nihao", "你好"),
            ("xiexie", "谢谢"),
            ("zaijian", "再见"),
        ],
        "简单句": [
            ("woaini", "我爱你"),
            ("nihao ma", "你好吗"),
            ("chichibufang", None),  # 不期望特定结果
        ],
        "一般词": [
            ("diannao", "电脑"),
            ("shouji", "手机"),
            ("feiji", "飞机"),
        ],
        "一般句": [
            ("jintiantianqihenhao", "今天天气很好"),
            ("woxiangchifan", "我想吃饭"),
            ("niqunaligongzuo", None),
        ],
        "难句": [
            ("womenbixuzaijinnianwanchengrenwu", None),
            ("rengongzhinengfazhanhengyoukuai", None),
        ],
        "纠错测试": [
            ("jintiantianqihenghao", "今天天气很好"),  # heng -> hen
            ("nihoa", "你好"),  # hoa -> hao
        ],
    }
    
    results = {}
    total = 0
    passed = 0
    
    for category, cases in test_cases.items():
        print(f"\n【{category}】")
        print("-" * 40)
        results[category] = []
        
        for pinyin, expected in cases:
            total += 1
            result = engine.process(pinyin)
            
            candidates = [c.text for c in result.candidates[:5]]
            top_candidate = candidates[0] if candidates else ""
            
            # 检查是否通过
            if expected:
                success = expected in candidates[:3]
            else:
                success = len(candidates) > 0  # 只要有候选就算通过
            
            if success:
                passed += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} {pinyin}")
            print(f"    切分: {result.segmented_pinyin}")
            print(f"    候选: {candidates[:5]}")
            if expected:
                print(f"    期望: {expected}")
            
            results[category].append({
                "pinyin": pinyin,
                "expected": expected,
                "candidates": candidates[:5],
                "segmented": result.segmented_pinyin,
                "passed": success,
            })
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"通过: {passed}/{total} ({100*passed/total:.1f}%)")
    
    for category, cases in results.items():
        cat_passed = sum(1 for c in cases if c["passed"])
        print(f"  {category}: {cat_passed}/{len(cases)}")
    
    # 保存结果
    output_path = "test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "pass_rate": passed / total,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    return passed, total


if __name__ == '__main__':
    run_tests()
