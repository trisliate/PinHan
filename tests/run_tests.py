#!/usr/bin/env python3
"""
测试运行脚本：运行所有测试并生成报告。
"""
import sys
import unittest
import json
from pathlib import Path
from datetime import datetime
from io import StringIO

def run_all_tests(verbose: int = 2):
    """运行所有测试."""
    test_dir = Path(__file__).parent
    
    # 发现并加载所有测试
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 运行测试
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbose)
    result = runner.run(suite)
    
    # 输出结果
    print(stream.getvalue())
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': result.testsRun,
        'successes': result.testsRun - len(result.failures) - len(result.errors),
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
    }
    
    # 详细信息
    if result.failures:
        report['failure_details'] = [
            {
                'test': str(test),
                'traceback': traceback
            }
            for test, traceback in result.failures
        ]
    
    if result.errors:
        report['error_details'] = [
            {
                'test': str(test),
                'traceback': traceback
            }
            for test, traceback in result.errors
        ]
    
    return result, report


def main():
    """主函数."""
    print("="*70)
    print("🧪 运行测试套件")
    print("="*70)
    print()
    
    result, report = run_all_tests(verbose=2)
    
    print()
    print("="*70)
    print("📊 测试总结")
    print("="*70)
    print(f"总测试数: {report['tests_run']}")
    print(f"成功: {report['successes']}")
    print(f"失败: {report['failures']}")
    print(f"错误: {report['errors']}")
    print(f"跳过: {report['skipped']}")
    print(f"成功率: {report['success_rate']:.1f}%")
    print("="*70)
    
    # 保存报告
    report_file = Path(__file__).parent / 'test_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 测试报告已保存到: {report_file}")
    
    # 返回退出码
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
