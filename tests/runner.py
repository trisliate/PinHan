"""
PinHan 测试运行器

用法:
    python tests/runner.py           # 冒烟测试 (快速)
    python tests/runner.py --smoke   # 冒烟测试
    python tests/runner.py --full    # 完整测试
    python tests/runner.py --save    # 保存测试报告
"""
import sys
import argparse

sys.path.insert(0, '.')

from tests.config import TestConfig, TestLevel
from tests.test_comprehensive import ComprehensiveTestRunner


def main():
    parser = argparse.ArgumentParser(description="PinHan 测试运行器")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试 (默认)")
    parser.add_argument("--full", action="store_true", help="完整测试")
    parser.add_argument("--save", action="store_true", help="保存测试报告")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--iterations", type=int, default=100, help="性能测试迭代次数")
    
    args = parser.parse_args()
    
    # 默认冒烟测试
    if not any([args.smoke, args.full]):
        args.smoke = True
    
    config = TestConfig(
        level=TestLevel.FULL if args.full else TestLevel.SMOKE,
        verbose=not args.quiet,
        save_report=args.save,
        performance_iterations=args.iterations
    )
    
    runner = ComprehensiveTestRunner(config)
    
    if args.full:
        report = runner.run_full()
    else:
        report = runner.run_smoke()
    
    sys.exit(0 if report.overall_rate >= 0.7 else 1)


if __name__ == "__main__":
    main()
