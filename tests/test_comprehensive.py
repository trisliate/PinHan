"""
PinHan 全方位仿真测试

模拟真实用户输入场景，覆盖：
1. 日常对话
2. 工作场景
3. 网络用语
4. 专业术语
5. 长句输入
6. 标点混合
7. 边界情况
8. 性能压力

用法:
    python tests/test_comprehensive.py
    python tests/test_comprehensive.py --level smoke
    python tests/test_comprehensive.py --level full --save
"""
import sys
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime

sys.path.insert(0, '.')

from engine import IMEEngineV3

# 测试框架
from tests.config import (
    RunConfig, RunLevel, CaseResult, ScenarioResult, 
    CategoryResult, PerformanceStats, Report,
    get_test_logger, REPORT_DIR
)

# 测试数据
from tests.datasets import (
    DAILY_CONVERSATIONS, WORK_SCENARIOS, INTERNET_SLANG,
    PROFESSIONAL_TERMS, LONG_SENTENCES, SMOKE_TESTS,
    SINGLE_CHARS, CONTEXT_DISAMBIGUATION, FUZZY_PINYIN,
    PUNCTUATION_TESTS, EDGE_CASES, STORY_INPUTS,
    PARAGRAPH_TESTS, LONG_TEXT_TESTS,
    PERFORMANCE_TEST_INPUTS
)


# ============================================
# 测试执行器
# ============================================

class ComprehensiveTestRunner:
    """全方位测试执行器"""
    
    def __init__(self, config: RunConfig = None):
        self.config = config or RunConfig()
        self.logger = get_test_logger()
        self.engine: Optional[IMEEngineV3] = None
        self.report = Report()
        self.start_time = None
        
    def _init_engine(self):
        if self.engine is None:
            self.logger.info("🔧 初始化引擎...")
            self.engine = IMEEngineV3()
            self.engine.process("test")  # 预热
            self.logger.info("")
    
    def _test_pinyin(self, pinyin: str, expected: str, context: str = "") -> CaseResult:
        """执行单个拼音测试"""
        start = time.perf_counter()
        result = self.engine.process(pinyin, context=context)
        elapsed = (time.perf_counter() - start) * 1000
        
        texts = [c.text for c in result.candidates]
        actual = texts[0] if texts else ""
        
        if expected in texts:
            rank = texts.index(expected) + 1
            passed = rank <= self.config.top_n
        else:
            rank = -1
            passed = False
        
        case_id = f"{pinyin}_{context[:5] if context else 'no_ctx'}"
        
        return CaseResult(
            id=case_id,
            pinyin=pinyin,
            expected=expected,
            actual=actual,
            rank=rank,
            passed=passed,
            elapsed_ms=elapsed,
            context=context
        )
    
    def run_scenario(self, name: str, pairs: List[Tuple[str, str]], 
                     category: str = "", use_context: bool = True) -> ScenarioResult:
        """运行场景测试（模拟连续输入）"""
        result = ScenarioResult(name=name, category=category)
        context = ""
        
        for pinyin, expected in pairs:
            ctx = context[-15:] if use_context else ""
            case = self._test_pinyin(pinyin, expected, context=ctx)
            case.category = category
            case.scenario = name
            result.cases.append(case)
            
            # 更新上下文
            if case.passed:
                context += expected
            else:
                context += case.actual
        
        # 日志记录
        if self.config.verbose:
            icon = "✓" if result.rate >= 0.8 else ("△" if result.rate >= 0.5 else "✗")
            self.logger.info(f"{icon} {name}: {result.passed}/{result.total} "
                           f"({result.rate*100:.0f}%) | {result.avg_latency:.1f}ms")
            
            # 显示失败用例
            for case in result.cases:
                if not case.passed:
                    self.logger.debug(f"   ✗ '{case.pinyin}' -> '{case.actual}' (期望: '{case.expected}')")
        
        return result
    
    def run_category(self, name: str, scenarios: List[Tuple[str, List]], 
                     use_context: bool = True) -> CategoryResult:
        """运行分类测试"""
        category = CategoryResult(name=name)
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📂 {name}")
            self.logger.info(f"{'='*60}")
        
        for scenario_name, pairs in scenarios:
            scenario = self.run_scenario(scenario_name, pairs, category=name, use_context=use_context)
            category.scenarios.append(scenario)
        
        self.report.categories[name] = category
        return category
    
    def run_single_char_test(self) -> CategoryResult:
        """单字测试"""
        category = CategoryResult(name="单字识别")
        scenario = ScenarioResult(name="高频单字", category="单字识别")
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📂 单字识别")
            self.logger.info(f"{'='*60}")
        
        for pinyin, expected in SINGLE_CHARS:
            case = self._test_pinyin(pinyin, expected)
            case.category = "单字识别"
            case.scenario = "高频单字"
            scenario.cases.append(case)
        
        category.scenarios.append(scenario)
        
        if self.config.verbose:
            icon = "✓" if scenario.rate >= 0.8 else "△"
            self.logger.info(f"{icon} 高频单字: {scenario.passed}/{scenario.total} ({scenario.rate*100:.0f}%)")
        
        self.report.categories["单字识别"] = category
        return category
    
    def run_context_test(self) -> CategoryResult:
        """上下文消歧义测试"""
        category = CategoryResult(name="上下文消歧义")
        scenario = ScenarioResult(name="易混淆字", category="上下文消歧义")
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📂 上下文消歧义")
            self.logger.info(f"{'='*60}")
        
        for pinyin, context, expected, desc in CONTEXT_DISAMBIGUATION:
            case = self._test_pinyin(pinyin, expected, context=context)
            case.category = "上下文消歧义"
            case.scenario = desc
            scenario.cases.append(case)
            
            if self.config.verbose:
                status = "✓" if case.passed else "✗"
                self.logger.info(f"{status} [{desc}] ctx='{context}' + '{pinyin}' -> '{case.actual}' (期望: '{expected}')")
        
        category.scenarios.append(scenario)
        self.report.categories["上下文消歧义"] = category
        return category
    
    def run_punctuation_test(self) -> CategoryResult:
        """标点符号测试"""
        category = CategoryResult(name="标点符号")
        scenario = ScenarioResult(name="标点混合", category="标点符号")
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📂 标点符号")
            self.logger.info(f"{'='*60}")
        
        for pinyin, expected in PUNCTUATION_TESTS:
            start = time.perf_counter()
            result = self.engine.process(pinyin)
            elapsed = (time.perf_counter() - start) * 1000
            
            actual = result.candidates[0].text if result.candidates else ""
            passed = actual == expected
            
            case = CaseResult(
                id=f"punct_{pinyin[:10]}",
                pinyin=pinyin,
                expected=expected,
                actual=actual,
                rank=1 if passed else -1,
                passed=passed,
                elapsed_ms=elapsed,
                category="标点符号",
                scenario="标点混合"
            )
            scenario.cases.append(case)
            
            if self.config.verbose:
                status = "✓" if passed else "✗"
                self.logger.info(f'{status} "{pinyin}" -> "{actual}" (期望: "{expected}")')
        
        category.scenarios.append(scenario)
        self.report.categories["标点符号"] = category
        return category
    
    def run_fuzzy_test(self) -> CategoryResult:
        """模糊音测试"""
        category = CategoryResult(name="模糊音纠错")
        scenario = ScenarioResult(name="模糊音", category="模糊音纠错")
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📂 模糊音纠错")
            self.logger.info(f"{'='*60}")
        
        for pinyin, expected, desc in FUZZY_PINYIN:
            case = self._test_pinyin(pinyin, expected)
            case.category = "模糊音纠错"
            case.scenario = desc
            scenario.cases.append(case)
            
            if self.config.verbose:
                status = "✓" if case.passed else "✗"
                self.logger.info(f"{status} [{desc}] '{pinyin}' -> '{case.actual}' (期望: '{expected}')")
        
        category.scenarios.append(scenario)
        self.report.categories["模糊音纠错"] = category
        return category
    
    def run_performance_test(self) -> PerformanceStats:
        """性能压力测试"""
        iterations = self.config.performance_iterations
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"⚡ 性能压力测试 ({iterations} 次)")
            self.logger.info(f"{'='*60}")
        
        latencies = []
        start_total = time.perf_counter()
        
        for i in range(iterations):
            pinyin = random.choice(PERFORMANCE_TEST_INPUTS)
            start = time.perf_counter()
            self.engine.process(pinyin)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        latencies.sort()
        stats = PerformanceStats(
            total_requests=iterations,
            total_time_ms=total_time,
            avg_ms=sum(latencies) / len(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            p50_ms=latencies[len(latencies) // 2],
            p90_ms=latencies[int(len(latencies) * 0.9)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            qps=iterations / (total_time / 1000),
        )
        
        if self.config.verbose:
            self.logger.info(f"📊 性能统计:")
            self.logger.info(f"   请求数: {stats.total_requests}")
            self.logger.info(f"   总耗时: {stats.total_time_ms:.0f}ms")
            self.logger.info(f"   平均延迟: {stats.avg_ms:.2f}ms")
            self.logger.info(f"   P50: {stats.p50_ms:.2f}ms | P90: {stats.p90_ms:.2f}ms | P99: {stats.p99_ms:.2f}ms")
            self.logger.info(f"   QPS: {stats.qps:.1f}")
        
        self.report.performance = stats
        return stats
    
    def run_smoke(self) -> Report:
        """冒烟测试"""
        self._init_engine()
        self.start_time = time.perf_counter()
        self.report.test_level = "smoke"
        self.report.timestamp = datetime.now().isoformat()
        
        self.logger.info("\n🚀 冒烟测试 (Smoke Test)\n")
        
        # 只运行基础测试
        category = CategoryResult(name="冒烟测试")
        scenario = ScenarioResult(name="基础功能", category="冒烟测试")
        
        for pinyin, context, expected, desc in SMOKE_TESTS:
            case = self._test_pinyin(pinyin, expected, context=context)
            case.scenario = desc
            scenario.cases.append(case)
            
            if self.config.verbose:
                status = "✓" if case.passed else "✗"
                self.logger.info(f"{status} [{desc}] '{pinyin}' -> '{case.actual}'")
        
        category.scenarios.append(scenario)
        self.report.categories["冒烟测试"] = category
        
        self._finalize_report()
        return self.report
    
    def run_full(self) -> Report:
        """完整测试"""
        self._init_engine()
        self.start_time = time.perf_counter()
        self.report.test_level = "full"
        self.report.timestamp = datetime.now().isoformat()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🧪 PinHan 全方位仿真测试")
        self.logger.info("="*60)
        
        # 1. 日常对话
        self.run_category("日常对话", DAILY_CONVERSATIONS)
        
        # 2. 工作场景
        self.run_category("工作场景", WORK_SCENARIOS)
        
        # 3. 网络用语
        self.run_category("网络用语", INTERNET_SLANG, use_context=False)
        
        # 4. 专业术语
        self.run_category("专业术语", PROFESSIONAL_TERMS, use_context=False)
        
        # 5. 长句测试
        self.run_category("长句输入", LONG_SENTENCES)
        
        # 6. 单字测试
        self.run_single_char_test()
        
        # 7. 上下文消歧义
        self.run_context_test()
        
        # 8. 标点符号
        self.run_punctuation_test()
        
        # 9. 模糊音
        self.run_fuzzy_test()
        
        # 10. 段落测试
        self.run_paragraph_test()
        
        # 11. 性能测试
        self.run_performance_test()
        
        self._finalize_report()
        return self.report
    
    def run_paragraph_test(self) -> CategoryResult:
        """段落级别连续输入测试"""
        category = CategoryResult(name="段落输入")
        
        if self.config.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📖 段落输入测试")
            self.logger.info(f"{'='*60}")
        
        for para in PARAGRAPH_TESTS:
            scenario = ScenarioResult(name=para["name"], category="段落输入")
            context = ""
            result_text = ""
            
            for pinyin, expected in para["pairs"]:
                ctx = context[-20:] if context else ""
                case = self._test_pinyin(pinyin, expected, context=ctx)
                case.category = "段落输入"
                case.scenario = para["name"]
                scenario.cases.append(case)
                
                # 模拟用户选择：使用期望值更新上下文
                if case.passed:
                    context += expected
                    result_text += expected
                else:
                    context += case.actual
                    result_text += f"[{case.actual}]"
            
            category.scenarios.append(scenario)
            
            if self.config.verbose:
                icon = "✓" if scenario.rate >= 0.9 else ("△" if scenario.rate >= 0.7 else "✗")
                self.logger.info(f"{icon} {para['name']}: {scenario.passed}/{scenario.total} ({scenario.rate*100:.0f}%)")
                self.logger.info(f"   输出: {result_text[:50]}{'...' if len(result_text) > 50 else ''}")
                
                # 显示失败用例
                failed = [c for c in scenario.cases if not c.passed]
                if failed and len(failed) <= 3:
                    for case in failed:
                        self.logger.info(f"   ✗ '{case.pinyin}' -> '{case.actual}' (期望: '{case.expected}')")
        
        self.report.categories["段落输入"] = category
        return category
    
    def _finalize_report(self):
        """完成报告"""
        self.report.duration_seconds = time.perf_counter() - self.start_time
        self.report.engine_stats = self.engine.get_stats()
        self.report.calculate_summary()
        
        # 打印报告
        self._print_report()
        
        # 保存报告
        if self.config.save_report:
            filepath = self.report.save()
            self.logger.info(f"\n📁 报告已保存: {filepath}")
    
    def _print_report(self):
        """打印测试报告"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 测试报告")
        self.logger.info("="*60)
        
        # 分类统计
        self.logger.info("\n分类统计:")
        self.logger.info("-" * 50)
        
        category_stats = [
            (name, cat.passed, cat.total, cat.rate * 100)
            for name, cat in self.report.categories.items()
        ]
        category_stats.sort(key=lambda x: x[3], reverse=True)
        
        for name, passed, total, rate in category_stats:
            icon = "✓" if rate >= 80 else ("△" if rate >= 60 else "✗")
            bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
            self.logger.info(f"{icon} {name:12} {bar} {passed:3}/{total:3} ({rate:5.1f}%)")
        
        self.logger.info("-" * 50)
        self.logger.info(f"📈 总计: {self.report.total_passed}/{self.report.total_cases} "
                        f"({self.report.overall_rate*100:.1f}%)")
        
        # 性能评级
        if self.report.performance:
            self.logger.info("\n性能评级:")
            self.logger.info("-" * 50)
            avg_latency = self.report.performance.avg_ms
            if avg_latency < 15:
                perf_grade = "A (优秀)"
            elif avg_latency < 30:
                perf_grade = "B (良好)"
            elif avg_latency < 50:
                perf_grade = "C (一般)"
            else:
                perf_grade = "D (需优化)"
            
            self.logger.info(f"平均延迟: {avg_latency:.1f}ms -> {perf_grade}")
            self.logger.info(f"QPS: {self.report.performance.qps:.1f} 请求/秒")
        
        # 引擎统计
        self.logger.info("\n引擎统计:")
        self.logger.info("-" * 50)
        stats = self.report.engine_stats
        self.logger.info(f"总请求: {stats.get('total_requests', 0)}")
        self.logger.info(f"缓存命中率: {stats.get('cache_hit_rate', 0)*100:.1f}%")
        
        # 待改进项
        failed_cases = self.report._get_failed_cases()
        if failed_cases:
            self.logger.info("\n⚠️ 待改进项:")
            self.logger.info("-" * 50)
            
            # 按分类分组
            by_category = {}
            for case in failed_cases:
                cat = case['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(case)
            
            for cat_name, cases in by_category.items():
                cat_rate = self.report.categories[cat_name].rate * 100
                self.logger.info(f"\n{cat_name} ({cat_rate:.0f}%):")
                for case in cases[:3]:
                    ctx = f" (ctx='{case['context']}')" if case['context'] else ""
                    self.logger.info(f"  - '{case['pinyin']}'{ctx} -> '{case['actual']}' (期望: '{case['expected']}')")
                if len(cases) > 3:
                    self.logger.info(f"  ... 还有 {len(cases) - 3} 个")
        
        # 最终评定
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 最终评定")
        self.logger.info("="*60)
        
        rate = self.report.overall_rate * 100
        grade_icons = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "❌"}
        grade_names = {"A": "优秀", "B": "良好", "C": "及格", "D": "需改进"}
        
        self.logger.info(f"准确率: {rate:.1f}% -> {grade_icons[self.report.grade]} {grade_names[self.report.grade]} ({self.report.grade})")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="PinHan 全方位测试")
    parser.add_argument("--level", choices=["smoke", "full"], default="full",
                       help="测试级别: smoke(冒烟) / full(完整)")
    parser.add_argument("--save", action="store_true", help="保存测试报告")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--iterations", type=int, default=100, help="性能测试迭代次数")
    
    args = parser.parse_args()
    
    config = RunConfig(
        level=RunLevel.SMOKE if args.level == "smoke" else RunLevel.FULL,
        verbose=not args.quiet,
        save_report=args.save,
        performance_iterations=args.iterations
    )
    
    runner = ComprehensiveTestRunner(config)
    
    if args.level == "smoke":
        report = runner.run_smoke()
    else:
        report = runner.run_full()
    
    # 返回退出码
    sys.exit(0 if report.overall_rate >= 0.7 else 1)


if __name__ == "__main__":
    main()
