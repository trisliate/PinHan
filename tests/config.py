"""
测试配置和日志设置
"""
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
REPORT_DIR = PROJECT_ROOT / 'logs' / 'test_reports'

# 确保目录存在
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


class TestLevel(Enum):
    """测试级别"""
    SMOKE = "smoke"        # 冒烟测试 - 快速验证
    BASIC = "basic"        # 基础测试 - 核心功能
    FULL = "full"          # 完整测试 - 全部用例
    STRESS = "stress"      # 压力测试 - 性能


@dataclass
class TestConfig:
    """测试配置"""
    level: TestLevel = TestLevel.FULL
    verbose: bool = True
    save_report: bool = True
    log_to_file: bool = True
    top_n: int = 3  # 前N名算通过
    performance_iterations: int = 100


# ============================================
# 测试日志器
# ============================================

class TestFormatter(logging.Formatter):
    """测试专用格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'PASS': '\033[32m',
        'FAIL': '\033[31m',
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # 自定义测试结果级别
        if hasattr(record, 'test_status'):
            status = record.test_status
            color = self.COLORS.get(status, '')
            record.msg = f"{color}[{status}]{self.RESET} {record.msg}"
        return super().format(record)


_test_logger = None

def get_test_logger() -> logging.Logger:
    """获取测试日志器"""
    global _test_logger
    if _test_logger is not None:
        return _test_logger
    
    logger = logging.getLogger('pinhan.test')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    if sys.stdout.isatty():
        console.setFormatter(TestFormatter('%(message)s'))
    else:
        console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    
    # 文件
    log_file = LOG_DIR / 'pinhan.test.log'
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    logger.addHandler(file_handler)
    
    _test_logger = logger
    return logger


# ============================================
# 测试结果数据结构
# ============================================

@dataclass
class TestCase:
    """单个测试用例结果"""
    id: str
    pinyin: str
    expected: str
    actual: str = ""
    rank: int = -1
    passed: bool = False
    elapsed_ms: float = 0.0
    context: str = ""
    category: str = ""
    scenario: str = ""


@dataclass
class ScenarioResult:
    """场景测试结果"""
    name: str
    category: str = ""
    cases: List[TestCase] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)
    
    @property
    def total(self) -> int:
        return len(self.cases)
    
    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        if not self.cases:
            return 0
        return sum(c.elapsed_ms for c in self.cases) / len(self.cases)


@dataclass
class CategoryResult:
    """分类测试结果"""
    name: str
    scenarios: List[ScenarioResult] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.scenarios)
    
    @property
    def total(self) -> int:
        return sum(s.total for s in self.scenarios)
    
    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0


@dataclass
class PerformanceStats:
    """性能统计"""
    total_requests: int = 0
    total_time_ms: float = 0
    avg_ms: float = 0
    min_ms: float = 0
    max_ms: float = 0
    p50_ms: float = 0
    p90_ms: float = 0
    p99_ms: float = 0
    qps: float = 0


@dataclass
class TestReport:
    """完整测试报告"""
    timestamp: str = ""
    test_level: str = ""
    duration_seconds: float = 0
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    performance: Optional[PerformanceStats] = None
    engine_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 汇总
    total_passed: int = 0
    total_cases: int = 0
    overall_rate: float = 0
    grade: str = ""
    
    def calculate_summary(self):
        """计算汇总数据"""
        self.total_passed = sum(c.passed for c in self.categories.values())
        self.total_cases = sum(c.total for c in self.categories.values())
        self.overall_rate = self.total_passed / self.total_cases if self.total_cases > 0 else 0
        
        rate = self.overall_rate * 100
        if rate >= 90:
            self.grade = "A"
        elif rate >= 80:
            self.grade = "B"
        elif rate >= 70:
            self.grade = "C"
        else:
            self.grade = "D"
    
    def to_dict(self) -> Dict:
        """转换为可序列化字典"""
        return {
            "timestamp": self.timestamp,
            "test_level": self.test_level,
            "duration_seconds": self.duration_seconds,
            "summary": {
                "total_passed": self.total_passed,
                "total_cases": self.total_cases,
                "overall_rate": self.overall_rate,
                "grade": self.grade,
            },
            "categories": {
                name: {
                    "passed": cat.passed,
                    "total": cat.total,
                    "rate": cat.rate,
                    "scenarios": [
                        {
                            "name": s.name,
                            "passed": s.passed,
                            "total": s.total,
                            "rate": s.rate,
                            "avg_latency_ms": s.avg_latency,
                        }
                        for s in cat.scenarios
                    ]
                }
                for name, cat in self.categories.items()
            },
            "performance": asdict(self.performance) if self.performance else None,
            "engine_stats": self.engine_stats,
            "failed_cases": self._get_failed_cases(),
        }
    
    def _get_failed_cases(self) -> List[Dict]:
        """获取失败用例列表"""
        failed = []
        for cat_name, cat in self.categories.items():
            for scenario in cat.scenarios:
                for case in scenario.cases:
                    if not case.passed:
                        failed.append({
                            "category": cat_name,
                            "scenario": scenario.name,
                            "pinyin": case.pinyin,
                            "context": case.context,
                            "expected": case.expected,
                            "actual": case.actual,
                            "rank": case.rank,
                        })
        return failed
    
    def save(self, filename: str = None):
        """保存报告到文件"""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{ts}.json"
        
        filepath = REPORT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        return filepath
