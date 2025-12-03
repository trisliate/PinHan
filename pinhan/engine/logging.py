"""
统一日志配置模块

提供结构化日志、文件轮转、性能监控等功能
"""

import os
import sys
import logging
import orjson
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
from pathlib import Path
from functools import wraps
import time


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'


class JsonFormatter(logging.Formatter):
    """JSON 格式日志（便于日志分析工具解析）"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加额外字段
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
            
        # 异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return orjson.dumps(log_data).decode('utf-8')


class ColorFormatter(logging.Formatter):
    """彩色控制台输出"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = 'pinhan',
    level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        json_format: 是否使用 JSON 格式（适合生产环境）
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
    
    Returns:
        配置好的 Logger 实例
    """
    # 创建日志目录
    LOG_DIR.mkdir(exist_ok=True)
    
    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有 handlers（避免重复添加）
    logger.handlers.clear()
    
    # 日志格式
    detailed_format = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
    simple_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    # 控制台输出
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if json_format:
            console_handler.setFormatter(JsonFormatter())
        else:
            # 检测是否支持颜色
            if sys.stdout.isatty():
                console_handler.setFormatter(ColorFormatter(simple_format))
            else:
                console_handler.setFormatter(logging.Formatter(simple_format))
        
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        # 主日志文件（轮转）
        main_log_path = LOG_DIR / f'{name}.log'
        file_handler = RotatingFileHandler(
            main_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(detailed_format))
        
        logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_log_path = LOG_DIR / f'{name}_error.log'
        error_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(detailed_format))
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = 'pinhan') -> logging.Logger:
    """获取已配置的 logger（如果未配置则自动配置）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


class LogContext:
    """日志上下文管理器，用于添加额外信息"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extra = kwargs
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def info(self, msg, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def debug(self, msg, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def warning(self, msg, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def _log(self, level, msg, **kwargs):
        extra = {**self.extra, **kwargs}
        record = self.logger.makeRecord(
            self.logger.name, level, '', 0, msg, (), None
        )
        for k, v in extra.items():
            setattr(record, k, v)
        self.logger.handle(record)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """装饰器：记录函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()
            
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"{func.__name__} 执行完成, 耗时: {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(f"{func.__name__} 执行失败, 耗时: {elapsed:.2f}ms, 错误: {e}")
                raise
        return wrapper
    return decorator


# 预配置的日志器
api_logger = None
engine_logger = None
train_logger = None


def get_api_logger() -> logging.Logger:
    """获取 API 日志器"""
    global api_logger
    if api_logger is None:
        api_logger = setup_logging('pinhan.api', level='INFO')
    return api_logger


def get_engine_logger() -> logging.Logger:
    """获取引擎日志器"""
    global engine_logger
    if engine_logger is None:
        engine_logger = setup_logging('pinhan.engine', level='INFO')
    return engine_logger


def get_train_logger() -> logging.Logger:
    """获取训练日志器"""
    global train_logger
    if train_logger is None:
        train_logger = setup_logging('pinhan.train', level='DEBUG')
    return train_logger
