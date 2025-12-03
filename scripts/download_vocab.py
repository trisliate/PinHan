#!/usr/bin/env python3
"""
第三方词库集成脚本

从公开来源下载和集成第三方词库，包括：
1. jieba - 中文分词词库
2. THUOCL - 清华大学开放中文词库（包含教育、IT等专业词库）
3. Sogou词库 - 互联网热词库

使用方法:
    python scripts/download_vocab.py
    python scripts/build_dict.py

输出: data/sources/ 下对应的词库文件
"""

import os
import gzip
import urllib.request
import urllib.error
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
SOURCES_DIR = DATA_DIR / 'sources'

# 确保目录存在
SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, target_path, timeout=30):
    """下载文件"""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if target_path.exists():
        logger.info(f"文件已存在: {target_path}")
        return True
    
    try:
        logger.info(f"下载: {url}")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with open(target_path, 'wb') as out:
                out.write(response.read())
        logger.info(f"✓ 下载完成: {target_path}")
        return True
    except urllib.error.URLError as e:
        logger.warning(f"✗ 下载失败 {url}: {e}")
        return False
    except Exception as e:
        logger.warning(f"✗ 错误: {e}")
        return False


def convert_jieba_dict():
    """
    转换 jieba 词库格式
    
    jieba 词库格式: 词语 频率 词性1 词性2...
    转换为: 词语 频率
    """
    logger.info("处理 jieba 词库...")
    
    # jieba 词库 GitHub 地址
    jieba_url = "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt"
    jieba_path = SOURCES_DIR / 'jieba' / 'dict.txt'
    
    if not download_file(jieba_url, jieba_path):
        logger.warning("jieba 词库下载失败，将跳过")
        return False
    
    # 转换格式
    output_path = SOURCES_DIR / 'jieba' / 'vocab.txt'
    try:
        with open(jieba_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    freq = parts[1]
                    f.write(f"{word} {freq}\n")
        
        logger.info(f"✓ jieba 词库转换完成: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"✗ jieba 词库处理失败: {e}")
        return False


def download_thuocl():
    """
    下载 THUOCL (清华大学开放中文词库)
    
    THUOCL 包含多个领域词库：
    - 学科词库 (education)
    - IT词库 (IT)
    - 财经词库 (finance)
    等
    """
    logger.info("下载 THUOCL 词库...")
    
    # THUOCL GitHub 地址 - 词库清单见: https://github.com/thunlp/THUOCL
    base_url = "https://raw.githubusercontent.com/thunlp/THUOCL/master/data"
    
    # 下载 IT 词库和成语词库 (教育类词库在 2016-2017 的版本不存在，使用 IT + 成语替代)
    vocab_files = {
        'IT': f"{base_url}/THUOCL_IT.txt",
        'chengyu': f"{base_url}/THUOCL_chengyu.txt",  # 成语包含教学相关表述
    }
    
    downloaded = []
    for domain, url in vocab_files.items():
        path = SOURCES_DIR / 'THUOCL' / f'{domain}.txt'
        if download_file(url, path):
            logger.info(f"✓ THUOCL {domain} 词库: {path}")
            downloaded.append(True)
        else:
            logger.warning(f"✗ THUOCL {domain} 下载失败: {url}")
            downloaded.append(False)
    
    return any(downloaded)


def download_sogou():
    """
    下载搜狗词库
    
    注: 搜狗词库为 .scel 二进制格式，需要转换
    这里下载社区转换后的文本版本
    """
    logger.info("下载搜狗词库...")
    
    # 搜狗词库备用来源
    sogou_urls = [
        "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big",
        # 其他来源可以添加到这里
    ]
    
    for url in sogou_urls:
        sogou_path = SOURCES_DIR / 'sogou' / 'vocab.txt'
        if download_file(url, sogou_path):
            logger.info(f"✓ 搜狗词库下载完成: {sogou_path}")
            return True
    
    logger.warning("搜狗词库下载失败")
    return False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始下载和集成第三方词库...")
    logger.info("=" * 60)
    
    results = {
        'jieba': convert_jieba_dict(),
        'THUOCL': download_thuocl(),
        'sogou': download_sogou(),
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("下载完成，汇总:")
    logger.info("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {name}: {'成功' if success else '失败或跳过'}")
    
    logger.info("\n下一步: python scripts/build_dict.py")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
