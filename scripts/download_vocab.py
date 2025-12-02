"""
下载并整合第三方词库

数据源:
1. mozillazg/phrase-pinyin-data - 短语拼音词库 (~9MB)
2. wainshine/Chinese-Names-Corpus - 中文人名语料库 (~12MB)

用法:
    python scripts/download_vocab.py           # 下载 + 整合
    python scripts/download_vocab.py --merge   # 仅整合(已下载)
    python scripts/download_vocab.py --download # 仅下载
"""
import argparse
import urllib.request
import orjson
from pathlib import Path
from pypinyin import lazy_pinyin

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCES_DIR = PROJECT_ROOT / 'data' / 'sources'
DICTS_DIR = PROJECT_ROOT / 'data' / 'dicts'

# ============ 下载源 ============
VOCAB_SOURCES = {
    'phrase_pinyin.txt': {
        'url': 'https://raw.githubusercontent.com/mozillazg/phrase-pinyin-data/master/phrase_pinyin.txt',
        'desc': '短语拼音词库 (mozillazg/phrase-pinyin-data)',
    },
    'chinese_names.txt': {
        'url': 'https://raw.githubusercontent.com/wainshine/Chinese-Names-Corpus/master/Chinese_Names_Corpus/Chinese_Names_Corpus%EF%BC%88120W%EF%BC%89.txt',
        'desc': '中文人名语料库 (wainshine/Chinese-Names-Corpus)',
    },
}

# 拼音声调转换表
TONE_MAP = {
    'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
    'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
    'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
    'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
    'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
    'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v',
    'ń': 'n', 'ň': 'n', 'ǹ': 'n',
}


def remove_tone(pinyin: str) -> str:
    """移除拼音声调"""
    return ''.join(TONE_MAP.get(c, c) for c in pinyin)


def download_file(url: str, dest: Path, desc: str) -> bool:
    """下载文件"""
    print(f"📥 下载 {desc}...")
    print(f"   URL: {url}")
    
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
            dest.write_bytes(data)
            size_mb = len(data) / 1024 / 1024
            print(f"   ✓ 成功: {dest.name} ({size_mb:.1f} MB)")
            return True
            
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        return False


def download_all() -> bool:
    """下载所有词库"""
    print("\n" + "=" * 50)
    print("📦 下载第三方词库")
    print("=" * 50)
    
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for filename, info in VOCAB_SOURCES.items():
        dest = SOURCES_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"⏭️  跳过 {filename} (已存在, {size_mb:.1f} MB)")
            success_count += 1
        else:
            if download_file(info['url'], dest, info['desc']):
                success_count += 1
    
    print(f"\n下载完成: {success_count}/{len(VOCAB_SOURCES)}")
    return success_count == len(VOCAB_SOURCES)


def parse_phrase_pinyin(filepath: Path) -> dict[str, list[str]]:
    """解析 phrase_pinyin.txt"""
    pinyin_to_words = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            word = parts[0].strip()
            pinyin_with_tone = parts[1].strip()
            
            if len(word) < 2:
                continue
            
            syllables = pinyin_with_tone.split()
            pinyin = ''.join(remove_tone(s) for s in syllables)
            
            if pinyin not in pinyin_to_words:
                pinyin_to_words[pinyin] = []
            if word not in pinyin_to_words[pinyin]:
                pinyin_to_words[pinyin].append(word)
    
    return pinyin_to_words


def parse_chinese_names(filepath: Path) -> dict[str, list[str]]:
    """解析中文人名语料库"""
    pinyin_to_names = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith(('By@', '#', '//', '2025')):
                continue
            
            if not (2 <= len(name) <= 4):
                continue
            if not all('\u4e00' <= c <= '\u9fff' for c in name):
                continue
            
            try:
                py_list = lazy_pinyin(name)
                pinyin = ''.join(py_list)
            except Exception:
                continue
            
            if pinyin not in pinyin_to_names:
                pinyin_to_names[pinyin] = []
            if name not in pinyin_to_names[pinyin]:
                pinyin_to_names[pinyin].append(name)
    
    return pinyin_to_names


def merge_vocab():
    """整合词库到字典"""
    print("\n" + "=" * 50)
    print("🔗 整合词库到字典")
    print("=" * 50)
    
    word_dict_path = DICTS_DIR / 'word_dict.json'
    if word_dict_path.exists():
        word_dict = orjson.loads(word_dict_path.read_bytes())
        print(f"加载现有词典: {len(word_dict):,} 条")
    else:
        word_dict = {}
        print("⚠️  词典不存在，请先运行 python scripts/build_dict.py")
        return False
    
    original_count = len(word_dict)
    new_pinyin_count = 0
    new_word_count = 0
    
    phrase_path = SOURCES_DIR / 'phrase_pinyin.txt'
    if phrase_path.exists():
        print(f"\n解析短语词库...")
        phrase_vocab = parse_phrase_pinyin(phrase_path)
        print(f"  获取 {len(phrase_vocab):,} 个拼音条目")
        
        for pinyin, words in phrase_vocab.items():
            if pinyin not in word_dict:
                word_dict[pinyin] = []
                new_pinyin_count += 1
            for word in words:
                if word not in word_dict[pinyin]:
                    word_dict[pinyin].append(word)
                    new_word_count += 1
    
    names_path = SOURCES_DIR / 'chinese_names.txt'
    if names_path.exists():
        print(f"\n解析人名语料库...")
        names_vocab = parse_chinese_names(names_path)
        print(f"  获取 {len(names_vocab):,} 个拼音条目")
        
        for pinyin, names in names_vocab.items():
            if pinyin not in word_dict:
                word_dict[pinyin] = []
                new_pinyin_count += 1
            for name in names:
                if name not in word_dict[pinyin]:
                    word_dict[pinyin].append(name)
                    new_word_count += 1
    
    word_dict_path.write_bytes(
        orjson.dumps(word_dict, option=orjson.OPT_INDENT_2)
    )
    
    print(f"\n" + "=" * 50)
    print(f"✅ 词典更新完成")
    print(f"   原拼音条目: {original_count:,}")
    print(f"   新增拼音: {new_pinyin_count:,}")
    print(f"   新增词条: {new_word_count:,}")
    print(f"   当前总计: {len(word_dict):,} 拼音条目")
    
    print(f"\n验证常用人名:")
    test_cases = ['xiaoming', 'xiaohua', 'zhangsan', 'lisi', 'wangwu', 'xiaohong']
    for py in test_cases:
        words = word_dict.get(py, [])
        if words:
            print(f"  ✓ {py}: {words[:5]}{'...' if len(words) > 5 else ''}")
        else:
            print(f"  ✗ {py}: 未找到")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="下载并整合第三方词库")
    parser.add_argument('--download', action='store_true', help='仅下载')
    parser.add_argument('--merge', action='store_true', help='仅整合')
    args = parser.parse_args()
    
    if args.download:
        download_all()
    elif args.merge:
        merge_vocab()
    else:
        if download_all():
            merge_vocab()


if __name__ == '__main__':
    main()
