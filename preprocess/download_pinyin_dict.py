"""下载并构建标准拼音-汉字字典（独立于语料数据）。

数据源：https://github.com/mozillazg/pinyin-data
这是一个完整的 Unicode 汉字拼音映射，覆盖所有常用汉字。

用法:
    python preprocess/download_pinyin_dict.py --out-dir dicts

输出:
  - dicts/pinyin_to_hanzi.json   (pinyin -> [hanzi...])  拼音到汉字映射
  - dicts/hanzi_to_pinyin.json   (hanzi -> [pinyin...])  汉字到拼音映射（多音字）
"""
import argparse
import re
import urllib.request
from pathlib import Path
from collections import defaultdict
import orjson

# 标准拼音数据源
PINYIN_DATA_URL = "https://raw.githubusercontent.com/mozillazg/pinyin-data/master/pinyin.txt"

# 声调标记到数字的映射
TONE_MARK_TO_NUM = {
    'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'),
    'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'),
    'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'),
    'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'),
    'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'),
    'ǖ': ('v', '1'), 'ǘ': ('v', '2'), 'ǚ': ('v', '3'), 'ǜ': ('v', '4'),
    'ń': ('n', '2'), 'ň': ('n', '3'), 'ǹ': ('n', '4'),
    'ḿ': ('m', '2'),
}


def tone_mark_to_number(pinyin_with_mark: str) -> str:
    """将带声调标记的拼音转换为数字音调形式。
    
    例：'mā' -> 'ma1', 'hǎo' -> 'hao3'
    """
    result = []
    tone = ''
    for char in pinyin_with_mark.lower():
        if char in TONE_MARK_TO_NUM:
            letter, tone_num = TONE_MARK_TO_NUM[char]
            result.append(letter)
            tone = tone_num
        else:
            result.append(char)
    
    # 如果没有声调标记，默认为轻声(0)或无声调
    if not tone:
        # 检查是否已经有数字声调
        if result and result[-1].isdigit():
            return ''.join(result)
        return ''.join(result)  # 不加声调
    
    return ''.join(result) + tone


def download_pinyin_data(url: str = PINYIN_DATA_URL) -> str:
    """下载拼音数据文件。"""
    print(f"正在下载拼音数据: {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')
        print(f"下载成功，共 {len(content)} 字节")
        return content
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def parse_pinyin_data(content: str) -> tuple:
    """解析拼音数据文件。
    
    格式：U+XXXX: pinyin1,pinyin2  # 汉字
    
    Returns:
        (hanzi_to_pinyin, pinyin_to_hanzi)
    """
    hanzi_to_pinyin = {}
    pinyin_to_hanzi = defaultdict(set)
    
    # 正则匹配: U+XXXX: pīnyīn,pīnyīn  # 汉字
    pattern = re.compile(r'^U\+([0-9A-F]+):\s*([^#]+?)(?:\s*#\s*(.*))?$')
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        match = pattern.match(line)
        if not match:
            continue
        
        code_point = match.group(1)
        pinyins_str = match.group(2).strip()
        hanzi = match.group(3).strip() if match.group(3) else None
        
        # 如果没有直接给出汉字，从 code point 转换
        if not hanzi:
            try:
                hanzi = chr(int(code_point, 16))
            except (ValueError, OverflowError):
                continue
        
        # 跳过非常用字符（控制字符等）
        if len(hanzi) != 1 or not hanzi.strip():
            continue
        
        # 解析拼音（可能有多个，用逗号分隔）
        pinyins = []
        for p in pinyins_str.split(','):
            p = p.strip()
            if p:
                # 转换声调标记为数字
                p_num = tone_mark_to_number(p)
                if p_num:
                    pinyins.append(p_num)
        
        if pinyins:
            hanzi_to_pinyin[hanzi] = pinyins
            for py in pinyins:
                pinyin_to_hanzi[py].add(hanzi)
    
    return hanzi_to_pinyin, pinyin_to_hanzi


def build_dicts(out_dir: Path):
    """构建并保存字典。"""
    # 下载数据
    content = download_pinyin_data()
    
    # 解析数据
    print("正在解析拼音数据...")
    hanzi_to_pinyin, pinyin_to_hanzi = parse_pinyin_data(content)
    
    print(f"解析完成:")
    print(f"  - 汉字数量: {len(hanzi_to_pinyin)}")
    print(f"  - 拼音数量: {len(pinyin_to_hanzi)}")
    
    # 统计多音字
    polyphonic = {h: p for h, p in hanzi_to_pinyin.items() if len(p) > 1}
    print(f"  - 多音字数量: {len(polyphonic)}")
    
    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 拼音 -> 汉字列表
    p2h_json = {p: sorted(list(hs)) for p, hs in pinyin_to_hanzi.items()}
    with open(out_dir / 'pinyin_to_hanzi.json', 'wb') as f:
        f.write(orjson.dumps(p2h_json, option=orjson.OPT_INDENT_2))
    print(f"已保存: {out_dir / 'pinyin_to_hanzi.json'}")
    
    # 2. 汉字 -> 拼音列表（多音字）
    with open(out_dir / 'hanzi_to_pinyin.json', 'wb') as f:
        f.write(orjson.dumps(hanzi_to_pinyin, option=orjson.OPT_INDENT_2))
    print(f"已保存: {out_dir / 'hanzi_to_pinyin.json'}")
    
    # 3. 打印一些示例
    print("\n示例拼音映射:")
    for py in ['ni3', 'hao3', 'shi4', 'zhong1', 'guo2']:
        if py in p2h_json:
            chars = p2h_json[py][:10]  # 只显示前10个
            more = f"... (共{len(p2h_json[py])}个)" if len(p2h_json[py]) > 10 else ""
            print(f"  {py}: {''.join(chars)}{more}")
    
    print("\n示例多音字:")
    for hanzi, pys in list(polyphonic.items())[:10]:
        print(f"  {hanzi}: {', '.join(pys)}")


def main():
    parser = argparse.ArgumentParser(description='下载并构建标准拼音字典')
    parser.add_argument('--out-dir', default='dicts', help='输出目录')
    args = parser.parse_args()
    
    build_dicts(Path(args.out_dir))


if __name__ == '__main__':
    main()
