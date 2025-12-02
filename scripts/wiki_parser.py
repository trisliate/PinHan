"""
维基百科 XML 解析器

从 zhwiki-latest-pages-articles.xml 提取纯文本
使用流式解析，边读边处理，支持大文件
"""

import re
from typing import Iterator, Optional


class WikiTextCleaner:
    """维基文本清洗器，移除 MediaWiki 标记"""
    
    # 需要移除的模式
    PATTERNS = [
        # 移除 {{ }} 模板（包括嵌套）
        (r'\{\{[^{}]*\}\}', ''),
        # 移除 [[ ]] 内部链接，保留显示文本
        (r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1'),
        # 移除 [ ] 外部链接
        (r'\[https?://[^\]]+\]', ''),
        # 移除 HTML 标签
        (r'<[^>]+>', ''),
        # 移除 ''' 和 '' 格式标记
        (r"'{2,}", ''),
        # 移除 == 标题标记
        (r'={2,}[^=]+={2,}', ''),
        # 移除 * # : ; 列表标记
        (r'^[\*#:;]+\s*', ''),
        # 移除 | 表格内容
        (r'\|[^\n]*', ''),
        # 移除 {| |} 表格标记
        (r'\{\|[^}]*\|\}', ''),
        # 移除多余空白
        (r'\s+', ' '),
    ]
    
    # 编译正则
    COMPILED = [(re.compile(p, re.MULTILINE), r) for p, r in PATTERNS]
    
    @classmethod
    def clean(cls, text: str) -> str:
        """清洗维基文本"""
        # 多次处理嵌套模板
        for _ in range(5):
            old_len = len(text)
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
            if len(text) == old_len:
                break
        
        # 应用其他清洗规则
        for pattern, replacement in cls.COMPILED[1:]:  # 跳过第一个模板规则
            text = pattern.sub(replacement, text)
        
        return text.strip()


def parse_wiki_xml(xml_path: str, max_articles: Optional[int] = None) -> Iterator[tuple]:
    """
    流式解析维基 XML 文件（边读边返回，不等待全部解析完）
    
    Args:
        xml_path: XML 文件路径
        max_articles: 最大文章数（None 表示全部）
    
    Yields:
        (title, cleaned_text) 元组
    """
    import xml.etree.ElementTree as ET
    
    count = 0
    
    # 使用 iterparse 流式解析
    context = ET.iterparse(xml_path, events=('end',))
    
    for event, elem in context:
        # 只处理 page 元素
        if elem.tag.endswith('page'):
            # 查找命名空间
            ns_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}ns')
            if ns_elem is None:
                ns_elem = elem.find('.//ns')
            
            ns = ns_elem.text if ns_elem is not None else None
            
            # 只处理主命名空间 (ns=0)
            if ns == '0':
                # 获取标题
                title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}title')
                if title_elem is None:
                    title_elem = elem.find('.//title')
                title = title_elem.text if title_elem is not None else ''
                
                # 获取文本
                text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}text')
                if text_elem is None:
                    text_elem = elem.find('.//text')
                text = text_elem.text if text_elem is not None else ''
                
                if text and len(text) > 50:
                    # 清洗文本
                    cleaned = WikiTextCleaner.clean(text)
                    
                    if len(cleaned) > 50:
                        count += 1
                        yield (title, cleaned)
                        
                        if max_articles and count >= max_articles:
                            # 清理并退出
                            elem.clear()
                            return
            
            # 释放内存
            elem.clear()


def extract_sentences(text: str) -> Iterator[str]:
    """
    从文本中提取句子
    
    按标点符号分割，过滤非中文内容
    """
    # 按句号、问号、感叹号分割
    sentences = re.split(r'[。！？；\n]', text)
    
    for sent in sentences:
        sent = sent.strip()
        # 过滤条件
        if len(sent) < 4:  # 太短
            continue
        if len(sent) > 100:  # 太长
            continue
        # 检查中文字符占比
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sent))
        if chinese_chars / max(len(sent), 1) < 0.7:  # 中文占比低于 70%
            continue
        # 移除剩余非中文字符（保留中文和基本标点）
        sent = re.sub(r'[^\u4e00-\u9fff，、：""''（）]', '', sent)
        if len(sent) >= 4:
            yield sent


if __name__ == '__main__':
    # 测试解析
    import os
    
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'zhwiki-latest-pages-articles.xml'
    )
    
    print("测试解析前 10 篇文章...")
    for i, (title, text) in enumerate(parse_wiki_xml(xml_path, max_articles=10)):
        print(f"\n=== {title} ===")
        sentences = list(extract_sentences(text))[:3]
        for s in sentences:
            print(f"  {s}")
