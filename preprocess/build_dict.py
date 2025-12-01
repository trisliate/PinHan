"""
字典构建脚本

从 pypinyin 构建 拼音 → 汉字 映射字典
"""

import orjson
import os
from collections import defaultdict
from pypinyin import pinyin, Style, lazy_pinyin


# 常用汉字范围
# CJK 基本区: 0x4E00 - 0x9FFF (约 20,000 字)
CJK_START = 0x4E00
CJK_END = 0x9FFF

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dicts')


def get_all_cjk_chars():
    """获取所有 CJK 基本区汉字"""
    chars = []
    for code in range(CJK_START, CJK_END + 1):
        chars.append(chr(code))
    return chars


def build_char_dict():
    """
    构建 拼音 → 汉字 字典
    使用 pypinyin 获取每个汉字的所有读音
    """
    print("开始构建单字字典...")
    
    char_dict = defaultdict(list)
    chars = get_all_cjk_chars()
    
    for i, char in enumerate(chars):
        if (i + 1) % 5000 == 0:
            print(f"  处理进度: {i + 1}/{len(chars)}")
        
        try:
            # 获取所有读音（无声调）
            readings = lazy_pinyin(char, style=Style.NORMAL)
            if readings and readings[0]:
                py = readings[0].lower()
                # 处理 ü → v
                py = py.replace('ü', 'v')
                if char not in char_dict[py]:
                    char_dict[py].append(char)
        except Exception:
            continue
    
    # 转为普通 dict
    char_dict = dict(char_dict)
    
    print(f"  完成: {len(char_dict)} 个拼音, 共 {sum(len(v) for v in char_dict.values())} 个字")
    return char_dict


def build_pinyin_table(char_dict):
    """从 char_dict 提取拼音表"""
    pinyin_list = sorted(char_dict.keys())
    print(f"拼音表: {len(pinyin_list)} 个拼音")
    return pinyin_list


def build_char_freq():
    """
    构建字频表
    使用扩展常用字表，后续可从语料统计优化
    """
    print("构建字频表...")
    
    # 按使用频率排序的常用汉字（约 3000 字）
    common_chars = (
        # 超高频 (前 100)
        "的一是不了在人有我他这个们中来上大为和国地到以说时"
        "要就出会可也你对生能而子那得于着下自之年过发后作里用"
        "道行所然家种事成方多经么去法学如都同现当没动面起看定"
        "天分还进好小部其些主样理心她本前开但因只从想实日军者"
        # 高频 (100-300)
        "意无力它与长把机十民第公此已工使情明性知全三又关点正"
        "业外将两高间由问很最重并物手应战向头文体政美相见被利"
        "什二等产或新己制身果加西斯月话合回特代内信表化老给世"
        "位次度门任常先海通教儿原东声提立及比员解水名真论处走"
        "义各入几口认条平系气题活尔更别打女变四神总何电数安少"
        # 中高频 (300-600)
        "报才结反受目太量再感建务做接必场件计管期市直德资命山"
        "金指克许统区保至队形社便空决治展马科司五基眼书非则听"
        "白却界达光放强即像难且权思王象完设式色路记南品住告类"
        "求据程北每花传元世百济华边线觉院改争领风共令群究各青"
        "笑容运深收早史众研尽今官单良切病远清兴病越清片州破持"
        "该照价值规备周宗节欢流落识死农古钱故标算需息转示河脸"
        "联哪角席呢势参根布案史火准叫段视离况找跟岁球似约怎血"
        "亲纪交谈责养精况校言推九读座随首断朝费影丽米注冷依越"
        "父章态环愿啊速精吃念久讲步跑足绝团飞客谁答希船究集片"
        # 中频 (600-1000)
        "极忙若批假修整游居另止努负责村枪局红验差错试销称静香"
        "忽装检块剧武景七伤底亚帮短馆引草既木画调陈纸志晚项黄"
        "句素积乐仅胜细助买春若艺拉阳技介买换顾毛苦房概供确穿"
        "鱼温阶星孩紧质假拿课待烟初威显职房职压楼户宝护测密烈"
        "充守套恐洋沉冲射喜附额帝突某围巨双夜忘福朋推虑宣降归"
        "止怀招乡降热演犯骨余松触彩征爸掉杀警诗置陆缺针震阵批"
        "稳脚健请激兰秀顶楚旅财牌弹银述限官财馆刚园杨纳误醒亿"
        "圆陆县坐油普闻审亡刻忆京含杯招劳环套乱犹杂托衣药载鲜"
        # 扩展常用字
        "甚吸梦乎池痛零坚苏伯悲洲冰玩牛敢刘障督训哭吓敌娘旁避"
        "脑纯露杰误堂释废迷暗详凡莫访疑卷盘紧刺殊释翻雪杨怒姓"
        "盛床奇亦雨脱宫庭圣桌逐暴承镜避渐雄冒嘴劲掌凉驶弟寻忍"
        "伟骗兄糊哥插绍吹刀闪拍灵迫拔浪勇藏窗辛骂符输驻寒湖竟"
        "扬毫跳逃辈伙妹卫播糖隐础宁乘促铁奶夺扫盖赞勒累碎竞伸"
        "折幸您盟遍恩珠扎键混墙链镇沿撤贴抓幕虚雷宽瞧辉灯勤航"
        "尾魔洗填诺毁丝践智咬暂齐颜恶唯偏催鬼割惊撞跃楼丁恨唐"
        "拥抱淡震泪寄仍隔喝妻筑滚添陪喊暖歌幅舞屋尘遗纷亮贵鼻"
        "骑聚肩胸阻欲肉旗涉驾颗享舍闭袋猛懂冬戒彻婚圈扰寂帐繁"
        "吗睛偷洞忧讨荣汽喷姑湾仔咱聊漂扣抛傲雅宿仁挑疗浮狂佳"
        "惜粮锋芳谢秋渡签肌坛蒙揭瓶鬼趣趟赔嫁箱偶冻喂灭戏沙雀"
        "缓夹摆揭鲁帽惠阅尝劲辩抽薄蔬喷胶帅滑匹摄丧幼描匆辅梅"
        "惯谈卧膀纲茶晨隆伍辞纵抬暑撒涂丑扫椅凝谷秘闷陵宅碗逼"
        "憾筋仪浓搞乏巷弯拾毅姿携绵躲腰泛滩搬旬牵炮蛋欣虎妙饭"
        "届媒萨悄姻魅尺矛驱冠剑珍贺慕乾搭瞬裁铜仿慎薪嗯咳偿耀"
        "吐柳韩愧丢贼欠踪绪晃坡仓促揉柔浙耐挤裂窄瓜扁毯倾丛叶"
    )
    
    char_freq = {}
    
    # 按位置赋予频率
    total = len(common_chars)
    for i, char in enumerate(common_chars):
        # 使用指数衰减，前面的字频率更高
        freq = 0.1 * (0.998 ** i)
        char_freq[char] = max(freq, 0.00001)
    
    print(f"  常用字频率: {len(char_freq)} 个")
    return char_freq


def save_json(data, filename):
    """保存 JSON 文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"已保存: {path}")


def save_txt(lines, filename):
    """保存文本文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"已保存: {path}")


def main():
    print("=" * 50)
    print("字典构建开始")
    print("=" * 50)
    
    # 1. 构建单字字典
    char_dict = build_char_dict()
    save_json(char_dict, 'char_dict.json')
    
    # 2. 提取拼音表
    pinyin_table = build_pinyin_table(char_dict)
    save_txt(pinyin_table, 'pinyin_table.txt')
    
    # 3. 构建字频表
    char_freq = build_char_freq()
    save_json(char_freq, 'char_freq.json')
    
    # 4. 词典先创建空文件（后续模块填充）
    save_json({}, 'word_dict.json')
    save_json({}, 'word_freq.json')
    
    print("=" * 50)
    print("字典构建完成")
    print("=" * 50)


if __name__ == '__main__':
    main()
