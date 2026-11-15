"""北京居民生活垃圾追问模块。Python 3.10+，只依赖标准库。

组员模型回答“这是什么东西”，本模块回答“在北京应该投哪个桶”。
Garbage265 的 265 个类别逐一对照了北京的规定（见 ITEMS 表），需要时向用户追问一两个问题。
"""
import argparse
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "2026-11-15.3"
CHECKED = "2026-11-15"

# 依据来源。键名在 ITEMS 和 RULES 里引用。
SOURCES = {
    "S1": ("《北京市生活垃圾管理条例》", "https://www.beijing.gov.cn/zhengce/dfxfg/202009/t20200929_2102701.html"),
    "S1b": ("《北京市生活垃圾管理条例》第三十四条", "https://www.beijing.gov.cn/zhengce/dfxfg/202009/t20200929_2102701.html"),
    "S2": ("《北京市可回收物指导目录（2021年版）》", "https://www.beijing.gov.cn/ywdt/gzdt/202103/t20210303_2297758.html"),
    "S3": ("北京市园林绿化局《北京市公园生活垃圾分类指导意见》", "https://www.beijing.gov.cn/zhengce/zhengcefagui/202008/t20200811_1979366.html"),
    "S4": ("丰台区政府《垃圾分类我们一起来》", "https://www.bjft.gov.cn/ftq/laflftzxd/202004/0c2e5818f8a441ffac1778fe75478056.shtml"),
    "S5": ("新京报：市城管委指导梳理的120种常见垃圾", "https://m.bjnews.com.cn/detail/158815350515844.html"),
    "S6": ("北京政法网：干电池、锂电池等的分类", "https://www.bj148.org/rd/201907/t20190704_1518435.html"),
    "S7": ("转述“北京市垃圾分类宝典”：充电宝属可回收物", "https://www.toutiao.com/zixun/7490068742513248290/"),
}
# 依据等级 → 结果里的 basis
BASIS = {
    "A": "beijing_rule",       # 北京官方文件点名（物品或其所属小类）
    "B": "beijing_rule",       # 市城管委指导的新闻清单点名
    "C": "beijing_inferred",   # 按北京官方类别定义类推
    "D": "dataset_label",      # 未找到北京依据，暂沿用数据集
    "X": "beijing_conflict",   # 北京来源之间有分歧，按保守方式处理
}

RECYCLE, KITCHEN, HAZARD, OTHER = "可回收物", "厨余垃圾", "有害垃圾", "其他垃圾"
CATEGORIES = (RECYCLE, KITCHEN, HAZARD, OTHER)
INSTRUCTIONS = {
    RECYCLE: "保持清洁干燥后投入可回收物容器。",
    KITCHEN: "沥干水分，去除包装后投入厨余垃圾容器。",
    HAZARD: "保持物品及包装完整，投入有害垃圾容器；破损或泄漏时先咨询社区投放点。",
    OTHER: "投入其他垃圾容器。",
}


@dataclass(frozen=True)
class Rule:
    """追问选项对应的投放结果。"""
    id: str
    label: str
    category: str
    sources: tuple[str, ...]
    level: str = "A"
    instructions: str = ""


RULES = {r.id: r for r in [
    # 食物与包装
    Rule("food", "食物本身", KITCHEN, ("S1",)),
    Rule("can", "清空的金属罐（如八宝粥罐）", RECYCLE, ("S2", "S5")),
    Rule("food_bag", "食品包装袋", OTHER, ("S3",)),
    Rule("plastic_cup", "用过的一次性塑料杯或吸管", OTHER, ("S3", "S5")),
    Rule("coffee_grounds", "咖啡渣", KITCHEN, ("S3",)),
    Rule("drink_solids", "杯里剩下的珍珠等食物（液体倒掉）", KITCHEN, ("S5",)),
    Rule("takeout_box", "外卖或快餐饭盒", OTHER, ("S2", "S3")),
    # 骨头与壳
    Rule("small_bone", "鸡骨、鱼骨等小骨头", KITCHEN, ("S3", "S5")),
    Rule("large_bone", "大棒骨等大块骨头", OTHER, ("S3", "S5")),
    Rule("soft_shell", "花生壳、瓜子壳", KITCHEN, ("S3",)),
    Rule("hard_shell", "核桃壳等硬坚果壳、椰子壳或榴莲壳", OTHER, ("S3",)),
    Rule("shell", "贝壳", OTHER, ("S3",)),
    Rule("nut_kernel", "果仁", KITCHEN, ("S1",), "C"),
    # 纸、容器、餐具
    Rule("paper", "干净的纸箱、纸板或纸制品", RECYCLE, ("S2",), instructions="保持干燥，纸箱压扁后投放。"),
    Rule("polluted_paper", "被油污、食物或水明显污染的纸", OTHER, ("S2",)),
    Rule("plastic_bottle", "已清空的食品或日化塑料瓶、桶（饮料瓶、洗发水瓶、食用油桶等）", RECYCLE, ("S2",),
         instructions="倒空内容物，保持清洁干燥后投放。"),
    Rule("glass_bottle", "已清空的食品或日化玻璃瓶（调料瓶、酒瓶、化妆品玻璃瓶等）", RECYCLE, ("S2",),
         instructions="倒空并保持清洁，轻拿轻放。"),
    Rule("paper_box", "干净的纸盒", RECYCLE, ("S2",)),
    Rule("plastic_box", "干净的塑料盒（如保鲜盒、收纳盒）", RECYCLE, ("S2",)),
    Rule("medicine", "家庭废药品或其包装（药瓶、药盒）", HAZARD, ("S1",),
         instructions="连同包装投放；有残留的密封后投放。"),
    Rule("chemical", "油漆、溶剂、杀虫剂或消毒剂的瓶罐", HAZARD, ("S1",),
         instructions="保持容器完整，有残留时密闭后投放。"),
    Rule("disposable_cup", "一次性纸杯或塑料杯", OTHER, ("S2", "S3")),
    Rule("reusable_cup", "可反复使用的玻璃、金属或硬塑料杯", RECYCLE, ("S2",)),
    Rule("ceramic", "陶瓷制品", OTHER, ("S3", "S5"), "C", "有尖锐边角的包裹后投放。"),
    Rule("reusable_tableware", "金属、玻璃或硬塑料材质", RECYCLE, ("S2",), "C"),
    Rule("disposable_tableware", "一次性餐具", OTHER, ("S3",)),
    # 织物
    Rule("clean_textile", "干净的", RECYCLE, ("S2",)),
    Rule("dirty_textile", "严重污损的", OTHER, ("S2",)),
    # 有害垃圾相关
    Rule("alkaline", "普通 1 号、5 号、7 号碱性干电池", OTHER, ("S5", "S6")),
    Rule("battery", "纽扣电池、锂电池、充电电池、镉镍电池或铅蓄电池", HAZARD, ("S1", "S3", "S6"),
         instructions="单独投放到有害垃圾容器，轻放。"),
    Rule("battery_pack", "电池组（如手机电池、电动车电池）", HAZARD, ("S1", "S6"), "C"),
    Rule("lamp", "荧光灯管、节能灯或含水银的 LED 灯", HAZARD, ("S1", "S5"),
         instructions="连同包装或包裹后轻放。"),
    Rule("lamp_fixture", "灯具本体（灯管灯泡已取下）", RECYCLE, ("S2",), "C",
         "取下的荧光灯管、节能灯泡按有害垃圾单独投放。"),
    Rule("mercury", "含汞温度计或血压计", HAZARD, ("S1",)),
]}

# 选中后不自动分类、转人工确认的选项：选项 ID → (显示文字, 原因)
REVIEW_OPTIONS = {
    "other": ("以上都不是", "identity_or_conditions_unconfirmed"),
    "unknown": ("不清楚", "identity_or_conditions_unconfirmed"),
    "medical_or_lab": ("来自医疗或实验活动", "non_household_source"),
    "residue": ("有残留，无法清空", "rule_not_covered"),
    "lamp_other": ("其他类型的灯", "rule_not_covered"),
    "solar": ("太阳能板", "rule_not_covered"),
}
MIXED = ("mixed", "食物和包装混在一起")

# 追问主题 → (问题, 规则选项, 额外选项)
GROUPS = {
    "food_pack": ("投放的是食物本身还是包装？", ("food", "can", "food_bag", "plastic_cup"), (MIXED,)),
    "coffee": ("投放的是咖啡渣还是杯子？", ("coffee_grounds", "disposable_cup"), ()),
    "takeout": ("投放的是剩饭菜还是饭盒？", ("food", "takeout_box"), (MIXED,)),
    "milktea": ("投放的是杯里的剩余食物还是杯子？", ("drink_solids", "plastic_cup"), (MIXED,)),
    "bone_type": ("是哪种骨头？", ("small_bone", "large_bone"), ()),
    "shell_type": ("是哪种壳？", ("soft_shell", "hard_shell", "shell"), ()),
    "nut": ("投放的是果仁还是壳？", ("nut_kernel", "soft_shell", "hard_shell"), ()),
    "paper_state": ("是否被油污、食物或水明显污染？", ("paper", "polluted_paper"), ()),
    "container": ("原来装的是什么？是否已清空？", ("plastic_bottle", "glass_bottle", "medicine", "chemical"),
                  ("residue",)),
    "box_use": ("是什么材质、原来装什么？", ("paper_box", "plastic_box", "medicine", "takeout_box"), ()),
    "cup": ("是一次性杯子还是可反复使用的杯子？", ("disposable_cup", "reusable_cup", "ceramic"), ()),
    "tableware": ("是什么材质？", ("reusable_tableware", "ceramic", "disposable_tableware"), ()),
    "textile_state": ("是否严重污损？", ("clean_textile", "dirty_textile"), ()),
    "battery_type": ("是哪种电池？", ("alkaline", "battery"), ()),
    "lamp_type": ("是哪种灯？", ("lamp",), ("lamp_other",)),
    "battery_panel": ("是电池组还是太阳能板？", ("battery_pack",), ("solar",)),
    "desk_lamp": ("灯管或灯泡还在灯里吗？要投放的是哪部分？", ("lamp_fixture", "lamp"), ()),
    # 以下只由包装文字（OCR）触发
    "medicine_origin": ("这是家庭产生的废药品或其包装吗？", ("medicine",), ("medical_or_lab",)),
    "mercury_type": ("能从标识确认它含汞吗？", ("mercury",), ()),
    "chemical_use": ("这是家庭使用的哪种用品或其包装？", ("chemical",), ()),
}
HAZARD_TOPICS = {"medicine_origin", "battery_type", "lamp_type", "mercury_type", "chemical_use", "battery_panel"}


@dataclass(frozen=True)
class Item:
    """Garbage265 的一个物品类别在北京的分类。

    category：北京分类；“special”表示需专门回收，“review”表示待人工确认。
    level：依据等级 A/B/C/D/X（见 BASIS）。
    need：must 必须追问 / suggest 建议追问 / tip 只给投放提示 / none 无需追问。
    topic：追问主题（GROUPS 的键）。
    """
    category: str
    level: str
    sources: tuple[str, ...]
    need: str
    topic: str | None
    note: str


# 由对照表生成：garbage265_beijing.csv，核对日期 CHECKED。
ITEMS = {
    '残渣剩饭': Item('厨余垃圾', 'A', ('S1',), 'tip', None, '沥干水分、去除包装后投放'),
    '菜根菜叶': Item('厨余垃圾', 'A', ('S1',), 'tip', None, '沥干水分、去除包装后投放'),
    '蔬菜': Item('厨余垃圾', 'A', ('S1',), 'tip', None, '沥干水分、去除包装后投放'),
    '饼干': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '沥干水分、去除包装后投放'),
    '面包': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '沥干水分、去除包装后投放'),
    '巧克力': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '沥干水分、去除包装后投放'),
    '肉类': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '沥干水分、去除包装后投放'),
    '苹果': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '橙子': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '菠萝': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '果皮': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '茶叶': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '甘蔗': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '瓜子': Item('厨余垃圾', 'A', ('S3',), 'tip', None, '《公园分类指导意见》点名其果皮/渣/瓜子壳；沥干水分、去除包装后投放'),
    '蛋': Item('厨余垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名鸡蛋壳，蛋壳也属厨余'),
    '鱼骨': Item('厨余垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名鱼刺鱼骨头'),
    '咖啡': Item('厨余垃圾', 'A', ('S3',), 'suggest', 'coffee', '《公园分类指导意见》点名咖啡渣'),
    '水果': Item('厨余垃圾', 'B', ('S5',), 'tip', None, '沥干水分、去除包装后投放'),
    '蛋糕': Item('厨余垃圾', 'B', ('S5',), 'tip', None, '沥干水分、去除包装后投放'),
    '豆': Item('厨余垃圾', 'B', ('S5',), 'tip', None, '沥干水分、去除包装后投放'),
    '鸡翅': Item('厨余垃圾', 'B', ('S5',), 'none', None, '新京报120种清单点名吃剩的鸡爪；小骨头属厨余'),
    '烤鸡': Item('厨余垃圾', 'B', ('S5',), 'none', None, '新京报120种清单点名吃剩的鸡爪；小骨头属厨余'),
    '菠萝蜜': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '草莓': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '地瓜': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '豆腐': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '番茄': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '粉条': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '哈密瓜': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '火龙果': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '辣椒': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '梨': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '萝卜': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '蘑菇': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '圣女果': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '薯条': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '蒜': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '属《条例》厨余定义中的瓜果、蔬菜或废弃食物；沥干水分、去除包装后投放'),
    '蛋挞': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '锡纸托分开投放'),
    '冰糖葫芦': Item('厨余垃圾', 'C', ('S1',), 'tip', None, '竹签属其他垃圾（《公园分类指导意见》点名竹筷子、牙签），先分开'),
    '八宝粥': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '冰激凌': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '果冻': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '薯片': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '火腿': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '肠': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '泡菜': Item('厨余垃圾', 'C', ('S1', 'S3'), 'must', 'food_pack', '此类商品照片常带包装；食物属厨余，包装按材质另投（《公园分类指导意见》点名食品包装袋属其他垃圾）'),
    '骨头': Item('厨余垃圾', 'A', ('S3', 'S5'), 'must', 'bone_type', '《公园分类指导意见》：鱼刺鱼骨头厨余、大棒骨其他；新京报120种清单：鸡骨头厨余、大棒骨其他'),
    '核桃': Item('厨余垃圾', 'A', ('S3',), 'must', 'nut', '《公园分类指导意见》：硬坚果壳属其他垃圾；花生壳、瓜子壳属厨余'),
    '坚果': Item('厨余垃圾', 'A', ('S3',), 'must', 'nut', '《公园分类指导意见》：硬坚果壳属其他垃圾；花生壳、瓜子壳属厨余'),
    '秸秆杯': Item('其他垃圾', 'C', ('S3',), 'none', None, '数据集标为厨余；北京按一次性餐具归其他垃圾（《公园分类指导意见》点名一次性餐具）——与数据集不一致，建议再核'),
    '秸秆碗': Item('其他垃圾', 'C', ('S3',), 'none', None, '数据集标为厨余；北京按一次性餐具归其他垃圾（《公园分类指导意见》点名一次性餐具）——与数据集不一致，建议再核'),
    '书': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '信封': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '纸牌': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '快递纸袋': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '购物纸袋': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '档案袋': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '日历': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '吊牌': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '保鲜膜内芯': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '废纸类；保持清洁干燥后投放'),
    '纸箱': Item('可回收物', 'A', ('S2', 'S5'), 'must', 'paper_state', '《可回收物目录》：污损纸张不宜回收，按其他垃圾'),
    '纸制品': Item('可回收物', 'A', ('S2',), 'suggest', 'paper_state', '《可回收物目录》：餐巾纸、一次性纸杯、污损纸张不属可回收物'),
    '饮料瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '洗发水瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '食用油桶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '调料瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '玻璃瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '酒瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '护肤品瓶': Item('可回收物', 'A', ('S2',), 'must', 'container', '《可回收物目录》点名食品及日化用品塑料瓶桶、调料瓶、酒瓶、化妆品玻璃瓶；装过药品或化学品的按有害垃圾'),
    '金属罐': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '新京报120种清单点名易拉罐；清空后投放'),
    '奶粉罐': Item('可回收物', 'A', ('S2', 'S5'), 'tip', None, '新京报120种清单点名易拉罐；清空后投放'),
    '玻璃器皿': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '玻璃制品': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '玻璃壶': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '玻璃球': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '鱼缸': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '放大镜': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》点名玻璃杯、家用玻璃制品；碎玻璃口径不一（新京报120种清单可回收、《公园分类指导意见》公园场景其他），有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '锅': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '刀': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '钉子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '不锈钢制品': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '金属制品': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '铝制用品': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '钥匙': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '镊子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '磁铁': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废金属；刀具等有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '塑料制品': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废塑料；保持清洁干燥后投放'),
    '塑料碗盆': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废塑料；保持清洁干燥后投放'),
    '泡沫板': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废塑料；保持清洁干燥后投放'),
    '衣架': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'tip', None, '废塑料；保持清洁干燥后投放'),
    '玩具': Item('可回收物', 'A', ('S3', 'S5'), 'tip', None, '带电池的先取出电池'),
    '衣服': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '裤子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '裙子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '毯子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '枕头': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '布制品': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '帽子': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '耳套': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '包': Item('可回收物', 'A', ('S2', 'S3', 'S5'), 'suggest', 'textile_state', '废织物；《可回收物目录》：内衣和严重污损的织物不宜回收'),
    '鞋': Item('可回收物', 'A', ('S2', 'S5'), 'none', None, '《可回收物目录》点名鞋靴，新京报120种清单点名旧皮鞋'),
    '鞋子': Item('可回收物', 'A', ('S2', 'S5'), 'none', None, '《可回收物目录》点名鞋靴，新京报120种清单点名旧皮鞋'),
    '拖鞋': Item('可回收物', 'A', ('S2', 'S5'), 'none', None, '《可回收物目录》点名鞋靴，新京报120种清单点名旧皮鞋'),
    '拉杆箱': Item('可回收物', 'B', ('S5',), 'none', None, '新京报120种清单点名行李箱'),
    '手机': Item('可回收物', 'B', ('S5',), 'tip', None, '电子废弃物；如带可拆卸电池，先取出电池单独投放'),
    '鼠标': Item('可回收物', 'B', ('S5',), 'tip', None, '电子废弃物；如带可拆卸电池，先取出电池单独投放'),
    '话筒': Item('可回收物', 'B', ('S5',), 'tip', None, '电子废弃物；如带可拆卸电池，先取出电池单独投放'),
    '电熨斗': Item('可回收物', 'B', ('S5',), 'tip', None, '电子废弃物；如带可拆卸电池，先取出电池单独投放'),
    '插线板': Item('可回收物', 'B', ('S5',), 'none', None, '新京报120种清单点名插座；属电子废弃物'),
    '插头电线': Item('可回收物', 'B', ('S5',), 'none', None, '新京报120种清单点名插座；属电子废弃物'),
    '充电线': Item('可回收物', 'B', ('S5',), 'none', None, '新京报120种清单点名插座；属电子废弃物'),
    '充电头': Item('可回收物', 'B', ('S5',), 'none', None, '新京报120种清单点名插座；属电子废弃物'),
    '打印机': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '显示器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电路板': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '吹风机': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电磁炉': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电动卷发棒': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电动剃须刀': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电饭煲': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电风扇': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电话': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '电子秤': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '豆浆机': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '耳机': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '计算器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '键盘': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '空气加湿器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '空气净化器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '路由器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '闹铃': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '收音机': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '手表': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '手电筒': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '扫地机器人': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '网卡': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '音响': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '遥控器': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '充电牙刷': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '体重秤': Item('可回收物', 'A', ('S2',), 'tip', None, '《可回收物目录》：废弃电器电子产品；如带可拆卸电池，先取出电池单独投放'),
    '燃气灶': Item('可回收物', 'A', ('S2',), 'tip', None, '属废弃电器；确认燃气已关闭断开'),
    '电视机': Item('可回收物', 'A', ('S2', 'S1b'), 'tip', None, '《可回收物目录》：废弃电器电子产品；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '太阳能热水器': Item('可回收物', 'A', ('S2', 'S1b'), 'tip', None, '《可回收物目录》：废弃电器电子产品；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '沙发': Item('可回收物', 'A', ('S2', 'S1b'), 'tip', None, '《可回收物目录》：废弃大件家具；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '桌子': Item('可回收物', 'A', ('S2', 'S1b'), 'tip', None, '《可回收物目录》：废弃大件家具；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '凳子': Item('可回收物', 'A', ('S2', 'S1b'), 'tip', None, '《可回收物目录》：废弃大件家具；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '单车': Item('可回收物', 'C', ('S2', 'S1b'), 'tip', None, '金属为主，按废金属类推；体积大时属大件，按《条例》第三十四条单独堆放到指定地点或预约回收'),
    '保温杯': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '水壶': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '热水瓶': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '量杯': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '尺子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '订书机': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '打气筒': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '搓衣板': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '打包绳': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '尼龙绳': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '呼啦圈': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '瑜伽球': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '乒乓球拍': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '棋子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '铅球': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '地球仪': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '望远镜': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '模具': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '笼子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '过滤网': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '盖子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '锅盖': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '垫子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '餐垫': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '卡片': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '地铁票': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '饰品': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '手链': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '雨伞': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '铁丝球': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '灯罩': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '箱子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木雕': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木棍': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木桶': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木制切菜板': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木质锅铲': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '木质梳子': Item('可回收物', 'C', ('S1', 'S2'), 'none', None, '按材质（金属、塑料、玻璃、织物、木）类推为可回收物，北京文件未逐项点名；木制小件、铁丝球等依据较弱，建议再核'),
    '袜子': Item('可回收物', 'D', (), 'suggest', 'textile_state', '北京文件未点名；《可回收物目录》只排除内衣和严重污损织物'),
    '登机牌': Item('可回收物', 'D', (), 'none', None, '若为热敏纸，多地按其他垃圾，北京未找到依据'),
    '轮胎': Item('可回收物', 'D', (), 'none', None, '汽车轮胎一般不属于居民生活垃圾，北京未找到依据'),
    '台灯': Item('可回收物', 'C', ('S2', 'S1'), 'suggest', 'desk_lamp', '灯具按废弃电器类推；灯管灯泡按《条例》属有害垃圾'),
    '水杯': Item('可回收物', 'A', ('S2', 'S3'), 'must', 'cup', '《可回收物目录》、《公园分类指导意见》：一次性纸杯属其他垃圾；《可回收物目录》点名玻璃杯可回收'),
    '餐具': Item('可回收物', 'A', ('S2', 'S3'), 'must', 'tableware', '《公园分类指导意见》：陶瓷制品碎片、一次性餐具属其他垃圾；新京报120种清单点名砂锅属其他垃圾'),
    '盘子': Item('可回收物', 'A', ('S2', 'S3'), 'must', 'tableware', '《公园分类指导意见》：陶瓷制品碎片、一次性餐具属其他垃圾；新京报120种清单点名砂锅属其他垃圾'),
    '碗': Item('可回收物', 'A', ('S2', 'S3'), 'must', 'tableware', '《公园分类指导意见》：陶瓷制品碎片、一次性餐具属其他垃圾；新京报120种清单点名砂锅属其他垃圾'),
    '烟灰缸': Item('可回收物', 'A', ('S2', 'S3'), 'must', 'tableware', '《公园分类指导意见》：陶瓷制品碎片、一次性餐具属其他垃圾；新京报120种清单点名砂锅属其他垃圾'),
    '盒子': Item('可回收物', 'A', ('S1', 'S2'), 'must', 'box_use', ''),
    '果冻杯': Item('其他垃圾', 'A', ('S3',), 'none', None, '数据集标为可回收；按《公园分类指导意见》一次性餐具、奶茶杯类推为其他垃圾——与数据集不一致'),
    '充电宝': Item('有害垃圾', 'X', ('S6', 'S7'), 'tip', None, '北京政法网：电子产品锂电池属有害垃圾；宝典转述：充电宝属可回收物。两者冲突，保守按有害垃圾单独投放（用户也难以确认电芯类型，故不追问）'),
    '燃气瓶': Item('special', 'A', ('S2',), 'none', None, '《可回收物目录》点名液化气罐不宜列为可回收物；应联系燃气供应单位回收——与数据集不一致'),
    '灭火器': Item('special', 'D', (), 'none', None, '压力容器，北京未找到依据；建议联系销售或专业单位回收——与数据集不一致'),
    '暖宝宝': Item('其他垃圾', 'D', (), 'none', None, '用过的暖宝宝多地按其他垃圾，北京未找到依据——与数据集不一致，建议再核'),
    '餐巾纸': Item('其他垃圾', 'A', ('S2', 'S3'), 'none', None, ''),
    '湿纸巾': Item('其他垃圾', 'A', ('S2', 'S3'), 'none', None, ''),
    '卫生纸': Item('其他垃圾', 'A', ('S2', 'S3'), 'none', None, ''),
    '塑料袋': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '口罩': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '一次性棉签': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '牙签': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '竹筷': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '串串竹签': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '创可贴': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '干燥剂': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '烟蒂': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '打火机': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '一次性杯子': Item('其他垃圾', 'A', ('S3',), 'none', None, '《公园分类指导意见》点名'),
    '除湿袋': Item('其他垃圾', 'A', ('S3',), 'none', None, '按《公园分类指导意见》点名的干燥剂'),
    '破碎陶瓷': Item('其他垃圾', 'A', ('S3', 'S4'), 'tip', None, '《公园分类指导意见》点名陶瓷制品碎片；有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '茶壶碎片': Item('其他垃圾', 'A', ('S3', 'S4'), 'tip', None, '《公园分类指导意见》点名陶瓷制品碎片；有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '陶瓷': Item('其他垃圾', 'A', ('S3', 'S4'), 'tip', None, '《公园分类指导意见》点名陶瓷制品碎片；有尖锐边角的应包裹后投放（丰台区投放要求）'),
    '笔': Item('其他垃圾', 'B', ('S5',), 'none', None, '新京报120种清单点名'),
    '餐盒': Item('其他垃圾', 'A', ('S2', 'S3'), 'must', 'takeout', '《公园分类指导意见》点名快餐盒；《可回收物目录》：沾有污渍的一次性塑料饭盒属其他垃圾'),
    '奶茶杯': Item('其他垃圾', 'A', ('S3', 'S5'), 'must', 'milktea', '《公园分类指导意见》点名奶茶杯；新京报120种清单点名奶茶中的珍珠属厨余'),
    '厨房抹布': Item('其他垃圾', 'A', ('S2',), 'suggest', 'textile_state', '《可回收物目录》：严重污损的织物不宜回收；干净毛巾可按织物回收（数据集毛巾标为其他）'),
    '毛巾': Item('其他垃圾', 'A', ('S2',), 'suggest', 'textile_state', '《可回收物目录》：严重污损的织物不宜回收；干净毛巾可按织物回收（数据集毛巾标为其他）'),
    '厨房手套': Item('其他垃圾', 'A', ('S2',), 'none', None, '《可回收物目录》点名一次性手套不宜回收'),
    '酒精棉': Item('其他垃圾', 'C', ('S3',), 'none', None, '按《公园分类指导意见》点名的棉签类推'),
    '大龙虾头': Item('厨余垃圾', 'B', ('S3', 'S5'), 'none', None, '新京报120种清单点名小龙虾、螃蟹壳属厨余，《公园分类指导意见》点名虾壳属厨余——与数据集（其他）不一致，大龙虾硬壳建议再核'),
    '百洁布': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '便利贴': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '彩票': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '苍蝇拍': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '草帽': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '唱片': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '车票': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '搓澡巾': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '起泡网': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '电蚊香': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '电影票': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '防霉防蛀片': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '鸡毛掸': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '胶带': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '空调滤芯': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '图钉': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '涂改带': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '牙刷': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '眼镜': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '眼镜布': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '验孕棒': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    'U型回形针': Item('其他垃圾', 'D', ('S1',), 'none', None, '北京文件未逐项点名；按《条例》“难以辨识类别的生活垃圾”可作其他垃圾，暂沿用数据集'),
    '胶水包装': Item('其他垃圾', 'D', ('S1',), 'none', None, '《条例》有害垃圾未列胶水，暂沿用数据集；有残留时建议密封'),
    '杀虫剂': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '药片': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '药品包装': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '药瓶': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '温度计': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '血压计': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '蓄电池': Item('有害垃圾', 'A', ('S1',), 'none', None, '《条例》第七十八条点名（废温度计、废血压计、废药品及其包装物、铅蓄电池等）'),
    '药膏': Item('有害垃圾', 'A', ('S1', 'S5'), 'none', None, '新京报120种清单点名止痛膏'),
    '纽扣电池': Item('有害垃圾', 'A', ('S3', 'S6'), 'none', None, '《公园分类指导意见》、北京政法网点名'),
    '指甲油': Item('有害垃圾', 'A', ('S3', 'S5'), 'none', None, '《公园分类指导意见》点名废指甲油'),
    '电池': Item('有害垃圾', 'A', ('S1', 'S5', 'S6'), 'must', 'battery_type', '新京报120种清单、北京政法网：普通干电池属其他垃圾；锂电池、纽扣电池、充电电池属有害垃圾'),
    '灯': Item('有害垃圾', 'A', ('S1', 'S5'), 'must', 'lamp_type', '《条例》点名荧光灯管、节能灯；新京报120种清单点名含水银的LED灯'),
    '电池板': Item('review', 'D', (), 'must', 'battery_panel', '标签含义不明确'),
    '胶水': Item('review', 'D', ('S1',), 'suggest', None, '数据集标为有害，但《条例》有害垃圾未列胶水'),
    '果壳': Item('厨余垃圾', 'A', ('S3',), 'must', 'shell_type', '《公园分类指导意见》：花生壳、瓜子壳属厨余；硬坚果壳、椰子壳、榴莲壳、贝壳属其他'),
}

# 预测结果里可能出现的别名
ALIASES = {"剩饭": "残渣剩饭", "剩菜": "残渣剩饭", "剩饭剩菜": "残渣剩饭", "菜帮菜叶": "菜根菜叶",
           "瓜果皮核": "果皮", "大棒骨": "骨头", "鸡骨": "骨头", "贝壳": "果壳", "陶瓷碎片": "破碎陶瓷",
           "食品包装袋": "塑料袋", "塑料瓶": "饮料瓶", "纸板": "纸箱"}

OCR_PATTERNS = {
    "medicine_origin": r"(?<![A-Za-z])OTC(?![A-Za-z])|国药准字",
    # 只写“电池”的说明文字（如“请装入电池”）很常见，排除这类动宾搭配。
    "battery_type": r"镉镍|镍镉|氧化汞|铅蓄|铅酸|(?<!装入)(?<!放入)(?<!更换)(?<!安装)(?<!取出)电池",
    "lamp_type": r"荧光灯|节能灯",
    "mercury_type": r"水银|含汞",
    "chemical_use": r"杀虫剂|胶粘剂|胶黏剂|油漆|消毒剂",
    "container": r"护肤品",
}
OCR_HAZARD_TOPICS = {"medicine_origin", "battery_type", "lamp_type", "mercury_type", "chemical_use"}
# 会让系统先确认物品身份的模型风险线索。hazard_tail_unchecked 只记录，不单独触发提问。
IDENTITY_RISKS = {"hazard_top1", "hazard_score", "detector_category_conflict"}
MAX_IDENTITY_OPTIONS = 5


def valid_score(score):
    return (isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and 0 <= score <= 1)


def item_name(candidate):
    item = candidate.get("item") or candidate.get("label", "").partition("-")[2]
    if not isinstance(item, str) or not item.strip():
        raise ValueError("每个模型候选都需要 item 或形如‘大类-物品’的 label")
    return item.strip()


def category_of(candidate):
    category = candidate.get("waste_category") or candidate.get("label", "").partition("-")[0]
    return OTHER if category == "其它垃圾" else category


def is_hazard(candidate):
    return category_of(candidate) == HAZARD


def lookup(item):
    return ITEMS.get(item) or ITEMS.get(ALIASES.get(item, ""))


def topic_of(item):
    entry = lookup(item)
    return entry.topic if entry else None


def hazard_related(item):
    """物品本身在北京可能属于有害垃圾：此时直接问它的主题或直接给结果，不必先确认身份。"""
    entry = lookup(item)
    return bool(entry) and (entry.category in (HAZARD, "review") or entry.topic in HAZARD_TOPICS)


def source_list(keys):
    return [{"name": SOURCES[k][0], "url": SOURCES[k][1]} for k in dict.fromkeys(keys)]


@dataclass
class ObjectState:
    object_id: str
    item: str
    candidates: list[str]
    triggers: list[str]
    ocr_themes: list[str]
    uncertain: bool
    score_scope: str
    hazard_candidate: str | None = None
    dataset_categories: dict[str, str] = field(default_factory=dict)
    identity_options: list[str] = field(default_factory=list)
    confirmed_item: str | None = None
    history: list[dict] = field(default_factory=list)
    pending: dict | None = None


class Questioning:
    """每张图片创建一个实例，add 和 answer 各自返回逐目标结果。

    ask_suggested=True 时，“建议追问”的物品也会追问；默认只追问“必须追问”的物品，
    其余直接按主要分类给出结果，并在投放建议里写明例外情况。
    """

    def __init__(self, *, min_score=0.55, min_margin=0.15, hazard_threshold=0.01,
                 ocr_threshold=0.7, max_questions=3, ask_suggested=False):
        if not all(map(valid_score, (min_score, min_margin, hazard_threshold, ocr_threshold))):
            raise ValueError("分数阈值必须在 0 到 1 之间")
        if not isinstance(max_questions, int) or max_questions < 1:
            raise ValueError("问题预算至少为 1")
        self.min_score = min_score
        self.min_margin = min_margin
        self.hazard_threshold = hazard_threshold
        self.ocr_threshold = ocr_threshold
        self.max_questions = max_questions
        self.ask_suggested = ask_suggested
        self.objects: dict[str, ObjectState] = {}

    def add(self, prediction_json, *, object_id="whole_image", ocr_lines=(),
            full_scores=None, class_labels=None, detector_category=None):
        """detector_category：检测模块对同一个框给出的四大类，可选，用于交叉校验。"""
        if not isinstance(object_id, str) or not object_id or object_id in self.objects:
            raise ValueError("当前图片内 object_id 必须非空且唯一")
        if not isinstance(prediction_json, dict) or not isinstance(
                prediction_json.get("prediction"), dict):
            raise ValueError("预测 JSON 需要包含 prediction 对象")
        if detector_category is not None and detector_category not in CATEGORIES:
            raise ValueError(f"detector_category 必须是 {'、'.join(CATEGORIES)} 之一")
        best = prediction_json["prediction"]
        topk = prediction_json.get("topk") or [best]
        if topk[0] != best or not all(valid_score(c.get("score")) for c in topk):
            raise ValueError("topk 首项必须是 prediction，且所有 score 合法")
        if any(a["score"] < b["score"] for a, b in zip(topk, topk[1:])):
            raise ValueError("topk 必须按分数降序排列")
        candidates = list(dict.fromkeys(map(item_name, topk)))
        themes = {
            theme for line in ocr_lines
            if line.get("object_id") == object_id and valid_score(line.get("score"))
            and line["score"] >= self.ocr_threshold
            for theme, pattern in OCR_PATTERNS.items()
            if re.search(pattern, line.get("text", ""), re.IGNORECASE)
        }
        scope = "topk_lower_bound"
        hazard_pairs = [(c["score"], item_name(c)) for c in topk if is_hazard(c)]
        if full_scores is not None or class_labels is not None:
            if (full_scores is None or class_labels is None
                    or len(full_scores) != 265 or len(class_labels) != 265
                    or not all(map(valid_score, full_scores))
                    or not math.isclose(sum(full_scores), 1, abs_tol=1e-4)
                    or len(set(class_labels)) != 265):
                raise ValueError("完整分数模式需要同一次推理的265维softmax与类别清单")
            for candidate in topk:
                index = candidate.get("class_id")
                if (type(index) is not int or not 0 <= index < 265
                        or candidate.get("label") != class_labels[index]
                        or not math.isclose(candidate["score"], full_scores[index], abs_tol=1e-6)):
                    raise ValueError("topk 与完整分数或类别顺序不一致")
            scope = "full_softmax"
            hazard_pairs = [(score, label.partition("-")[2])
                            for label, score in zip(class_labels, full_scores)
                            if label.partition("-")[0] == HAZARD]

        triggers = []
        if themes:
            triggers.append("ocr_keyword")
        if is_hazard(best):
            triggers.append("hazard_top1")
        top_hazard = max(hazard_pairs, default=None)
        if top_hazard and top_hazard[0] >= self.hazard_threshold:
            triggers.append("hazard_score")
        if scope == "topk_lower_bound" and not (
            len(topk) == 265 or topk[-1]["score"] < self.hazard_threshold
        ):
            # 仅记录：列表外可能还有达到阈值的有害细类。接入完整分数后此项消失。
            triggers.append("hazard_tail_unchecked")
        if detector_category is not None and category_of(best) != detector_category:
            triggers.append("detector_category_conflict")

        margin = best["score"] - topk[1]["score"] if len(topk) > 1 else None
        uncertain = best["score"] < self.min_score or margin is None or margin < self.min_margin
        state = ObjectState(
            object_id, candidates[0], candidates, triggers, sorted(themes), uncertain, scope,
            hazard_candidate=top_hazard[1] if "hazard_score" in triggers else None,
        )
        for candidate in topk:
            state.dataset_categories.setdefault(item_name(candidate), category_of(candidate))
        if state.hazard_candidate:
            state.dataset_categories.setdefault(state.hazard_candidate, HAZARD)
        self.objects[object_id] = state
        return self._next(state)

    def answer(self, object_id, question_id, answer_id):
        state = self.objects.get(object_id)
        if state is None:
            raise ValueError("当前图片中没有这个物品 ID")
        question = state.pending
        if question is None or question["question_id"] != question_id:
            raise ValueError("问题已失效，或不属于当前物品")
        if answer_id not in {option["id"] for option in question["options"]}:
            raise ValueError("无效的回答选项")
        state.history.append({"topic": question["topic"], "answer_id": answer_id})
        state.pending = None
        return self._next(state)

    # ---------- 结果 ----------
    def _result(self, state, status, **kwargs):
        return {
            "status": status, "object_id": state.object_id, "model_item": state.item,
            "confirmed_item": state.confirmed_item,
            "category": None, "triggers": list(state.triggers),
            "hazard_score_scope": state.score_scope, "rule_version": VERSION,
            "history": [step.copy() for step in state.history], **kwargs,
        }

    def _review(self, state, reason, instructions="请确认物品身份或咨询小区垃圾分类管理人员。", **kwargs):
        return self._result(state, "needs_review", reason=reason, instructions=instructions, **kwargs)

    def _resolve_rule(self, state, rule_id):
        rule = RULES[rule_id]
        return self._result(
            state, "resolved", category=rule.category, basis=BASIS[rule.level],
            evidence_level=rule.level, rule_id=rule.id, rule_label=rule.label,
            instructions=rule.instructions or INSTRUCTIONS[rule.category],
            sources=source_list(rule.sources), checked=CHECKED,
        )

    def _resolve_item(self, state, item, entry):
        if entry.category == "special":
            return self._review(state, "special_disposal", instructions=entry.note,
                                sources=source_list(entry.sources))
        if entry.category == "review":
            return self._review(state, "rule_not_covered", note=entry.note)
        instructions = INSTRUCTIONS[entry.category]
        if entry.need in ("tip", "suggest") and entry.note:
            instructions += f"（提示：{entry.note}）"
        return self._result(
            state, "resolved", category=entry.category, basis=BASIS[entry.level],
            evidence_level=entry.level, rule_id=f"item:{item}", rule_label=item,
            instructions=instructions, note=entry.note,
            sources=source_list(entry.sources), checked=CHECKED,
        )

    def _fallback(self, state, item):
        """对照表之外的物品：按数据集标注的大类给结果，有害类别不兜底。"""
        category = state.dataset_categories.get(item)
        if category not in CATEGORIES or category == HAZARD:
            return self._review(state, "no_rule_for_item")
        return self._result(
            state, "resolved", category=category, basis="dataset_label", evidence_level="D",
            rule_id=None, rule_label=f"数据集标注：{category}-{item}",
            instructions=INSTRUCTIONS[category] + "（依据数据集标签，未按北京规则核实。）",
            sources=[], checked=None,
        )

    # ---------- 提问 ----------
    def _ask(self, state, topic, prompt, options, trigger=None):
        if len(state.history) >= self.max_questions:
            return self._review(state, "question_budget_exhausted")
        if trigger and trigger not in state.triggers:
            state.triggers.append(trigger)
        options = [*options, *((key, REVIEW_OPTIONS[key][0]) for key in ("other", "unknown"))]
        state.pending = {
            "question_id": f"{state.object_id}:{len(state.history) + 1}",
            "topic": topic, "text": prompt,
            "options": [{"id": key, "text": label} for key, label in options],
        }
        return self._result(state, "ask", question=deepcopy(state.pending))

    def _ask_group(self, state, topic, trigger=None):
        prompt, ids, extras = GROUPS[topic]
        options = [(key, RULES[key].label) for key in ids]
        for extra in extras:
            options.append(extra if isinstance(extra, tuple) else (extra, REVIEW_OPTIONS[extra][0]))
        return self._ask(state, topic, prompt, options, trigger)

    def _ask_identity(self, state, trigger):
        items = state.candidates[:MAX_IDENTITY_OPTIONS]
        hazard = state.hazard_candidate
        if hazard and hazard not in items:
            items = [*items[:MAX_IDENTITY_OPTIONS - 1], hazard]
        state.identity_options = items
        options = [(f"candidate_{i}", item) for i, item in enumerate(items)]
        return self._ask(state, "item_identity", "你要投放的是哪件物品？", options, trigger)

    def _next(self, state):
        answers = {step["topic"]: step["answer_id"] for step in state.history}
        for answer in answers.values():
            if answer in REVIEW_OPTIONS:
                return self._review(state, REVIEW_OPTIONS[answer][1])
        if MIXED[0] in answers.values():
            return self._result(
                state, "needs_review", reason="split_and_reclassify",
                instructions="请把食物和包装分开，再分别识别投放。",
            )

        # 1. 包装文字：同一物品的多条高分文字指向不同有害主题时转人工。
        ocr_hazard = [t for t in state.ocr_themes if t in OCR_HAZARD_TOPICS]
        if len(ocr_hazard) > 1:
            return self._review(state, "conflicting_ocr_themes")
        if ocr_hazard:
            topic = ocr_hazard[0]
            model_topic = topic_of(state.item)
            if ("hazard_top1" in state.triggers and model_topic in HAZARD_TOPICS
                    and model_topic != topic):
                return self._review(state, "conflicting_model_ocr_risk")
            if topic not in answers:
                return self._ask_group(state, topic)
            return self._resolve_rule(state, answers[topic])

        # 2. 确认身份：模型拿不准，或有风险线索而首选物品本身与有害垃圾无关。
        risk = IDENTITY_RISKS.intersection(state.triggers)
        needs_identity = state.uncertain or (risk and not hazard_related(state.item))
        if needs_identity and "item_identity" not in answers:
            trigger = "uncertain_identity" if state.uncertain else "risk_identity_check"
            return self._ask_identity(state, trigger)

        item = state.item
        if "item_identity" in answers:
            index = int(answers["item_identity"].removeprefix("candidate_"))
            item = state.confirmed_item = state.identity_options[index]
        entry = lookup(item)

        # 3. 状态追问：分类取决于物品状态时提问。
        topic = None
        if entry and entry.topic and (entry.need == "must" or self.ask_suggested):
            topic = entry.topic
        if topic is None and entry is None and "container" in state.ocr_themes:
            topic = "container"
        if topic is not None:
            if topic not in answers:
                return self._ask_group(state, topic, "missing_rule_condition")
            return self._resolve_rule(state, answers[topic])

        # 4. 给出结果。
        if entry is None:
            return self._fallback(state, item)
        return self._resolve_item(state, item, entry)


def synthetic_prediction(item, category=RECYCLE, score=0.98):
    best = {"item": item, "waste_category": category, "score": score}
    second = {"item": "箱子", "waste_category": RECYCLE, "score": 0.004}
    return {"prediction": best, "topk": [best, second]}


def run_demo():
    engine = Questioning()
    rows = []
    for obj, item, category, option in [
        ("box", "纸箱", RECYCLE, "paper"),
        ("food", "残渣剩饭", KITCHEN, None),
        ("bone", "骨头", KITCHEN, "large_bone"),
        ("medicine", "药片", HAZARD, None),
    ]:
        result = engine.add(synthetic_prediction(item, category), object_id=obj)
        rows.append(result)
        if option:
            rows.append(engine.answer(obj, result["question"]["question_id"], option))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", nargs="?", type=Path, help="组员脚本输出的JSON路径")
    parser.add_argument("--ocr", type=Path, help="已关联到 whole_image 的OCR行JSON数组")
    parser.add_argument("--detector-category", choices=CATEGORIES,
                        help="检测模块对同一目标给出的四大类，用于交叉校验")
    parser.add_argument("--ask-suggested", action="store_true", help="“建议追问”的物品也追问")
    parser.add_argument("--demo", action="store_true", help="运行四类交互演示")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
        return
    if args.prediction is None:
        parser.error("请提供预测JSON文件，或使用 --demo")
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    ocr = json.loads(args.ocr.read_text(encoding="utf-8")) if args.ocr else []
    engine = Questioning(ask_suggested=args.ask_suggested)
    result = engine.add(prediction, ocr_lines=ocr, detector_category=args.detector_category)
    while result["status"] == "ask":
        question = result["question"]
        print(question["text"])
        for option in question["options"]:
            print(f'  {option["id"]}  {option["text"]}')
        try:
            answer_id = input("请输入左侧选项 ID > ").strip()
        except EOFError:
            answer_id = "unknown"
        try:
            result = engine.answer(result["object_id"], question["question_id"], answer_id)
        except ValueError as error:
            print(error)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
