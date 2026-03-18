from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Item(BaseModel):
    """
    物品数据模型：字段尽量与原始 XML 属性一一对应，便于追溯与回填。
    """

    name: str = Field(..., description="物品名称（XML: <item name=...> 或同级属性）")
    displayname: Optional[str] = Field(
        default=None,
        description=(
            "显示名/描述（XML: displayname）。多数情况下与 name 相同；"
            "若与 name 不同，后续可将其当做物品描述传给大模型。"
        ),
    )

    type: Optional[str] = Field(default=None, description="大分类（XML: type；如 武器/防具/消耗品/收集品）")
    use: Optional[str] = Field(default=None, description="细分类型（XML: use；如 刀/长枪/手枪/材料/药剂等）")

    actiontype: Optional[str] = Field(
        default=None,
        description="动作类型（XML: actiontype；主要对 use=刀 的武器需要）",
    )
    weapontype: Optional[str] = Field(
        default=None,
        description='武器子类型（XML: <item weapontype="冲锋枪"> 这样的属性；仅枪械有）',
    )

    price: Optional[int] = Field(default=None, description="价格（XML: price）")

    level: int = Field(
        default=0,
        description="等级（XML: data.level；data 或 level 可能缺失，缺失则为 0）",
    )

    source_path: Optional[str] = Field(default=None, description="来源文件路径（便于排查解析差异）")
    raw: Optional[dict[str, Any]] = Field(
        default=None,
        description="保留必要的原始字段/扩展字段（用于兼容未来不同 XML 结构）",
    )

    @property
    def effective_displayname(self) -> Optional[str]:
        """
        若 displayname 与 name 相同则返回 None（通常不必传给大模型），否则返回 displayname。
        """

        if not self.displayname:
            return None
        if self.displayname == self.name:
            return None
        return self.displayname


class Task(BaseModel):
    id: int
    title: str = Field(..., description="任务标题 key（例如 $MAIN_TITLE_0）或直接标题")
    description: Optional[str] = Field(default=None, description="任务描述 key（例如 $MAIN_DESCRIPTION_0）")

    get_requirements: list[int] = Field(default_factory=list, description="接取前置任务 id 列表（agent 任务禁止 -1）")
    get_conversation: Optional[str] = None
    get_npc: Optional[str] = None

    finish_requirements: list[str] = Field(default_factory=list, description='通关要求数组（"关卡名#难度"）')
    finish_submit_items: list[str] = Field(default_factory=list, description='提交物品数组（"物品名#数量"）')
    finish_contain_items: list[str] = Field(default_factory=list, description='持有物品数组（"物品名#数量"）')
    finish_conversation: Optional[str] = None
    finish_npc: Optional[str] = None

    rewards: list[str] = Field(default_factory=list, description='奖励数组（"物品名#数量"）')
    announcement: Optional[str] = None
    chain: Optional[str] = None

    raw: Optional[dict[str, Any]] = None


class StageInfo(BaseModel):
    area: str = Field(..., description="stages 子目录名（例如 基地门口）")
    name: str = Field(..., description="关卡名（对应关卡 xml 文件名，不含 .xml）")
    type: Optional[str] = Field(default=None, description="关卡类型（__list__.xml: <Type>）")
    unlock_condition: Optional[int] = Field(default=None, description="解锁主线任务 id（__list__.xml: <UnlockCondition>）")
    description: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


class LootDrop(BaseModel):
    name: str
    min_count: int = 0
    max_count: int = 0


class LootCrate(BaseModel):
    identifier: str = Field(..., description="箱子类型标识（纸箱/资源箱/装备箱）")
    drops: list[LootDrop] = Field(default_factory=list)
    raw: Optional[dict[str, Any]] = None


class Recipe(BaseModel):
    title: str
    name: str = Field(..., description="产物物品名")
    price: int = 0
    kprice: int = 0
    value: Optional[int] = None
    materials: list[str] = Field(default_factory=list, description='材料数组（"物品名#数量"）')
    source: Optional[str] = Field(default=None, description="来源表（例如 烹饪/武器合成）")
    raw: Optional[dict[str, Any]] = None


class ShopItem(BaseModel):
    name: str
    raw: Optional[dict[str, Any]] = None


class KShopItem(BaseModel):
    id: str
    item: str
    type: Optional[str] = None
    price: int = 0
    raw: Optional[dict[str, Any]] = None

