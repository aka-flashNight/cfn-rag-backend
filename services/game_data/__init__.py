"""Game data access layer (registries + parsers + models)."""

from .models import Item
from .item_registry import ItemRegistry
from .paths import get_game_data_root, find_resources_directory
from .registry import GameDataRegistry, get_game_data_registry, init_game_data_registry

__all__ = [
    "Item",
    "ItemRegistry",
    "get_game_data_root",
    "find_resources_directory",
    "GameDataRegistry",
    "get_game_data_registry",
    "init_game_data_registry",
]

