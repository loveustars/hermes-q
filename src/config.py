"""配置加载 —— 冻结配置只读，任何覆盖都记录来源。"""
from __future__ import annotations

import copy
import json
import os

from .data import store


def config_path(name: str = "base") -> str:
    return os.path.join(store.project_root(), "configs", f"{name}.json")


def load(name: str = "base", overrides: dict | None = None) -> dict:
    with open(config_path(name), encoding="utf-8") as f:
        cfg = json.load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
        cfg.setdefault("_overrides", []).append(overrides)
    cfg["_config_name"] = name
    return cfg


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def save(cfg: dict, name: str) -> str:
    path = config_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path


def symbols(cfg: dict, include_reserve: bool = False) -> list[str]:
    u = cfg["universe"]
    return list(u["core"]) + (list(u.get("reserve", [])) if include_reserve else [])


def guard_frozen(cfg: dict) -> None:
    """铁律：冻结池不得就地修改。"""
    if cfg["universe"]["mode"] != "FROZEN":
        raise RuntimeError("标的池不是 FROZEN 模式，样本外评估将失去可比性。")
