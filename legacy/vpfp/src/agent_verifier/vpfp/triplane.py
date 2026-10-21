from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TriplaneBatch:
    p_xy: Any
    p_xt: Any
    p_yt: Any

    def as_tuple(self):
        return self.p_xy, self.p_xt, self.p_yt


def plane_tuple(planes):
    if isinstance(planes, TriplaneBatch):
        return planes.p_xy, planes.p_xt, planes.p_yt
    return planes


def plane_shapes(planes: TriplaneBatch) -> dict[str, list[int]]:
    return {
        "p_xy": [int(v) for v in planes.p_xy.shape],
        "p_xt": [int(v) for v in planes.p_xt.shape],
        "p_yt": [int(v) for v in planes.p_yt.shape],
    }
