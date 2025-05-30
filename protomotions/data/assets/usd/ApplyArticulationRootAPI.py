"""Isaac Sim の Script Editor 上で実行し、USD の root prim を articulation として設定する。

USD を IsaacLab に読み込ませるには ArticulationRootAPI が必要だが、URDF Importer が
これを付けない場合があるため手動で適用する。
"""

import sys

from omni.isaac.core.utils.stage import get_current_stage
from pxr import Sdf, UsdPhysics

DEFAULT_PRIM_PATH = "/g1_29dof_with_hand_rev_1_0"


def apply_articulation_root(prim_path: str, fixed_base: bool = False) -> None:
    stage = get_current_stage()
    robot_prim = stage.GetPrimAtPath(prim_path)

    if not robot_prim.IsValid():
        raise RuntimeError(f"Prim not found or not valid: {prim_path}")

    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    attr = robot_prim.CreateAttribute("physics:fixedBase", Sdf.ValueTypeNames.Bool)
    attr.Set(fixed_base)

    art_api = UsdPhysics.ArticulationRootAPI.Get(stage, robot_prim.GetPath())
    print(f"ArticulationRootAPI bound: {bool(art_api)}")
    print(f"physics:fixedBase = {robot_prim.GetAttribute('physics:fixedBase').Get()}")


if __name__ == "__main__":
    apply_articulation_root(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRIM_PATH)
