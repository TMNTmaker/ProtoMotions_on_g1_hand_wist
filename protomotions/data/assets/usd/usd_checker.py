"""USD 内の rigid body 名を列挙する。

robot 設定の `body_names` を USD の実体と突き合わせるために使う。
"""

import sys

from pxr import Usd, UsdPhysics


def list_rigid_bodies(usd_path: str) -> list[str]:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    return [
        prim.GetPath().name
        for prim in stage.Traverse()
        if UsdPhysics.RigidBodyAPI(prim).IsRigidBody()
    ]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: usd_checker.py <robot.usd>")
    print(list_rigid_bodies(sys.argv[1]))
