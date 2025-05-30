"""USD からボディ名と revolute joint 名を抽出する。

G1 の robot 設定（`body_names` / `dof_names`）を書き起こす際に、USD 側の
実際の名前と順序を確認するために使う。
"""

import sys

from pxr import Usd, UsdGeom, UsdPhysics

JOINT_TYPES = [UsdPhysics.RevoluteJoint]


def list_bodies_and_joints(usd_path: str) -> tuple[list[str], list[str]]:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    rigid_bodies: list[str] = []
    joints: list[str] = []

    def recurse_prim(prim):
        has_mass = UsdPhysics.MassAPI(prim).GetMassAttr().HasAuthoredValue()
        if prim.IsA(UsdGeom.Xform) and has_mass:
            if prim.GetName() not in rigid_bodies:
                rigid_bodies.append(prim.GetName())

        for joint_type in JOINT_TYPES:
            if joint_type(prim):
                if prim.GetName() not in joints:
                    joints.append(prim.GetName())
                break

        for child in prim.GetChildren():
            recurse_prim(child)

    recurse_prim(stage.GetPseudoRoot())

    return rigid_bodies, joints


def write_to_file(bodies: list[str], joints: list[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Unique Body Names ===\n")
        f.write(f"{bodies}\n")
        f.write("\n=== Unique Joint Names ===\n")
        f.write(f"{joints}\n")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: usd_get_bodt_joint.py <robot.usd> [output.txt]")

    bodies, joints = list_bodies_and_joints(sys.argv[1])
    print(f"bodies ({len(bodies)}): {bodies}")
    print(f"joints ({len(joints)}): {joints}")

    if len(sys.argv) == 3:
        write_to_file(bodies, joints, sys.argv[2])
