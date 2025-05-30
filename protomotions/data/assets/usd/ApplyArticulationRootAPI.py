from omni.isaac.core.utils.stage import get_current_stage
import omni.usd
from pxr import UsdPhysics

stage = get_current_stage()
robot_prim = stage.GetPrimAtPath("/g1_29dof_with_hand_rev_1_0")
UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
art_api = UsdPhysics.ArticulationRootAPI(robot_prim)
print(art_api.GetPrim().GetPropertyNames())
#art_api.CreateFixedBaseAttr(False) 


# まず Prim が有効か確認
if not robot_prim.IsValid():
    print("[ERROR] Prim not found or not valid.")
else:
    # ArticulationRootAPI のインスタンス化
    art_api = UsdPhysics.ArticulationRootAPI.Get(stage, robot_prim.GetPath())

    # バインドされているかどうかは `.Get()` が None かで判断する
    if art_api:
        print("✅ ArticulationRootAPI is bound")
        if art_api.GetFixedBaseAttr().IsAuthored():
            print("Fixed base:", art_api.GetFixedBaseAttr().Get())
        else:
            print("Fixed base attr not authored (default False)")
    else:
        print("❌ ArticulationRootAPI is NOT bound")






from omni.isaac.core.utils.stage import get_current_stage
import omni.usd
from pxr import UsdPhysics, Sdf

stage = get_current_stage()
robot_prim = stage.GetPrimAtPath("/g1_29dof_with_hand_rev_1_0")

if not robot_prim.IsValid():
    print("[ERROR] Invalid prim")
else:
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    print("✅ ArticulationRootAPI is bound")

    # fixedBase 属性の作成と設定
    attr = robot_prim.CreateAttribute("physics:fixedBase", Sdf.ValueTypeNames.Bool)
    attr.Set(False)

    # 確認
    print("✅ physics:fixedBase =", robot_prim.GetAttribute("physics:fixedBase").Get())
