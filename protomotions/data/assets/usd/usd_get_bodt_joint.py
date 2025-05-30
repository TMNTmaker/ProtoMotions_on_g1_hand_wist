from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

def list_bodies_and_joints(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    rigid_bodies = []
    joints = []

    def recurse_prim(prim):
        # Rigid body の取得
        if (prim.IsA(UsdGeom.Xform) and UsdPhysics.MassAPI(prim).GetMassAttr().HasAuthoredValue()):
            if prim.GetName() not in  rigid_bodies:
                rigid_bodies.append(prim.GetName())

        # Joint の取得（e.g., FixedJoint, RevoluteJoint, etc.）
        joint_types = [
            #UsdPhysics.Joint,
            #UsdPhysics.FixedJoint,
            #UsdPhysics.PrismaticJoint,
            UsdPhysics.RevoluteJoint,
            #PhysxSchema.PhysxJoint,
        ]
        for joint_type in joint_types:
            if joint_type(prim):
                if prim.GetName() not in  joints:
                    joints.append(prim.GetName())
                    break

        for child in prim.GetChildren():
            recurse_prim(child)

    recurse_prim(stage.GetPseudoRoot())

    return rigid_bodies, joints
def write_to_file(bodies, joints, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Unique Body Names ===\n")
        f.write(f"{bodies}\n")
        f.write("\n=== Unique Joint Names ===\n")
        f.write(f"{joints}\n")
# 例: 使用方法
if __name__ == "__main__":
    usd_file_path = "/root/Documents/Kit/g1_description/g1_29dof_with_hand_rev_1_0.usd"
    bodies, joints = list_bodies_and_joints(usd_file_path)

    print("=== Rigid Bodies ===")
    for b in bodies:
        print(b)
    print(bodies)
    print(len(bodies))

    print("\n=== Joints ===")
    for j in joints:
        print(j)
    print(joints)
    print(len(joints))
    output_file_path = "/root/Documents/Kit/g1_description/g1_29dof_with_hand_rev_1_0_bodies_and_joints.txt"
    write_to_file(bodies, joints, output_file_path)
    
    