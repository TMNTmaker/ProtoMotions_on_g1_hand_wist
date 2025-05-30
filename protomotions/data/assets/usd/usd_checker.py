from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("./g1_hand.usd")
body_names = []
for prim in stage.Traverse():
    if UsdPhysics.RigidBodyAPI(prim).IsRigidBody():
        body_names.append(prim.GetPath().name)

print(body_names)
