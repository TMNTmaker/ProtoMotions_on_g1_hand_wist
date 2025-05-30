from pxr import Usd

stage = Usd.Stage.Open("/root/Documents/Kit/g1_description/g1_29dof_with_hand_rev_1_0/g1_29dof_with_hand_rev_1_0.usd")
stage.Flatten().Export("/root/Documents/Kit/g1_description/g1_29dof_with_hand_flat.usd")