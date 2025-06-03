
import os
from pathlib import Path
from typing import Optional

import ipdb
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot

from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
#from poselib.visualization.common import plot_skeleton_motion_interactive

import time

from datetime import timedelta

from bvh import Bvh


def euler_to_quaternion_continuous(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """
    多関節オイラー角 (T, J, 3) をクォータニオン (T, J, 4) に変換し、時間軸方向の符号連続性を保つ。

    Args:
        euler: np.ndarray of shape (T, J, 3) - オイラー角 (ラジアン)
        order: str - 回転順 (デフォルトは "XYZ")

    Returns:
        quat: np.ndarray of shape (T, J, 4) - クォータニオン (x, y, z, w)
    """
    T, J, _ = euler.shape
    quat = np.zeros((T, J, 4))

    for j in range(J):
        r = sRot.from_euler(order, euler[:, j, :], degrees=False)
        q = r.as_quat()  # shape: (T, 4) - (x, y, z, w)

        # ノルム正規化（念のため）
        q = q / np.linalg.norm(q, axis=1, keepdims=True)

        # 時間方向の連続性を確保
        for t in range(1, T):
            if np.dot(q[t], q[t - 1]) < 0:
                q[t] *= -1

        quat[:, j, :] = q

    return quat

def make_quaternion_signs_consistent(qi):
    """
    クォータニオン配列 q (T, J, 4) に対して時間方向に符号を揃える。
    同じ回転を表す q と -q のジャンプを防ぐため。
    """
    q = qi.copy()  # 破壊的変更を避ける場合
    T, J, _ = q.shape

    for t in range(1, T):
        dot = np.sum(q[t] * q[t - 1], axis=-1)  # shape: (J,)
        flip_mask = dot < 0  # shape: (J,)
        q[t, flip_mask] *= -1.0

    return q


def main(
    bvh_root_dir: Path = None,
    robot_type: str = None,
    force_remake: bool = True,
    generate_flipped: bool = False,
    not_upright_start: bool = True,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Path = None,
    force_retarget: bool = True,
    output_dir: Path = None,
    ):
    assert bvh_root_dir is not None,"require bvh_root_dir!!"
    if output_dir is None:
        output_dir = bvh_root_dir
    assert (robot_type in ["h1", "g1","g1_hand","g1_hand_wrist"]
            ),"Choose from h1/g1/g1_hand/g1_hand_wrist !!"
    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted"
    upright_start = not not_upright_start

    g1_hand_wrist_body_name=['pelvis', 'head',
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 
    'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 
    'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
    'waist_yaw_link', 'waist_roll_link', 'torso_link', 
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
    'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
    'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_index_0_link', 'left_hand_index_1_link', 
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 
    'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 
    'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_index_0_link', 'right_hand_index_1_link']
    bvh_body_name =[
        'Hips', 
        'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 
        'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 
        'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 
        'LeftHand', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb', 
        'RightShoulder', 'RightArm', 'RightForeArm', 
        'RightHand', 'RightFingerBase', 'RightHandIndex1', 'RThumb',
        "None"
        ]
    bvh2mu_mapping = {
        'pelvis':"Hips",
        'head': "Head",
        'left_hip_pitch_link': "LeftUpLeg", 
        'left_hip_roll_link': "LeftUpLeg",
        'left_hip_yaw_link': "LeftUpLeg", 
        'left_knee_link': "LeftLeg", 
        'left_ankle_pitch_link':"LeftFoot", 
        'left_ankle_roll_link':"LeftFoot", 
        'right_hip_pitch_link':"RightUpLeg", 
        'right_hip_roll_link':"RightUpLeg", 
        'right_hip_yaw_link':"RightUpLeg", 
        'right_knee_link':"RightLeg", 
        'right_ankle_pitch_link':"RightFoot", 
        'right_ankle_roll_link':"RightFoot", 
        'waist_yaw_link':"Spine", 
        'waist_roll_link':"Spine", 
        'torso_link':"Spine", 
        'left_shoulder_pitch_link':"LeftArm", 
        'left_shoulder_roll_link':"LeftArm", 
        'left_shoulder_yaw_link':"LeftArm", 
        'left_elbow_link':"LeftForeArm", 
        'left_wrist_roll_link':"LeftHand", 
        'left_wrist_pitch_link':"LeftHand", 
        'left_wrist_yaw_link':"LeftFingerBase", 
        'left_hand_index_0_link':"None", 
        'left_hand_index_1_link':"LeftHandIndex1", 
        'left_hand_middle_0_link':"None", 
        'left_hand_middle_1_link':"None", 
        'left_hand_thumb_0_link':"None", 
        'left_hand_thumb_1_link':"None", 
        'left_hand_thumb_2_link':"None", 
        'right_shoulder_pitch_link':"RightArm", 
        'right_shoulder_roll_link':"RightArm", 
        'right_shoulder_yaw_link':"RightArm", 
        'right_elbow_link':"RightForeArm", 
        'right_wrist_roll_link':"RightHand", 
        'right_wrist_pitch_link':"RightHand", 
        'right_wrist_yaw_link':"RightFingerBase", 
        'right_hand_index_0_link':"None", 
        'right_hand_index_1_link':"None", 
        'right_hand_middle_0_link':"None", 
        'right_hand_middle_1_link':"None", 
        'right_hand_thumb_0_link':"None", 
        'right_hand_thumb_1_link':"None", 
        'right_hand_thumb_2_link':"None"
    }
    left_to_right_index = []
    for idx, entry in enumerate(g1_hand_wrist_body_name):
        # swap text "R_" and "L_"
        if entry.startswith("right_"):
            left_to_right_index.append(g1_hand_wrist_body_name.index("left_" + entry[6:]))
        elif entry.startswith("left_"):
            left_to_right_index.append(g1_hand_wrist_body_name.index("right_" + entry[5:]))
        else:
            left_to_right_index.append(idx)
    
    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(bvh_root_dir) if f.is_dir()
    ]
    print(folder_names)

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    else:
        assert True, "Please provide humanoid_mjcf_path"
    #ipdb.set_trace()
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0

    for folder_name in folder_names:
        data_dir = bvh_root_dir / folder_name
        print(data_dir)
        save_dir = output_dir / f"{folder_name}-{append_name}"
        all_files_in_folder = [
            f
            for f in Path(data_dir).glob("*.bvh")
        ]
        files_to_process = all_files_in_folder
        print(
            f"Processing {len(files_to_process)}/{len(all_files_in_folder)} files in {folder_name}"
        )
        total_files_to_process += len(files_to_process)
        total_files += len(all_files_in_folder)

    print(f"Total files to process: {total_files_to_process}/{total_files}")
    #ipdb.set_trace()
    for folder_name in folder_names:
        data_dir = bvh_root_dir / folder_name
        save_dir = output_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(save_dir, exist_ok=True)
        print(data_dir)
        files = [
            f
            for f in Path(data_dir).glob("*.bvh")
        ]
        print(f"Processing {len(files)} files")

        files.sort()     

        for filename in tqdm(files):
            #try:
                relative_path_dir = filename.relative_to(data_dir).parent
                outpath = (
                    save_dir
                    / relative_path_dir
                    / filename.name.replace(".bvh", ".npy")
                )

                # Check if the output file already exists
                if not force_remake and outpath.exists():
                    # print(f"Skipping {filename} as it already exists.")
                    continue

                # Create the output directory if it doesn't exist
                os.makedirs(save_dir / relative_path_dir, exist_ok=True)

                print(f"Processing {filename}")
                if filename.suffix == ".bvh":
                    with open(filename, "r") as f:
                        mocap = Bvh(f.read())
                    # 角度データの格納（deg）
                    # 対象関節（rotationチャネルがあるジョイントだけ）
                    bvh_joint_names = []
                    for joint in mocap.get_joints():
                        name = joint.name
                        channels = mocap.joint_channels(name)
                        if any("rotation" in ch.lower() for ch in channels):
                            bvh_joint_names.append(name)
                    
                    num_frames = mocap.nframes
                    qpos = []
                    #ipdb.set_trace()
                    for frame_idx in range(num_frames):
                        frame_data = []
                        for joint in bvh_joint_names:
                            rot = mocap.frame_joint_channels(frame_idx, joint, 
                                ['Yrotation', 
                                 'Xrotation',
                                 'Zrotation'])
                            rotvec = np.radians(rot)
                            #rotvec = sRot.from_euler("XYZ", rot, degrees=True).as_rotvec()
                            frame_data.extend(rotvec)
                        qpos.append(frame_data)
                    bvh_pose = np.array(qpos)  # shape = (T, J*3)
                    print(f"bvh_pose shape: {bvh_pose.shape}")

                    #ipdb.set_trace()
                    # Rootの位置をa抽出
                    root_pos = []
                    for frame_idx in range(num_frames):
                        pos = mocap.frame_joint_channels(frame_idx, "Hips", 
                                ['Yposition', 
                                 'Xposition', 
                                 'Zposition'])
                        root_pos.append(pos)
                    bvh_trans = np.array(root_pos)* 0.0254  # shape (T, 3)                    
                    print(f"bvh_trans shape: {bvh_trans.shape}")

                    mocap_fr = int(1.0 / mocap.frame_time)
                    #ipdb.set_trace()
                else:
                    print(f"Skipping {filename} as it is not a valid file")
                    continue

                motion_data = {
                            "pose_aa": bvh_pose,
                            "trans": bvh_trans,
                }


                bvh_2_g1 = [ bvh_body_name.index(bvh2mu_mapping[q]) for q in g1_hand_wrist_body_name ]
                
                batch_size = mocap.nframes


                pose_aa = np.concatenate([motion_data["pose_aa"],np.zeros((batch_size, 3))],axis=1)
                

                pose_aa_mj = pose_aa.reshape(batch_size, 32, 3)[:, bvh_2_g1]
                
                

                import matplotlib.pyplot as plt
                x=pose_aa_mj[:,0]
                # フレーム番号
                frames = np.arange(x.shape[0])

                # 描画
                plt.figure(figsize=(10, 6))
                plt.plot(frames, x[:, 0], label='X')
                plt.plot(frames, x[:, 1], label='Y')
                plt.plot(frames, x[:, 2], label='Z')
                plt.xlabel('Frame')
                plt.ylabel('Angle (radian)')
                plt.title('Joint Rotation over Time')
                plt.legend()
                plt.grid(True)

                # 保存
                plt.savefig("rotation_angles.png", dpi=300)
                plt.close()

                
                
                #import ipdb
                #ipdb.set_trace()
                pose_quat = euler_to_quaternion_continuous(pose_aa_mj)
                #pose_quat[:, 0, :] = np.array([0, 0, 0, 1])
                #none_indices = [i for i, name in enumerate(g1_hand_wrist_body_name) if bvh2mu_mapping[name] == "None"]
                #for idx in none_indices:
                #    pose_quat[:, idx, :] = np.array([0, 0, 0, 1])  # 全frame、該当関節を単位クォータニオンで埋める

                x = pose_quat[:,0,:]
                plt.figure(figsize=(10, 6))
                plt.plot(frames, x[:, 0], label='x')
                plt.plot(frames, x[:, 1], label='y')
                plt.plot(frames, x[:, 2], label='z')
                plt.plot(frames, x[:, 3], label='w')
                plt.xlabel('Frame')
                plt.ylabel('Quaternion Component Value')
                plt.title('Quaternion Components over Time')
                plt.legend()
                plt.grid(True)

                # 保存
                plt.savefig("quaternion_components.png", dpi=300)
                plt.close()

                r = sRot.from_quat(x)  # x: shape=(248, 4)

                # 回転ベクトルに変換 (axis * angle 形式, shape=(248, 3))
                rotvec = r.as_rotvec()

                # 可視化
                frames = np.arange(rotvec.shape[0])
                plt.figure(figsize=(10, 6))
                plt.plot(frames, rotvec[:, 0], label='rotvec-x')
                plt.plot(frames, rotvec[:, 1], label='rotvec-y')
                plt.plot(frames, rotvec[:, 2], label='rotvec-z')
                plt.xlabel('Frame')
                plt.ylabel('Rotation Vector Value (rad)')
                plt.title('Rotation Vector over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig("rotation_vector.png", dpi=300)
                plt.close()




                euler = r.as_euler('XYZ', degrees=False)  # shape=(248, 3)
                #euler=np.unwrap(euler, axis=0)
                # 可視化
                plt.figure(figsize=(10, 6))
                plt.plot(frames, euler[:, 0], label='Euler X')
                plt.plot(frames, euler[:, 1], label='Euler Y')
                plt.plot(frames, euler[:, 2], label='Euler Z')
                plt.xlabel('Frame')
                plt.ylabel('Euler Angle (rad)')
                plt.title('Euler Angles over Time (XYZ)')
                plt.legend()
                plt.grid(True)
                plt.savefig("euler_angles.png", dpi=300)
                plt.close()


                root_quat = pose_quat[:, 0].copy()
                pose_quat[:, 0, :] = np.array([0, 0, 0, 1])  # Set root quaternion to identity
                root_trans_offset = (
                    torch.from_numpy(motion_data["trans"])
                    + skeleton_tree.local_translation[0]
                )


                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True,
                )
                #ipdb.set_trace()
                if generate_flipped:
                    formats = ["regular", "flipped"]
                else:
                    formats = ["regular"]

                for format in formats:
                    if upright_start:
                        B = pose_aa.shape[0]
                        pose_quat_global = (
                            (
                                sRot.from_quat(
                                    sk_state.global_rotation.reshape(-1, 4).numpy()
                                )
                                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                            )
                            .as_quat()
                            .reshape(B, -1, 4)
                        )
                    else:
                        pose_quat_global = sk_state.global_rotation.numpy()
                
                    trans = root_trans_offset.clone()
                    if format == "flipped":
                        pose_quat_global = pose_quat_global[:, left_to_right_index]
                        pose_quat_global[..., 0] *= -1
                        pose_quat_global[..., 2] *= -1
                        trans[..., 1] *= -1
                    """
                    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,
                        torch.from_numpy(pose_quat_global),
                        trans,
                        is_local=True,
                    )
                    
                    new_sk_motion = SkeletonMotion.from_skeleton_state(
                        new_sk_state, fps=mocap_fr
                    )
                    """
                    if force_retarget:
                        from data.scripts.retargeting.mink_retarget import (
                            retarget_motion,
                        )

                        print("Force retargeting motion using mink retargeter...")
                        # Convert to 30 fps to speedup Mink retargeting
                        cnv_fps=120
                        skip = int(mocap_fr // cnv_fps)
                        #import ipdb
                        #ipdb.set_trace()



                        pose_quat_local = pose_quat
                        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                            skeleton_tree,
                            torch.from_numpy(pose_quat_local[::skip]),
                            trans[::skip],
                            is_local=False,
                        )
                        
                        new_sk_motion = SkeletonMotion.from_skeleton_state(
                            new_sk_state, fps=cnv_fps
                        )
                        



                        new_sk_motion = retarget_motion(
                            motion=new_sk_motion, 
                            robot_type=robot_type, 
                            euler_angles_root=pose_aa_mj[:,0][::skip],
                            pos_root=skeleton_tree.local_translation[0],
                            render=True,
                            smplx_mujoco_joint_names=g1_hand_wrist_body_name
                        )
                        
                        #dict_keys(['global_translation', 'global_rotation_mat', 'global_rotation', 'global_velocity', 'global_angular_velocity', 'local_rotation', 'global_root_velocity', 'global_root_angular_velocity', 'dof_pos', 'dof_vels', 'fps'])

                    if format == "flipped":
                        outpath = outpath.with_name(
                            outpath.stem + "_flipped" + outpath.suffix
                        )
                    print(f"Saving to {outpath}")
                    if robot_type in ["h1", "g1","g1_hand_wrist","g1_hand"]:

                        torch.save(new_sk_motion, str(outpath))
                    else:
                        new_sk_motion.to_file(str(outpath))

                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files_to_process - processed_files
                    estimated_time_remaining = avg_time_per_file * remaining_files

                    print(
                        f"\nProgress: {processed_files}/{total_files_to_process} files"
                    )
                    print(
                        f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}"
                    )
                    print(
                        f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
                    )
                    print(
                        f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
                    )
            #except Exception as e:
            #    print(f"Error processing {filename}")
            #    print(f"Error: {e}")
            #    print(f"Line: {e.__traceback__.tb_lineno}")
            #    continue

if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
