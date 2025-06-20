import os
import time
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def print_pose_info(label, fk_vec):
    pos = fk_vec[:3]
    quat_wxyz = np.array([fk_vec[6], fk_vec[3], fk_vec[4], fk_vec[5]])
    r = R.from_quat(quat_wxyz[[1,2,3,0]])  # wxyz to xyzw
    euler = r.as_euler('xyz', degrees=True)
    print(f"{label} position: {pos}")
    print(f"{label} orientation (wxyz): {quat_wxyz}")
    print(f"{label} orientation euler (deg): {euler}")
    return pos, quat_wxyz

def orientation_difference_deg(q1, q2):
    r1 = R.from_quat(q1[[1,2,3,0]])
    r2 = R.from_quat(q2[[1,2,3,0]])
    rel_rot = r1.inv() * r2
    return np.rad2deg(rel_rot.magnitude())

def main():
    urdf_path = "leader.urdf"
    mesh_dir = "meshes"
    urdf = URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)

    # Print all link names for EE debugging
    print("Robot link names (for EE selection):")
    print(robot.links.names)
    target_link_name = "trigger_1"
    target_link_index = robot.links.names.index(target_link_name)
    print(f"Using target_link_name: {target_link_name}, index: {target_link_index}")

    # Key poses, each one changes a different joint a lot
    q_neutral = np.zeros(robot.joints.num_actuated_joints)
    q_joint1 = np.array([1.0, 0, 0, 0, 0, 0, 0])
    q_joint2 = np.array([0, 1.0, 0, 0, 0, 0, 0])
    q_joint3 = np.array([0, 0, -1.5, 0, 0, 0, 0])
    q_joint4 = np.array([0, 0, 0, 1.0, 0, 0, 0])
    q_joint5 = np.array([0, 0, 0, 0, 1.5, 0, 0])
    q_joint6 = np.array([0, 0, 0, 0, 0, 2.0, 0])
    q_joint7 = np.array([0, 0, 0, 0, 0, 0, -2.0])
    q_combo  = np.array([0.6, -1.2, 0.7, -0.7, 0.9, 1.5, -1.1])

    key_poses = [
        q_neutral, q_joint1, q_joint2, q_joint3, q_joint4, q_joint5, q_joint6, q_joint7, q_combo
    ]

    # Print pose info for each key pose at the start
    print("\nFK poses for each key pose:")
    fk_poses = []
    for idx, q in enumerate(key_poses):
        fk_vec = robot.forward_kinematics(q)[target_link_index]
        pos, quat = print_pose_info(f"Key pose {idx}", fk_vec)
        fk_poses.append((pos, quat))

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base_link")

    duration = 2.0  # seconds per transition
    steps = int(duration / 0.02)

    idx = 0
    while True:
        q_start = key_poses[idx % len(key_poses)]
        q_end = key_poses[(idx + 1) % len(key_poses)]
        fk_start = robot.forward_kinematics(q_start)[target_link_index]
        fk_end = robot.forward_kinematics(q_end)[target_link_index]
        move_distance = np.linalg.norm(fk_start[:3] - fk_end[:3])
        quat_start = np.array([fk_start[6], fk_start[3], fk_start[4], fk_start[5]])
        quat_end   = np.array([fk_end[6],   fk_end[3],   fk_end[4],   fk_end[5]])
        orient_change = orientation_difference_deg(quat_start, quat_end)
        print(f"\n--- Transition {idx % len(key_poses)} to {(idx+1)%len(key_poses)} ---")
        print(f"EE position move distance: {move_distance:.4f} m")
        print(f"EE orientation angle change: {orient_change:.2f} deg")

        if move_distance < 1e-3:
            print("WARNING: End effector does not move much between these poses. Check joint definitions or key_poses.")
        if orient_change < 1.0:
            print("WARNING: End effector orientation does not change much between these poses.")

        for i in range(steps + 1):
            alpha = i / steps
            q = (1 - alpha) * q_start + alpha * q_end
            urdf_vis.update_cfg(np.asarray(q))
            fk_vec = robot.forward_kinematics(q)[target_link_index]
            fk_pos = fk_vec[:3]
            fk_quat_wxyz = np.array([fk_vec[6], fk_vec[3], fk_vec[4], fk_vec[5]])
            fk_frame = server.scene.add_frame("/fk_pose", position=fk_pos, wxyz=fk_quat_wxyz)
            if i == 0 or i == steps:
                print_pose_info(f"Step {i}", fk_vec)
            time.sleep(duration / steps)
        idx += 1

if __name__ == "__main__":
    main()
