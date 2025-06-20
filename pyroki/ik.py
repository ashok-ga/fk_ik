import time
import numpy as np
import pyroki as pk
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R

import pyroki_snippets as pks  # Your IK utility

def main():
    urdf = URDF.load("leader.urdf", mesh_dir="meshes")
    target_link_name = "trigger_1"
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index(target_link_name)

    # Example target pose (edit as needed)
    target_position = np.array([0.3, 0.0, 0.4])
    target_wxyz = np.array([0, 0, 1, 0])  # (w, x, y, z)

    print("Solving IK for target:")
    print("Position:", target_position)
    print("Quaternion (wxyz):", target_wxyz)

    # --- Solve IK ---
    solution = pks.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=target_position,
        target_wxyz=target_wxyz,
    )

    print("\nJoint angles (degrees):")
    for idx, ang in enumerate(np.degrees(solution)):
        print(f"  Joint {idx+1}: {ang:.2f}°")

    # --- FK: get actual EE pose for these joints ---
    fk_vec = robot.forward_kinematics(np.asarray(solution))[target_link_index]  # shape (7,)
    fk_pos = fk_vec[:3]
    fk_quat_wxyz = np.array([fk_vec[6], fk_vec[3], fk_vec[4], fk_vec[5]])

    # --- Compute error between target and actual EE pose ---
    pos_err = np.linalg.norm(target_position - fk_pos)
    r1 = R.from_quat(target_wxyz[[1,2,3,0]])  # viser: wxyz → scipy: xyzw
    r2 = R.from_quat(fk_quat_wxyz[[1,2,3,0]])
    relative_rot = r1.inv() * r2
    angle_err_deg = np.rad2deg(relative_rot.magnitude())

    print(f"\nFK achieved position: {fk_pos}")
    print(f"FK achieved quaternion (wxyz): {fk_quat_wxyz}")
    print(f"\nPosition error: {pos_err:.4f} m")
    print(f"Orientation error: {angle_err_deg:.2f} deg")

if __name__ == "__main__":
    main()
