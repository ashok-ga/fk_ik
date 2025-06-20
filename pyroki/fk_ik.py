import time
import numpy as np
import pyroki as pk
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R

import pyroki_snippets as pks  # Your IK utility

def main():
    urdf = URDF.load("leader.urdf", mesh_dir="meshes")
    target_link_name = "trigger_1"

    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index(target_link_name)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base_link")

    # Interactive target
    ik_target = server.scene.add_transform_controls(
        "/ik_target", position=(0.3, 0.0, 0.4), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Frame to show FK (actual EE pose)
    fk_frame = server.scene.add_frame("/fk_pose", position=(0, 0, 0), wxyz=(1, 0, 0, 0))

    while True:
        # --- Solve IK ---
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # --- Visualize ---
        urdf_vis.update_cfg(np.asarray(solution))

        # --- FK: get actual EE pose for these joints ---
        fk_vec = robot.forward_kinematics(np.asarray(solution))[target_link_index]  # shape (7,)
        fk_pos = fk_vec[:3]
        fk_quat_wxyz = np.array([fk_vec[6], fk_vec[3], fk_vec[4], fk_vec[5]])

        fk_frame = server.scene.add_frame(
            "/fk_pose", position=fk_pos, wxyz=fk_quat_wxyz
        )

        # --- Debug: Compute error between target and actual EE pose ---
        target_pos = np.array(ik_target.position)
        target_quat = np.array(ik_target.wxyz)  # (w, x, y, z)
        # Position error
        pos_err = np.linalg.norm(target_pos - fk_pos)
        # Orientation error (angle in degrees)
        r1 = R.from_quat(target_quat[[1,2,3,0]])  # viser: wxyz → scipy: xyzw
        r2 = R.from_quat(fk_quat_wxyz[[1,2,3,0]])
        relative_rot = r1.inv() * r2
        angle_err_deg = np.rad2deg(relative_rot.magnitude())

        print(f"Position error: {pos_err:.4f} m | Orientation error: {angle_err_deg:.2f} deg")

        time.sleep(0.05)

if __name__ == "__main__":
    main()
