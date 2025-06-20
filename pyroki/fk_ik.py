import time
import numpy as np
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R

from pyroki_utils import PyrokiIkBeamHelper

def main():
    urdf_path = "leader.urdf"
    mesh_dir = "meshes"
    ee_link_name = "trigger_1"

    ik_beam = PyrokiIkBeamHelper(urdf_path, mesh_dir, ee_link_name)
    robot = ik_beam.robot
    target_link_index = ik_beam.target_link_index

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, ik_beam.urdf, root_node_name="/base_link")

    ik_target = server.scene.add_transform_controls(
        "/ik_target", position=(0.3, 0.0, 0.4), wxyz=(0, 0, 1, 0)
    )
    fk_frame = server.scene.add_frame("/fk_pose", position=(0, 0, 0), wxyz=(1, 0, 0, 0))

    while True:
        solution = ik_beam.solve_ik(
            target_wxyz=np.array(ik_target.wxyz),
            target_position=np.array(ik_target.position),
        )
        urdf_vis.update_cfg(np.asarray(solution))

        fk_vec = robot.forward_kinematics(np.asarray(solution))[target_link_index]
        fk_pos = fk_vec[:3]
        fk_quat_wxyz = np.array([fk_vec[6], fk_vec[3], fk_vec[4], fk_vec[5]])

        fk_frame.position = tuple(fk_pos)
        fk_frame.wxyz = tuple(fk_quat_wxyz)

        # Print error (optional)
        r1 = R.from_quat(np.array(ik_target.wxyz)[[1,2,3,0]])
        r2 = R.from_quat(fk_quat_wxyz[[1,2,3,0]])
        pos_err = np.linalg.norm(np.array(ik_target.position) - fk_pos)
        angle_err_deg = np.rad2deg((r1.inv() * r2).magnitude())
        print(f"Position error: {pos_err:.4f} m | Orientation error: {angle_err_deg:.2f} deg")

        time.sleep(0.05)

if __name__ == "__main__":
    main()
