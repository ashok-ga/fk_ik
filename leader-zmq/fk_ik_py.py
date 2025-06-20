"""
Simplest Inverse Kinematics Example using PyRoki for 'leader.urdf'
"""

import time
import numpy as np
import pyroki as pk
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf

import pyroki_snippets as pks  # assumes your own IK utility is defined here


def main():
    # Load the URDF model
    urdf = URDF.load("leader.urdf")
    target_link_name = "trigger_1"  # Replace with your actual end-effector frame

    # Create PyRoki robot from URDF
    robot = pk.Robot.from_urdf(urdf)

    # Start viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base_link")  # Adjust root if needed

    # Add transform control for IK target
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.4), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update robot visualization
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
