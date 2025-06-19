import viser
from yourdfpy import URDF
from pyroki._robot_urdf_parser import RobotURDFParser
from pyroki._robot import Robot
from viser.extras import ViserUrdf
import jax.numpy as jnp
import numpy as np

def main():
    # Load your URDF file
    urdf = URDF.load("leader.urdf")
    joint_info, link_info = RobotURDFParser.parse(urdf)
    robot = Robot(joint_info, link_info, jnp.float32)

    # Set initial joint configuration (e.g., zeros)
    q = jnp.zeros(joint_info.num_actuated_joints, dtype=np.float32)

    # Set up Viser server (viewer)
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Show the robot in its current config
    urdf_vis.update_cfg(q)

    print("Open the viewer in your browser at:", server.client_url)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
