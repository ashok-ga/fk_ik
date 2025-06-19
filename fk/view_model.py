import time
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

# ---- Setup paths ----
urdf_path = "leader.urdf"
mesh_dir = "meshes"
ee_frame = "trigger_1"

# ---- Load robot ----
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model = robot.model
data = model.createData()

# ---- Visualizer ----
viz = MeshcatVisualizer()
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel("pinocchio")

# ---- Show neutral pose first ----
q_neutral = pin.neutral(model)
robot.display(q_neutral)
print("Displayed robot at home (neutral) configuration.")
print(f"Meshcat URL: {viz.viewer.url()}")  # Prints the link to open in your browser

# ---- Print joint info ----
lower = model.lowerPositionLimit
upper = model.upperPositionLimit
print("Joint lower limits:", lower)
print("Joint upper limits:", upper)
print(f"Number of joints: {model.nq}")

# ---- Main loop: animate random changes to only one joint ----
ee_id = model.getFrameId(ee_frame)
print("\nAnimating: only one joint moves at each step. (Ctrl+C to quit)")

while True:
    # Start from neutral
    q = pin.neutral(model).copy()
    # Randomly pick a joint index to move
    joint_idx = np.random.randint(0, model.nq)
    # Assign it a random value within its limit
    q[joint_idx] = np.random.uniform(lower[joint_idx], upper[joint_idx])
    try:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        robot.display(q)
        ee_pose = data.oMf[ee_id]
        print(f"Moved joint {joint_idx}: value {q[joint_idx]:.3f}")
        print("q:", q)
        print("End-effector position:", ee_pose.translation)
        print("End-effector rotation:\n", ee_pose.rotation)
    except Exception as e:
        print("Failed to update configuration:", e)
        pass
    time.sleep(1.0)  # 1 Hz update
