import time
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import meshcat.transformations as tf

# ---- Setup paths ----
urdf_path = "robot.urdf"
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

# ---- Function to draw EE axes ----
def draw_ee_axes(viz, ee_pose, name="ee_axes", length=0.07, radius=0.003):
    # X = red, Y = green, Z = blue
    axes = [
        {"axis": [1,0,0], "color": 0xFF0000, "name": "x"},
        {"axis": [0,1,0], "color": 0x00FF00, "name": "y"},
        {"axis": [0,0,1], "color": 0x0000FF, "name": "z"},
    ]
    base_tf = np.eye(4)
    base_tf[:3,:3] = ee_pose.rotation
    base_tf[:3,3] = ee_pose.translation
    for a in axes:
        cyl = mg.Cylinder(length, radius)
        mat = mg.MeshLambertMaterial(color=a["color"])
        tfm = np.eye(4)
        tfm[:3,3] = [length/2, 0, 0]
        axis_rot = {
            "x": np.eye(4),
            "y": tf.rotation_matrix(np.pi/2, [0,0,1]),
            "z": tf.rotation_matrix(-np.pi/2, [0,1,0]),
        }[a["name"]]
        tf_cyl = base_tf @ axis_rot @ tfm
        viz.viewer[f"{name}/{a['name']}"].set_object(cyl, mat)
        viz.viewer[f"{name}/{a['name']}"].set_transform(tf_cyl)

# ---- Main loop: animate random changes to only one joint ----
ee_id = model.getFrameId(ee_frame)
print("\nAnimating: only one joint moves at each step. (Ctrl+C to quit)")

from scipy.spatial.transform import Rotation as R

while True:
    q = np.random.uniform(lower, upper)
    try:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        robot.display(q)
        ee_pose = data.oMf[ee_id]
        viz.viewer["ee_axes"].delete()
        draw_ee_axes(viz, ee_pose)

        # --- Print pose in required format, with labels ---
        pos_cm = ee_pose.translation * 100
        rpy_rad = R.from_matrix(ee_pose.rotation).as_euler('xyz', degrees=False)
        rpy_deg = np.degrees(rpy_rad)

        print("Joint vector (q):", np.round(q, 3))
        print(f"End-effector position X (cm): {pos_cm[0]:.2f}")
        print(f"End-effector position Y (cm): {pos_cm[1]:.2f}")
        print(f"End-effector position Z (cm): {pos_cm[2]:.2f}")
        print(f"End-effector orientation roll  (deg): {rpy_deg[0]:.1f}")
        print(f"End-effector orientation pitch (deg): {rpy_deg[1]:.1f}")
        print(f"End-effector orientation yaw   (deg): {rpy_deg[2]:.1f}")
        print("-" * 40)

    except Exception as e:
        print("Failed to update configuration:", e)
        pass
    time.sleep(3.0)

