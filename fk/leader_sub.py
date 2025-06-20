import time
import zmq
import json
import numpy as np
import threading
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import meshcat.transformations as tf
from scipy.spatial.transform import Rotation as R

# ---- Robot params ----
urdf_path = "robot.urdf"
mesh_dir = "meshes"
ee_frame = "trigger_1"
N_DYNAMIXEL = 7
DYNAMIXEL_RESOLUTION = 4096
TWO_PI = 2 * np.pi

# ---- Load robot ----
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model = robot.model

# ---- Visualizer ----
viz = MeshcatVisualizer()
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel("pinocchio")
q_neutral = pin.neutral(model)
robot.display(q_neutral)

# ---- Joint name mapping ----
dynamixel_joint_names = [f"joint_{i+1}" for i in range(N_DYNAMIXEL)]
pinocchio_joint_names = [str(x) for x in model.names[1:model.nq+1]]
remap = [pinocchio_joint_names.index(dname) for dname in dynamixel_joint_names]

def draw_ee_axes(viz, ee_pose, name="ee_axes", length=0.07, radius=0.003):
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

# ---- Shared data for threads ----
angles_rad = np.zeros(N_DYNAMIXEL)
angles_lock = threading.Lock()
reference_ticks = None

# ---- ZMQ Reader Thread ----
def zmq_reader():
    global reference_ticks
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    while True:
        msg = socket.recv()
        try:
            msg_str = msg.decode('utf-8')
            pos_dict = json.loads(msg_str)
            joint_ids = [f"id{i}" for i in range(N_DYNAMIXEL)]
            ticks = np.array([pos_dict[k] for k in joint_ids], dtype=np.float64)
            if reference_ticks is None:
                reference_ticks = ticks.copy()
            ticks_delta = ticks - reference_ticks
            new_angles = (ticks_delta / DYNAMIXEL_RESOLUTION) * TWO_PI
            with angles_lock:
                angles_rad[:] = new_angles
        except Exception as e:
            print("Failed to parse ZMQ message:", e)

# ---- FK/Visualizer/EE Axes Thread ----
def fk_visualizer():
    ee_id = model.getFrameId(ee_frame)
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    while True:
        with angles_lock:
            current_angles = angles_rad.copy()
        q = pin.neutral(model)
        for i, idx in enumerate(remap):
            q[idx] = current_angles[i]
        # OVERRIDE joint 3 (index 2) with a negative value
        q[2] = -abs(q[2])
        q = np.clip(q, lower, upper)
        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        robot.display(q)
        ee_pose = data.oMf[ee_id]
        try:
            viz.viewer["ee_axes"].delete()
        except Exception:
            pass
        draw_ee_axes(viz, ee_pose)
        time.sleep(0.001)

# ---- Start threads ----
threading.Thread(target=zmq_reader, daemon=True).start()
threading.Thread(target=fk_visualizer, daemon=True).start()

# ---- Main thread: keep alive ----
while True:
    time.sleep(1)
