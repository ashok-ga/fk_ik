import time
import zmq
import json
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

# ---- Setup paths ----
urdf_path = "leader.urdf"
mesh_dir = "meshes"
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
print("Displayed robot at home (neutral) configuration.")
print(f"Meshcat URL: {viz.viewer.url()}")

# ---- Print joint info ----
lower = model.lowerPositionLimit
upper = model.upperPositionLimit
print("Joint lower limits:", lower)
print("Joint upper limits:", upper)
print(f"Number of joints: {model.nq}")

# ---- ZMQ subscriber setup ----
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("\nZMQ subscriber running. Will print joint angle differences (in radians) from initial value.")

reference_ticks = None
count = 0
last_time = time.time()

while True:
    msg = socket.recv()
    count += 1
    now = time.time()
    try:
        msg_str = msg.decode('utf-8')
        pos_dict = json.loads(msg_str)
        joint_ids = [f"id{i}" for i in range(N_DYNAMIXEL)]
        ticks = np.array([pos_dict[k] for k in joint_ids], dtype=np.float64)
        if reference_ticks is None:
            reference_ticks = ticks.copy()
            print("Set initial ticks reference:", reference_ticks)
        ticks_delta = ticks - reference_ticks
        angles_rad = (ticks_delta / DYNAMIXEL_RESOLUTION) * TWO_PI
    except Exception as e:
        print("Failed to parse ZMQ message:", e)
        continue

    if now - last_time >= 1.0:
        print(f"\n[{time.strftime('%H:%M:%S')}] Angle differences from initial (rad): {angles_rad}")
        print("Raw encoder values:", ticks)
        count = 0
        last_time = now
