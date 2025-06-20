import time
import threading
import numpy as np
import csv
from datetime import datetime
from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncRead,
    GroupSyncWrite,
)
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# === Paths and Model ===
URDF_PATH = "robot.urdf"
MESH_DIR = ["meshes"]

# Load robot model (Pinocchio)
robot = RobotWrapper.BuildFromURDF(URDF_PATH, MESH_DIR)
model = robot.model
data = robot.data

# === Dynamixel Constants ===
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 4000000
PROTOCOL_VERSION = 2.0

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11

MOTOR_IDS = list(range(7))
DXL_RESOLUTION = 4096

# Torque constants and current LSB per joint (adjust for your model)
TORQUE_CONSTANTS = np.array([1.1] * 7)  # [Nm/A] for each joint
DXL_CURRENT_UNIT = 2.69e-3  # [A/LSB] for XL430, XM430. Change if needed!

# === Control Parameters ===
STIFFNESS = np.diag([6, 40, 30, 9.0, 0.8, 0.8, 0.0])
DAMPING = np.diag([2.0, 2.0, 1.2, 1.0, 0.8, 0.8, 0.1])
FEEDFORWARD_GAIN = np.diag([6, -10, 10, 4, 2, 2, 0.1])
VEL_THRESHOLD = 0.01
STILL_CYCLES = 30
CONTROL_FREQ = 50
DT = 1.0 / CONTROL_FREQ

# === Initialize SDK ===
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
assert portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

groupRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
groupWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

for dxl_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, 0)  # current mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
    groupRead.addParam(dxl_id)

def raw_to_rad(raw, center):
    # Convert Dynamixel raw position to radians (accounting for captured zero offset)
    return (raw - center) * (2 * np.pi / DXL_RESOLUTION)

def compute_gravity_torque(q):
    # Pinocchio expects shape (7,), radians
    pin.computeGeneralizedGravity(model, data, q)
    return data.g.copy()

def send_currents(currents):
    groupWrite.clearParam()
    for i, dxl_id in enumerate(MOTOR_IDS):
        # Convert Nm to Amps using torque constant, then Amps to LSB using DXL_CURRENT_UNIT
        amps = currents[i] / TORQUE_CONSTANTS[i]
        current_lsb = int(np.clip(amps / DXL_CURRENT_UNIT, -2047, 2047))
        param = [current_lsb & 0xFF, (current_lsb >> 8) & 0xFF]
        groupWrite.addParam(dxl_id, param)
    groupWrite.txPacket()

raw_zero = {}

def zero_pose_capture():
    global raw_zero
    print("Waiting for joint data to capture zero pose...")
    while True:
        groupRead.txRxPacket()
        if all(groupRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION) for dxl_id in MOTOR_IDS):
            raw_zero = {
                dxl_id: groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                for dxl_id in MOTOR_IDS
            }
            print("Zero pose captured.")
            break
        time.sleep(0.1)

class ControlLoop:
    def __init__(self):
        self.q_target = np.zeros(7)
        self.q_prev = np.zeros(7)
        self.hold_enabled = False

    def hold_position(self, q):
        self.q_target = q.copy()
        print("🔵 HOLD: Holding compliant pose at current position.")

    def run(self):
        still_counter = 0
        debug_counter = 0
        log_file = open("impedance_log.csv", "w", newline="")
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(
            ["time"] +
            [f"q{i}" for i in range(7)] +
            [f"q_err{i}" for i in range(7)] +
            [f"dq{i}" for i in range(7)] +
            [f"tau{i}" for i in range(7)]
        )

        while True:
            start_time = time.time()
            groupRead.txRxPacket()
            q_raw = [groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION) for dxl_id in MOTOR_IDS]
            q = np.array([raw_to_rad(q_raw[i], raw_zero[MOTOR_IDS[i]]) for i in range(7)])
            dq = (q - self.q_prev) / DT
            self.q_prev = q.copy()

            if not self.hold_enabled:
                if np.linalg.norm(dq) < VEL_THRESHOLD:
                    still_counter += 1
                else:
                    still_counter = 0
                if still_counter >= STILL_CYCLES:
                    self.q_target = q.copy()
                    still_counter = 0
                    print("🟢 Updated floating hold pose (auto)")

            if self.hold_enabled:
                self.hold_position(q)
                self.hold_enabled = False

            q_error = np.clip(q - self.q_target, -0.5, 0.5)
            tau_g = compute_gravity_torque(q)
            tau_imp = -STIFFNESS @ q_error
            tau_damp = -DAMPING @ dq
            tau_ff = FEEDFORWARD_GAIN @ dq
            tau = tau_g + tau_imp + tau_damp + tau_ff

            send_currents(tau)

            now = time.time()
            csv_writer.writerow([now] + list(q) + list(q_error) + list(dq) + list(tau))

            debug_counter += 1
            if debug_counter % 25 == 0:
                print(f"DEBUG q: {q.round(3)} | q_err: {q_error.round(3)} | dq: {dq.round(3)} | tau_g: {tau_g.round(3)}")

            time.sleep(max(0.0, DT - (time.time() - start_time)))

def input_thread(loop_obj):
    while True:
        cmd = input("Type 'h' + Enter to HOLD, 'f' + Enter to FLOAT: ").strip().lower()
        if cmd == "h":
            loop_obj.hold_enabled = True
        elif cmd == "f":
            print("🟠 FLOAT: Floating mode resumed.")
        else:
            print("Unknown command.")

if __name__ == "__main__":
    zero_pose_capture()
    print("Starting compliant impedance control with gravity compensation and logging...")
    loop = ControlLoop()
    control_thread = threading.Thread(target=loop.run)
    control_thread.daemon = True
    control_thread.start()
    input_thread(loop)
