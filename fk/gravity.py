import time
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# === Config ===
URDF_PATH = "robot.urdf"
MESH_DIR = "meshes"

# Put your active joints here (e.g., [0,1,2,3,4,5,6] for all, or [0] for single joint test)
MOTOR_IDS = [0,1]
# If any joint is reversed relative to URDF, put -1 for that joint in this array
SIGN_FLIPS = np.array([1, 1, -1, 1, 1, 1, 1])  # Change -1 where needed

DXL_RESOLUTION = 4096
TORQUE_CONSTANTS = np.array([0.6] * 7)  # For XM430/XL430; use 2.2 for XH540, etc.
DXL_CURRENT_UNIT = 2.69e-3  # A/LSB for XM/XL430

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 4000000
PROTOCOL_VERSION = 2.0

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11

CONTROL_FREQ = 50
DT = 1.0 / CONTROL_FREQ

# === Pinocchio Model ===
robot = RobotWrapper.BuildFromURDF(URDF_PATH, [MESH_DIR])
model = robot.model
data = robot.data

# === Dynamixel Setup ===
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
assert portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
groupRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
groupWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

for dxl_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, 0)  # Current mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
    groupRead.addParam(dxl_id)

def raw_to_rad(raw, center, sign=1):
    return sign * (raw - center) * (2 * np.pi / DXL_RESOLUTION)

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

def compute_gravity_torque(q_full):
    pin.computeGeneralizedGravity(model, data, q_full)
    return data.g.copy()

def send_currents(currents):
    groupWrite.clearParam()
    for i, dxl_id in enumerate(MOTOR_IDS):
        amps = currents[i] / TORQUE_CONSTANTS[dxl_id]
        current_lsb = int(np.clip(amps / DXL_CURRENT_UNIT, -2047, 2047))
        param = [current_lsb & 0xFF, (current_lsb >> 8) & 0xFF]
        groupWrite.addParam(dxl_id, param)
    groupWrite.txPacket()

if __name__ == "__main__":
    zero_pose_capture()
    print("Starting gravity-hold. Use Ctrl+C to stop.")
    while True:
        start_time = time.time()
        groupRead.txRxPacket()
        q_raw = [groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION) for dxl_id in MOTOR_IDS]
        q = np.zeros(7)
        for i, dxl_id in enumerate(MOTOR_IDS):
            q[dxl_id] = raw_to_rad(q_raw[i], raw_zero[dxl_id], SIGN_FLIPS[dxl_id])
        tau_g = compute_gravity_torque(q)
        send_currents(tau_g[MOTOR_IDS])

        # --- Debug Info ---
        for i, dxl_id in enumerate(MOTOR_IDS):
            amps = tau_g[dxl_id] / TORQUE_CONSTANTS[dxl_id]
            lsb = amps / DXL_CURRENT_UNIT
            print(f"J{dxl_id}: q={q[dxl_id]:.3f} rad | tau_g={tau_g[dxl_id]:.3f} Nm | amps={amps:.3f} A | lsb={lsb:.1f}")
        time.sleep(max(0.0, DT - (time.time() - start_time)))
