
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
TORQUE_CONSTANT = 1.1  # Nm/A

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
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, 0)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
    groupRead.addParam(dxl_id)

def raw_to_rad(raw, center):
    return (raw - center) * (2 * np.pi / DXL_RESOLUTION)

def compute_gravity_torque(q):
    return np.zeros(7)

def send_currents(currents):
    groupWrite.clearParam()
    for i, dxl_id in enumerate(MOTOR_IDS):
        amps = currents[i] / TORQUE_CONSTANT
        current_val = int(np.clip(amps, -2047, 2047))
        param = [current_val & 0xFF, (current_val >> 8) & 0xFF]
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

def control_loop():
    q_prev = np.zeros(7)
    q_target = np.zeros(7)
    still_counter = 0
    debug_counter = 0

    log_file = open("impedance_log.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["time"] + [f"q{i}" for i in range(7)] + [f"q_err{i}" for i in range(7)] + [f"dq{i}" for i in range(7)] + [f"tau{i}" for i in range(7)])

    while True:
        start_time = time.time()
        groupRead.txRxPacket()
        q_raw = [groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION) for dxl_id in MOTOR_IDS]
        q = np.array([raw_to_rad(q_raw[i], raw_zero[MOTOR_IDS[i]]) for i in range(7)])
        dq = (q - q_prev) / DT
        q_prev = q.copy()

        if np.linalg.norm(dq) < VEL_THRESHOLD:
            still_counter += 1
        else:
            still_counter = 0

        if still_counter >= STILL_CYCLES:
            q_target = q.copy()
            still_counter = 0
            print("🟢 Updated floating hold pose")

        q_error = np.clip(q - q_target, -0.5, 0.5)  # Limit error magnitude
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
            print(f"DEBUG q: {q.round(3)} | q_err: {q_error.round(3)} | dq: {dq.round(3)}")

        time.sleep(max(0.0, DT - (time.time() - start_time)))

if __name__ == "__main__":
    zero_pose_capture()
    print("Starting compliant impedance control with logging...")
    control_thread = threading.Thread(target=control_loop)
    control_thread.start()