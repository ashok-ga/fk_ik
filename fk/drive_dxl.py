import time
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite

# === CONFIGURABLE PARAMETERS ===
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 4000000
PROTOCOL_VERSION = 2.0

MOTOR_IDS = [0, 1, 2, 3, 4, 5, 6]
DXL_RESOLUTION = 4096
TORQUE_CONSTANTS = np.array([1.1] * 7)
DXL_CURRENT_UNIT = 2.69e-3  # [A/LSB]

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

OPERATING_MODE = 0     # 0 = Current mode (direct torque)
CURRENT_LIMIT = 700    # About 1.9A for XM/XL

# Stiffness (Nm/rad) and Damping (Nm/(rad/s)), tune per joint!
STIFFNESS = np.array([1.0, 1.0,  1.0, 1.0, 0.8, 0.8, 0.0])
DAMPING   = np.array([1.0, 1.0, 1.2, 1.0, 0.8, 0.8, 0.1])
feedforward_nm = np.array([1.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0])

CONTROL_FREQ = 50
DT = 1.0 / CONTROL_FREQ

# === Setup SDK ===
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
assert portHandler.openPort(), "Failed to open port"
portHandler.setBaudRate(BAUDRATE)

groupRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
groupWriteCurrent = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

for dxl_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE)
    packetHandler.write2ByteTxRx(portHandler, dxl_id, 38, CURRENT_LIMIT) # ADDR_CURRENT_LIMIT
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
    groupRead.addParam(dxl_id)

print(f"Operating mode: {OPERATING_MODE}, Current limit: {CURRENT_LIMIT}")

def rad_to_raw(rad):
    return int((rad % (2 * np.pi)) * (DXL_RESOLUTION / (2 * np.pi)))

def raw_to_rad(raw):
    return (raw % DXL_RESOLUTION) * (2 * np.pi / DXL_RESOLUTION)

# --- Capture Initial Joint Angles ---
print("Reading initial position to hold...")
init_raw = []
while True:
    groupRead.txRxPacket()
    all_valid = True
    vals = []
    for dxl_id in MOTOR_IDS:
        if groupRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            vals.append(groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION))
        else:
            all_valid = False
            break
    if all_valid:
        init_raw = vals
        break
    time.sleep(0.1)
hold_positions_raw = np.array(init_raw)
hold_positions_rad = np.array([raw_to_rad(x) for x in hold_positions_raw])
print("Holding initial positions (rad):", hold_positions_rad.round(3))

# Initialize state for velocity calculation
prev_rad = hold_positions_rad.copy()
prev_time = time.time()
vel_rad = np.zeros(7)

def send_currents(currents_nm):
    groupWriteCurrent.clearParam()
    for i, dxl_id in enumerate(MOTOR_IDS):
        amps = currents_nm[i] / TORQUE_CONSTANTS[i]
        current_lsb = int(np.clip(amps / DXL_CURRENT_UNIT, -2047, 2047))
        param_current = [current_lsb & 0xFF, (current_lsb >> 8) & 0xFF]
        groupWriteCurrent.addParam(dxl_id, bytearray(param_current))
    groupWriteCurrent.txPacket()
    groupWriteCurrent.clearParam()

print("Impedance-control: compliant hold. Move arm to change hold pos. Ctrl+C to stop.")
try:
    while True:
        groupRead.txRxPacket()
        curr_raw = []
        curr_rad = []
        for dxl_id in MOTOR_IDS:
            raw = groupRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            curr_raw.append(raw)
            curr_rad.append(raw_to_rad(raw))
        curr_rad = np.array(curr_rad)
        
        # Velocity estimate (simple finite diff)
        now = time.time()
        dt = now - prev_time
        vel_rad = (curr_rad - prev_rad) / max(dt, 1e-4)
        prev_rad = curr_rad.copy()
        prev_time = now
        
        # --- Teleoperation: Set hold_positions_rad from your teleop master here ---
        # Example: Replace the next line with target positions from your teleop master
        q_target = hold_positions_rad.copy()
        
        # --- Optional: If joint moved by hand, update hold pos (for "leader" mode) ---
        UPDATE_THRESHOLD = 0.01
        diff = np.abs(curr_rad - hold_positions_rad)
        if np.any(diff > UPDATE_THRESHOLD):
            print(f"Moved! Updating hold position to: {curr_rad.round(3)}")
            hold_positions_rad = curr_rad.copy()
            q_target = curr_rad.copy()
        
        # --- Impedance control ---
        # τ = Kp*(q_target - q) - Kd*q_dot + τ_ff
        tau_nm = (
            STIFFNESS * (q_target - curr_rad)
            - DAMPING * vel_rad
            + feedforward_nm
        )
        
        send_currents(tau_nm)
        time.sleep(DT)
except KeyboardInterrupt:
    print("Exiting and disabling torque...")
    for dxl_id in MOTOR_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
