import time
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 4000000
PROTOCOL_VERSION = 2.0

MOTOR_IDS = [0, 1, 2, 3, 4, 5, 6]  # Update for your setup!
TORQUE_CONSTANTS = np.array([1.1] * 7)  # [Nm/A]
DXL_CURRENT_UNIT = 2.69e-3  # [A/LSB]

ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11

TORQUE_MAGNITUDE = 0.15  # Nm (safe test value; lower if you want less movement)
HOLD_TIME = 1.0  # seconds per direction

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
assert portHandler.openPort(), "Failed to open port"
portHandler.setBaudRate(BAUDRATE)

def set_current(dxl_id, torque_nm):
    amps = torque_nm / TORQUE_CONSTANTS[dxl_id]
    current_lsb = int(np.clip(amps / DXL_CURRENT_UNIT, -2047, 2047))
    param = [current_lsb & 0xFF, (current_lsb >> 8) & 0xFF]
    packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_GOAL_CURRENT, current_lsb)

def torque_on(dxl_id):
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, 0)  # Current mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

def torque_off(dxl_id):
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)

print("=== Motor Direction Test ===")
for i, dxl_id in enumerate(MOTOR_IDS):
    print(f"\nTesting joint {i} (Dynamixel ID {dxl_id})")
    torque_on(dxl_id)
    print(f"Applying +{TORQUE_MAGNITUDE} Nm for {HOLD_TIME} seconds.")
    set_current(dxl_id, +TORQUE_MAGNITUDE)
    time.sleep(HOLD_TIME)
    set_current(dxl_id, 0)
    input("Did the joint move in the + direction? (Y/n): ")

    print(f"Applying -{TORQUE_MAGNITUDE} Nm for {HOLD_TIME} seconds.")
    set_current(dxl_id, -TORQUE_MAGNITUDE)
    time.sleep(HOLD_TIME)
    set_current(dxl_id, 0)
    input("Did the joint move in the - direction? (Y/n): ")

    torque_off(dxl_id)
    print("Torque OFF for this joint. Press Enter for next.")
    input()

print("Direction test complete. All joints OFF.")
portHandler.closePort()
