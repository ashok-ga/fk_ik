import time
from dynamixel_sdk import *  # Dynamixel SDK

# ----- Control Table Addresses -----
ADDR_PRESENT_POSITION = 132   # XL430, XM430
LEN_PRESENT_POSITION = 4

# ----- Communication Settings -----
PROTOCOL_VERSION = 2.0
DXL_BAUDRATE = 4000000
DXL_PORT = "/dev/ttyUSB0"
DXL_IDS = list(range(0, 7))  # Dynamixel IDs 0 to 6

# ----- Initialize PortHandler -----
portHandler = PortHandler(DXL_PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print(f"❌ Failed to open port {DXL_PORT}")
    quit()

# Set baudrate
if not portHandler.setBaudRate(DXL_BAUDRATE):
    print(f"❌ Failed to set baudrate to {DXL_BAUDRATE}")
    quit()

# ----- Initialize GroupSyncRead -----
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

# Add parameter storage for each ID
for dxl_id in DXL_IDS:
    if not groupSyncRead.addParam(dxl_id):
        print(f"❌ Failed to add ID {dxl_id} to group sync read")
        quit()

print("✅ Start reading from Dynamixel motors (IDs 0–6)... Press Ctrl+C to stop.")

# ----- Main Loop -----
try:
    while True:
        start_time = time.time()

        # Send group sync read request
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
  #          print(f"❌ Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            continue

        # Read values
        for dxl_id in DXL_IDS:
            if groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                pos = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
#                print(f"[ID:{dxl_id}] Pos: {pos}", end=" | ")
 #           else:
 #               print(f"[ID:{dxl_id}] ❌ No data", end=" | ")

        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        print(f"FPS: {fps:.1f}")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

# ----- Cleanup -----
portHandler.closePort()

