#include <iostream>
#include <chrono>
#include <vector>
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "dynamixel_sdk/group_fast_bulk_read.h"  // Include Fast Bulk Read

using namespace dynamixel;
using namespace std::chrono;

// --- Constants ---
#define PROTOCOL_VERSION        2.0
#define BAUDRATE                4000000
#define DEVICENAME              "/dev/ttyUSB0"
#define ADDR_PRESENT_POSITION   132
#define LEN_PRESENT_POSITION    4

int main() {
    // Setup SDK handlers
    PortHandler *portHandler = PortHandler::getPortHandler(DEVICENAME);
    PacketHandler *packetHandler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    GroupFastBulkRead groupFastBulkRead(portHandler, packetHandler);

    std::vector<uint8_t> dxl_ids = {0, 1, 2, 3, 4, 5, 6};

    // Open port
    if (!portHandler->openPort()) {
        std::cerr << "❌ Failed to open port\n";
        return 1;
    }
    if (!portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "❌ Failed to set baudrate\n";
        return 1;
    }

    // Add params to GroupFastBulkRead
    for (uint8_t id : dxl_ids) {
        if (!groupFastBulkRead.addParam(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) {
            std::cerr << "❌ Failed to add ID " << (int)id << " to fast bulk read\n";
            return 1;
        }
    }

    std::cout << "✅ Starting Fast Bulk Read loop...\n";

    int count = 0;
    auto t_start = high_resolution_clock::now();

    while (true) {
        int result = groupFastBulkRead.txRxPacket();
        if (result != COMM_SUCCESS) {
            std::cerr << "❌ Comm error: " << packetHandler->getTxRxResult(result) << "\n";
            continue;
        }

        for (uint8_t id : dxl_ids) {
            if (groupFastBulkRead.isAvailable(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) {
                int32_t pos = groupFastBulkRead.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
                // Optional: log pos
            } else {
                std::cerr << "⚠️ ID " << (int)id << " data unavailable\n";
            }
        }

        count++;
        if (count % 500 == 0) {
            auto t_now = high_resolution_clock::now();
            double elapsed = duration<double>(t_now - t_start).count();
            double fps = count / elapsed;
            std::cout << "FPS: " << fps << std::endl;
        }
    }

    portHandler->closePort();
    return 0;
}
