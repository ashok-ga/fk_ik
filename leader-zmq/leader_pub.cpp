#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sstream>
#include <zmq.hpp>
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "dynamixel_sdk/group_fast_bulk_read.h"

using namespace dynamixel;

#define PROTOCOL_VERSION        2.0
#define BAUDRATE                4000000
#define DEVICENAME              "/dev/ttyUSB0"
#define ADDR_PRESENT_POSITION   132
#define LEN_PRESENT_POSITION    4
#define ZMQ_PORT                5556

int main() {
    PortHandler *portHandler = PortHandler::getPortHandler(DEVICENAME);
    PacketHandler *packetHandler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    GroupFastBulkRead groupFastBulkRead(portHandler, packetHandler);
    std::vector<uint8_t> dxl_ids = {0, 1, 2, 3, 4, 5, 6};

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) return 1;

    for (uint8_t id : dxl_ids) {
        if (!groupFastBulkRead.addParam(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) return 1;
    }
    zmq::context_t context(1);
    zmq::socket_t publisher(context, ZMQ_PUB);
    publisher.bind("tcp://*:5556");

    while (true) {
        if (groupFastBulkRead.txRxPacket() != COMM_SUCCESS) continue;

        std::ostringstream msg;
        msg << "{";

        for (size_t i = 0; i < dxl_ids.size(); ++i) {
            uint8_t id = dxl_ids[i];
            if (!groupFastBulkRead.isAvailable(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) continue;
            int32_t pos = groupFastBulkRead.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
            msg << "\"id" << (int)id << "\":" << pos;
            if (i < dxl_ids.size() - 1) msg << ",";
        }

        msg << "}";
        publisher.send(zmq::buffer(msg.str()), zmq::send_flags::none);
    }

    portHandler->closePort();
    return 0;
}
