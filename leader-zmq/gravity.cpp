#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace dynamixel;
using namespace Eigen;

constexpr char DEVICENAME[] = "/dev/ttyUSB0";
constexpr int BAUDRATE = 4000000;
constexpr float TORQUE_CONSTANT = 1.1f;
constexpr float DXL_RESOLUTION = 4096.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr int NUM_JOINTS = 7;
constexpr float DT = 1.0f / 50.0f;

const uint16_t ADDR_PRESENT_POSITION = 132;
const uint16_t ADDR_GOAL_CURRENT = 102;
const uint8_t ADDR_OPERATING_MODE = 11;
const uint8_t ADDR_TORQUE_ENABLE = 64;

PortHandler *portHandler;
PacketHandler *packetHandler;
GroupSyncRead *groupRead;
GroupSyncWrite *groupWrite;
std::vector<uint8_t> MOTOR_IDS = {0, 1, 2, 3, 4, 5, 6};
std::map<uint8_t, int32_t> raw_zero;

Matrix<float, NUM_JOINTS, 1> rawToRad(const std::vector<int32_t>& raw) {
    Matrix<float, NUM_JOINTS, 1> q;
    for (int i = 0; i < NUM_JOINTS; ++i)
        q(i) = (raw[i] - raw_zero[MOTOR_IDS[i]]) * (2 * PI / DXL_RESOLUTION);
    return q;
}

void sendCurrents(const VectorXf& tau) {
    groupWrite->clearParam();
    for (int i = 0; i < NUM_JOINTS; ++i) {
        float amps = tau(i) / TORQUE_CONSTANT;
        int current = static_cast<int>(std::round(amps));
        current = std::min(std::max(current, -2047), 2047);
        uint8_t param[2] = { static_cast<uint8_t>(current & 0xFF), static_cast<uint8_t>((current >> 8) & 0xFF) };
        groupWrite->addParam(MOTOR_IDS[i], param);
    }
    groupWrite->txPacket();
}

void configureMotors() {
    for (auto id : MOTOR_IDS) {
        int dxl_comm_result;
        dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 0);
        dxl_comm_result |= packetHandler->write1ByteTxRx(portHandler, id, ADDR_OPERATING_MODE, 0);
        dxl_comm_result |= packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 1);
        if (dxl_comm_result != COMM_SUCCESS) {
            std::cerr << "Failed to configure motor ID " << static_cast<int>(id) << std::endl;
        }
        groupRead->addParam(id);
    }
}

int main() {
    portHandler = PortHandler::getPortHandler(DEVICENAME);
    packetHandler = PacketHandler::getPacketHandler(2.0);
    groupRead = new GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4);
    groupWrite = new GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, 2);

    portHandler->openPort();
    portHandler->setBaudRate(BAUDRATE);

    configureMotors();

    std::cout << "Capturing zero pose...\n";
    while (true) {
        groupRead->txRxPacket();
        bool all_available = true;
        for (auto id : MOTOR_IDS) {
            if (!groupRead->isAvailable(id, ADDR_PRESENT_POSITION, 4)) {
                all_available = false;
                break;
            }
        }
        if (all_available) {
            for (auto id : MOTOR_IDS) {
                raw_zero[id] = groupRead->getData(id, ADDR_PRESENT_POSITION, 4);
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Zero pose captured.\n";

    VectorXf q_prev = VectorXf::Zero(NUM_JOINTS);
    VectorXf q_target = VectorXf::Zero(NUM_JOINTS);
    const Matrix<float, NUM_JOINTS, NUM_JOINTS> STIFFNESS = Vector<float, NUM_JOINTS>({10, 30, 20, 4, 2, 0.5f, 0.0f}).asDiagonal();
    const Matrix<float, NUM_JOINTS, NUM_JOINTS> DAMPING = Vector<float, NUM_JOINTS>({2, 5, 2.2f, 2, 0.8f, 0.8f, 0.1f}).asDiagonal();
    const Matrix<float, NUM_JOINTS, NUM_JOINTS> FEEDFORWARD = Vector<float, NUM_JOINTS>({8, 10, 10, 4, 1, 1, 0.1f}).asDiagonal();

    int still_counter = 0;
    const float vel_thresh = 0.01f;

    while (true) {
        groupRead->txRxPacket();
        std::vector<int32_t> raw;
        for (auto id : MOTOR_IDS)
            raw.push_back(groupRead->getData(id, ADDR_PRESENT_POSITION, 4));

        VectorXf q = rawToRad(raw);
        VectorXf dq = (q - q_prev) / DT;
        q_prev = q;

        if (dq.norm() < vel_thresh)
            still_counter++;
        else
            still_counter = 0;

        if (still_counter > 30) {
            q_target = q;
            still_counter = 0;
            std::cout << "Updated floating hold pose.\n";
        }

        VectorXf q_error = (q - q_target).cwiseMax(-0.5f).cwiseMin(0.5f);
        VectorXf tau = -STIFFNESS * q_error - DAMPING * dq + FEEDFORWARD * dq;
        sendCurrents(tau);

        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(DT * 1000)));
    }

    return 0;
}