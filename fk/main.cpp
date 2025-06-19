#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <zmq.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <Eigen/Dense>
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "dynamixel_sdk/group_fast_bulk_read.h"

using namespace dynamixel;

#define PROTOCOL_VERSION        2.0
#define BAUDRATE                4000000
#define DEVICENAME              "/dev/ttyUSB0"
#define ADDR_PRESENT_POSITION   132
#define LEN_PRESENT_POSITION    4
#define ZMQ_PORT                5556

std::vector<double> latest_angles;
std::mutex joint_mutex;
std::atomic<bool> running{true};

void dynamixel_reader(const std::vector<uint8_t>& dxl_ids) {
    PortHandler *portHandler = PortHandler::getPortHandler(DEVICENAME);
    PacketHandler *packetHandler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    GroupFastBulkRead groupFastBulkRead(portHandler, packetHandler);

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "Failed to open port or set baudrate!" << std::endl;
        running = false;
        return;
    }

    for (uint8_t id : dxl_ids)
        groupFastBulkRead.addParam(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);

    while (running) {
        if (groupFastBulkRead.txRxPacket() != COMM_SUCCESS) continue;

        std::vector<double> tmp_angles(dxl_ids.size(), 0.0);

        for (size_t i = 0; i < dxl_ids.size(); ++i) {
            uint8_t id = dxl_ids[i];
            if (!groupFastBulkRead.isAvailable(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) continue;
            int32_t pos = groupFastBulkRead.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
            double angle_rad = (pos / 4096.0) * 2.0 * M_PI; // adjust if your DXL has a different scale!
            tmp_angles[i] = angle_rad;
        }
        {
            std::lock_guard<std::mutex> lock(joint_mutex);
            latest_angles = tmp_angles;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    portHandler->closePort();
}

void print_vector(const std::vector<double>& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_matrix(const Eigen::Matrix3d& m) {
    for (int r = 0; r < 3; ++r) {
        std::cout << "[ ";
        for (int c = 0; c < 3; ++c)
            std::cout << m(r,c) << " ";
        std::cout << "]\n";
    }
}

void fk_and_publish(const std::string& urdf_path, const std::string& ee_name, size_t num_joints) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    // Print all joint names
    std::cout << "Loaded model with " << model.nq << " DOF\n";
    std::cout << "Joint names (index in order):\n";
    for (size_t i = 1; i < model.joints.size(); ++i)
        std::cout << "  [" << i << "]: " << model.names[i] << "\n";

    // Print all frame names and types
    std::cout << "\nAvailable frames (index, name, type):\n";
    for (size_t i = 0; i < model.frames.size(); ++i)
        std::cout << "  [" << i << "]: " << model.frames[i].name
                  << " (type=" << model.frames[i].type << ")\n";

    pinocchio::FrameIndex fid = model.getFrameId(ee_name);
    if (fid == (pinocchio::FrameIndex)-1) {
        std::cerr << "ERROR: Frame '" << ee_name << "' not found in model!\n";
        running = false;
        return;
    }
    std::cout << "\nUsing end-effector frame: " << ee_name << " (frame ID " << fid << ")"
              << " with type " << model.frames[fid].type << std::endl;

    zmq::context_t context(1);
    zmq::socket_t publisher(context, ZMQ_PUB);
    publisher.bind("tcp://*:5556");

    std::vector<double> prev_angles;
    int iter = 0;

    while (running) {
        std::vector<double> joint_angles_copy;
        {
            std::lock_guard<std::mutex> lock(joint_mutex);
            joint_angles_copy = latest_angles;
        }
        if (joint_angles_copy.size() != num_joints) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (prev_angles.empty())
            prev_angles = joint_angles_copy;

        std::vector<double> angle_delta(num_joints);
        for (size_t i = 0; i < num_joints; ++i)
            angle_delta[i] = joint_angles_copy[i] - prev_angles[i];

        prev_angles = joint_angles_copy;

        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        for (size_t i = 0; i < num_joints && i < (size_t)model.nq; ++i)
            q[i] = joint_angles_copy[i];

        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacements(model, data);
        const auto& oMf = data.oMf[fid];

        // --- DEBUGGING: Print every 200 iterations ---
        if (++iter % 200 == 0) {
            std::cout << "\n[Debug] Joint vector (q): ";
            print_vector(joint_angles_copy);
            std::cout << "[Debug] FK end-eff pos: " << oMf.translation().transpose() << std::endl;
            std::cout << "[Debug] FK end-eff rot matrix:\n";
            print_matrix(oMf.rotation());

            // Check if rotation matrix is close to identity (all zeros?) or not orthogonal or has NaN
            if (oMf.rotation().isZero(1e-8))
                std::cout << "[Warning] Rotation matrix is all zeros!\n";
            else if (!oMf.rotation().transpose().isApprox(oMf.rotation().inverse(), 1e-6))
                std::cout << "[Warning] Rotation matrix is not orthogonal!\n";
            if (oMf.rotation().hasNaN())
                std::cout << "[Error] Rotation matrix contains NaN!\n";
        }

        // Compose a binary message [deltas..., pos(x,y,z), rot(3x3 row-major)]
        std::vector<float> msg_bin;
        msg_bin.reserve(num_joints + 3 + 9);
        for (size_t i = 0; i < num_joints; ++i)
            msg_bin.push_back(static_cast<float>(angle_delta[i]));
        msg_bin.push_back(static_cast<float>(oMf.translation().x()));
        msg_bin.push_back(static_cast<float>(oMf.translation().y()));
        msg_bin.push_back(static_cast<float>(oMf.translation().z()));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                msg_bin.push_back(static_cast<float>(oMf.rotation()(r, c)));

        publisher.send(zmq::buffer(msg_bin.data(), msg_bin.size() * sizeof(float)), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int main() {
    std::vector<uint8_t> dxl_ids = {0, 1, 2, 3, 4, 5, 6};
    latest_angles.resize(dxl_ids.size(), 0.0);

    std::string urdf_path = "leader.urdf";
    std::string ee_name = "trigger_1";  // Make sure this matches your URDF tip

    std::thread read_thread(dynamixel_reader, dxl_ids);
    std::thread fk_thread(fk_and_publish, urdf_path, ee_name, dxl_ids.size());

    std::cout << "Press ENTER to quit...\n";
    std::cin.get();
    running = false;
    read_thread.join();
    fk_thread.join();
    return 0;
}
