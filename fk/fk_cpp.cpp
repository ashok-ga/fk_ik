#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/visualize/meshcat-visualizer.hpp>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

int main() {
    // ---- Setup paths ----
    std::string urdf_path = "leader.urdf";
    std::string mesh_dir = "meshes"; // Set this to your mesh path
    std::string ee_frame = "trigger_1";

    // ---- Load model ----
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    // ---- Visualizer ----
    auto viz = std::make_shared<pinocchio::MeshcatVisualizer>();
    viz->initViewer();
    viz->loadViewerModel(model, mesh_dir);
    std::cout << "Meshcat URL: http://127.0.0.1:7000/static/" << std::endl;

    // ---- Show neutral pose first ----
    Eigen::VectorXd q = pinocchio::neutral(model);
    viz->display(q);
    std::cout << "Displayed robot at home (neutral) configuration." << std::endl;

    // ---- Print joint info ----
    std::cout << "Joint lower limits: " << model.lowerPositionLimit.transpose() << std::endl;
    std::cout << "Joint upper limits: " << model.upperPositionLimit.transpose() << std::endl;
    std::cout << "Number of joints: " << model.nq << std::endl;

    // ---- Find EE frame ID ----
    pinocchio::FrameIndex ee_id = model.getFrameId(ee_frame);
    if (ee_id == model.nframes) {
        std::cerr << "End-effector frame '" << ee_frame << "' not found!" << std::endl;
        return 1;
    }

    // ---- Main loop: animate random configurations ----
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dist;
    std::cout << "\nAnimating to random configurations... (Ctrl+C to quit)" << std::endl;

    while (true) {
        // Random valid configuration
        Eigen::VectorXd q_rand(model.nq);
        for (int i = 0; i < model.nq; ++i) {
            dist.param(std::uniform_real_distribution<>::param_type(model.lowerPositionLimit[i], model.upperPositionLimit[i]));
            q_rand[i] = dist(gen);
        }

        // FK and update
        try {
            pinocchio::forwardKinematics(model, data, q_rand);
            pinocchio::updateFramePlacements(model, data);
            viz->display(q_rand);

            // Print joint/EE pose
            const auto &oMf = data.oMf[ee_id];
            std::cout << "q: " << q_rand.transpose() << std::endl;
            std::cout << "End-effector position: " << oMf.translation().transpose() << std::endl;
            std::cout << "End-effector rotation:\n" << oMf.rotation() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Failed to update configuration: " << e.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
