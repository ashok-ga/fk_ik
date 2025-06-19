cmake_minimum_required(VERSION 3.0)
project(fk)

find_package(pinocchio REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

set(DYNAMIXEL_SDK_INCLUDE_DIR "/home/ga/DynamixelSDK/c++/include")
set(DYNAMIXEL_SDK_LIB_DIR "/home/ga/DynamixelSDK/c++/build/linux64")
include_directories(${DYNAMIXEL_SDK_INCLUDE_DIR})
link_directories(${DYNAMIXEL_SDK_LIB_DIR})

add_executable(run_fk main.cpp)
target_link_libraries(run_fk
    pinocchio::pinocchio
    ${ZMQ_LIBRARIES}
    zmq
    dxl_x64_cpp    # <-- Correct name for your SDK lib
)
