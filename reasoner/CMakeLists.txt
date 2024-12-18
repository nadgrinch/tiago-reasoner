cmake_minimum_required(VERSION 3.8)
project(tiago-reasoner)

if(CMAKE_COMPILER_IS_GNUCXX or CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/DeicticSolution.msg"
    "msg/GDRNObject.msg"
    "msg/GDRNSolution.msg"
    "msg/HRICommand.msg"
    DEPENDECIES std_msgs geometry_msgs
)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
    reasoner/reasoner_node.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
