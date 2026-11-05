cmake_minimum_required(VERSION 3.10)
project(magnetic_gripper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GZ_SIM REQUIRED gz-sim8)
pkg_check_modules(GZ_PLUGIN REQUIRED gz-plugin2)
pkg_check_modules(GZ_TRANSPORT REQUIRED gz-transport13)
pkg_check_modules(GZ_MSGS REQUIRED gz-msgs10)
pkg_check_modules(GZ_MATH REQUIRED gz-math7)

add_library(magnetic_gripper SHARED MagneticGripper.cc)

target_include_directories(magnetic_gripper PRIVATE
  ${GZ_SIM_INCLUDE_DIRS}
  ${GZ_PLUGIN_INCLUDE_DIRS}
  ${GZ_TRANSPORT_INCLUDE_DIRS}
  ${GZ_MSGS_INCLUDE_DIRS}
  ${GZ_MATH_INCLUDE_DIRS}
)

target_link_directories(magnetic_gripper PRIVATE
  ${GZ_SIM_LIBRARY_DIRS}
  ${GZ_PLUGIN_LIBRARY_DIRS}
  ${GZ_TRANSPORT_LIBRARY_DIRS}
  ${GZ_MSGS_LIBRARY_DIRS}
  ${GZ_MATH_LIBRARY_DIRS}
)

target_link_libraries(magnetic_gripper
  ${GZ_SIM_LIBRARIES}
  ${GZ_PLUGIN_LIBRARIES}
  ${GZ_TRANSPORT_LIBRARIES}
  ${GZ_MSGS_LIBRARIES}
  ${GZ_MATH_LIBRARIES}
)
# Standard "lib" prefix kept (-> libmagnetic_gripper.so), matching every
# other gz-sim plugin's naming convention - <plugin filename="magnetic_gripper">
# in SDF resolves against that prefix automatically.
