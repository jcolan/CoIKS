cmake_minimum_required(VERSION 3.0.2)
project(coiks)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    eigen_conversions
)

# Pinnochio
# list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots/")
find_package(pinocchio REQUIRED)

# Casadi
list(APPEND CMAKE_PREFIX_PATH "/usr/local/")
find_package(CASADI REQUIRED)

# RELEASE TYPE
set(CMAKE_BUILD_TYPE Release)

# # Eigen optimizations
# add_definitions(-DEIGEN_NO_DEBUG)

# # Parallel compilation
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES coiks
 CATKIN_DEPENDS roscpp 
 DEPENDS
 )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(coiks 
    src/coiks.cpp
)


# add_dependencies(coiks pinocchio::pinocchio Eigen3::Eigen  casadi)


target_link_libraries(coiks 
    Eigen3::Eigen 
    pinocchio::pinocchio 
    ${catkin_LIBRARIES}
    casadi
)

