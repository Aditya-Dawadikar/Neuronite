cmake_minimum_required(VERSION 3.10)
project(Neuronite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")


include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(neuronite main.cpp ${SOURCES})
