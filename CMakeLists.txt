cmake_minimum_required(VERSION 3.12)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

add_executable(cnn main.cpp utility.cpp utility.h ConvulutionalNeuralNetwork.cpp ConvulutionalNeuralNetwork.h utilityIO.cpp utilityIO.h)