cmake_minimum_required(VERSION 3.0)
project (cf CXX)
add_executable(cf main.cpp)
target_compile_features(cf PRIVATE cxx_range_for)
