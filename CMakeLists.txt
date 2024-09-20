cmake_minimum_required(VERSION 3.10)
add_compile_options(-std=c++17 -O2 -Wall)
set(CMAKE_CXX_STANDARD 23)
project(rbm
        VERSION 1.0
        DESCRIPTION "Calc rbm lib"
        LANGUAGES CXX
)

add_subdirectory(./src)
add_subdirectory(./test)
add_subdirectory(./experiment)
add_executable(${PROJECT_NAME} ./main.cpp)
target_link_libraries(${PROJECT_NAME} libmodel libsampling)
