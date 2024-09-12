cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-std=c++17 -O2 -Wall)
add_library(librbm
            SHARED
            rbm.cpp)

add_library(libsampling
            SHARED
            sampling.cpp)

add_library(libsigmoid
            SHARED
            utils/sigmoid.cpp)


add_library(libbias
            SHARED
            data_types/bias.cpp)

add_library(libweight
            SHARED
            data_types/weight.cpp)

add_library(libparametar
            SHARED
            data_types/parametar.cpp)

add_library(libdata
            SHARED
            data_types/data.cpp)


  # Serch Boost
find_package(Boost REQUIRED COMPONENTS)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Serch Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()

target_include_directories(librbm
  PUBLIC . 
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libsigmoid
  PUBLIC ./utils
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libsampling
  PUBLIC . 
  PUBLIC ${EIGEN3_INCLUDE_DIR})


target_include_directories(libbias
  PUBLIC ./data_types
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libweight
  PUBLIC ./data_types
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libparametar
  PUBLIC ./data_types
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libdata
  PUBLIC ./data_types
  PUBLIC ${EIGEN3_INCLUDE_DIR})

  # target_include_directories(csv2vector
  #   PUBLIC ./utils
  #   PUBLIC ${EIGEN3_INCLUDE_DIR}
  #   )

target_compile_definitions(librbm PRIVATE EIGEN_MPL2_ONLY)
target_link_libraries(
  librbm
  libparametar
  ${Boost_LIBRARIES}
  Eigen3::Eigen)

target_link_libraries(
  libsampling
  libdata
  libsigmoid
  Eigen3::Eigen)
# target_link_libraries(
#   libsigmoid
#   librbm
#   Eigen3::Eigen)

target_link_libraries(
  libsigmoid
  libbias
  Eigen3::Eigen)

target_link_libraries(
  libbias
  Eigen3::Eigen)

target_link_libraries(
  libweight
  Eigen3::Eigen)

target_link_libraries(
  libparametar
  libbias
  libweight
  Eigen3::Eigen)

target_link_libraries(
  Eigen3::Eigen)
# target_link_libraries(
#   arg_config
#   ${Boost_LIBRARIES}
# )
# 
# target_link_libraries(
#   csv2vector
#   ${Boost_LIBRARIES}
#   Eigen3::Eigen
# )
