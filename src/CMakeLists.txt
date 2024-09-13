cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-std=c++17 -O2 -Wall)

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
# target_compile_definitions(libmodel PRIVATE EIGEN_MPL2_ONLY)




# Main
add_library(libmodel
            SHARED
            model.cpp)

add_library(libsampling
            SHARED
            sampling.cpp)

add_library(liblearn
            SHARED
            learn.cpp)

# Utils
add_library(libsigmoid
            SHARED
            utils/sigmoid.cpp)

add_library(libbinary
            SHARED
            utils/binary.cpp)

add_library(libcsv
            SHARED
            utils/csv.cpp)


# data_type
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

add_library(libgrad
            SHARED
            data_types/grad.cpp)


target_compile_definitions(libmodel PRIVATE EIGEN_MPL2_ONLY)
target_include_directories(libmodel
  PUBLIC . 
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libsampling
  PUBLIC . 
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(liblearn
  PUBLIC . 
  PUBLIC ${EIGEN3_INCLUDE_DIR})

#Utils

target_include_directories(libsigmoid
  PUBLIC ./utils
  PUBLIC ${EIGEN3_INCLUDE_DIR})


target_include_directories(libbinary
  PUBLIC ./utils
  ${Boost_LIBRARIES}
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_include_directories(libcsv
  PUBLIC ./utils)


# data_type
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

target_include_directories(libgrad
  PUBLIC ./data_types
  PUBLIC ${EIGEN3_INCLUDE_DIR})

target_link_libraries(
  libmodel
  libparametar
  ${Boost_LIBRARIES}
  Eigen3::Eigen)

target_link_libraries(
  libsampling
  libdata
  libsigmoid
  Eigen3::Eigen)

target_link_libraries(
  liblearn
  libmodel
  libsigmoid
  libgrad)

# Utils
target_link_libraries(
  libsigmoid
  libbias
  Eigen3::Eigen)


target_link_libraries(
  libbinary
  libdata
  ${Boost_LIBRARIES})
  
target_link_libraries(
  libcsv
  libparametar)
  

# data_type
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
