cmake_minimum_required(VERSION 3.10)
add_compile_options(-std=c++17 -O2 -Wall)

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Define executables
add_executable(learn_exact learn_exact/learn_exact.cpp)
add_executable(learn_mnist learn_mnist/learn_mnist.cpp)
add_executable(gendata learn_mnist/genpng/gendata.cpp)

# Link libraries
target_link_libraries(learn_exact
  PRIVATE libmodel
  PRIVATE libsampling
  PRIVATE libbinary
  PRIVATE liblearn
  PRIVATE libfile
  PRIVATE libcsv
  ${Boost_LIBRARIES})

target_link_libraries(learn_mnist
  PRIVATE libmodel
  PRIVATE libsampling
  PRIVATE libbinary
  PRIVATE liblearn
  PRIVATE libcsv
  PRIVATE libfile
  ${Boost_LIBRARIES})

target_link_libraries(gendata
  PRIVATE libmodel
  PRIVATE libsampling
  PRIVATE libbinary
  PRIVATE liblearn
  PRIVATE libcsv
  PRIVATE libfile
  ${Boost_LIBRARIES})
