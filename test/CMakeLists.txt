cmake_minimum_required(VERSION 3.10)
project(test)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

# Enable the testing features.
enable_testing()

# Include GoogleTest
include(GoogleTest)

# Add the executable for the testcase which is using googletest
add_executable(rbm_test rbm_test.cpp)
add_executable(sigmoid_test  sigmoid_test.cpp)
# add_executable(gibbssampling_test  gibbssampling_test/gibbssampling_test.cpp)
# add_executable(rbm_result_test rbm_result_test/rbm_result_test.cpp)
# add_executable(cd_rbm_test cd_rbm_test/cd_rbm_test.cpp)
# add_executable(mnist_test mnist_test/mnist_test.cpp)

target_link_libraries(rbm_test PRIVATE librbm GTest::gtest GTest::gtest_main)
target_link_libraries(sigmoid_test PRIVATE libsigmoid GTest::gtest GTest::gtest_main)
# target_link_libraries(gibbssampling_test PRIVATE librbm)
# target_link_libraries(rbm_result_test
#   PRIVATE librbm
#   PRIVATE arg_config
#   ${Boost_LIBRARIES}
# )
# target_link_libraries(cd_rbm_test
#   PRIVATE librbm
#   PRIVATE arg_config
#   ${Boost_LIBRARIES}
# )
# target_link_libraries(mnist_test
#   PRIVATE librbm
#   PRIVATE arg_config
#   PRIVATE csv2vector
#   ${Boost_LIBRARIES}
# )

# Discover tests
gtest_discover_tests(rbm_test)
# add_test(NAME SigmoidTest)
# add_test(NAME RbmTest)

# ctestの設定
add_test(NAME SigmoidTests COMMAND sigmoid_test)
add_test(NAME RBMTests COMMAND rbm_test)
