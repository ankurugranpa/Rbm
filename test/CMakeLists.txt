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
add_executable(model_test model_test.cpp)
add_executable(sigmoid_test  sigmoid_test.cpp)
add_executable(binary_test  binary_test.cpp)
add_executable(grad_test  grad_test.cpp)
add_executable(bias_test bias_test.cpp)
add_executable(weight_test weight_test.cpp)
add_executable(softplus_test softplus_test.cpp)
# add_executable(gibbssampling_test  gibbssampling_test/gibbssampling_test.cpp)
# add_executable(rbm_result_test rbm_result_test/rbm_result_test.cpp)
# add_executable(cd_model_test cd_model_test/cd_model_test.cpp)
# add_executable(mnist_test mnist_test/mnist_test.cpp)

target_link_libraries(model_test PRIVATE libmodel GTest::gtest GTest::gtest_main)
target_link_libraries(sigmoid_test PRIVATE libsigmoid GTest::gtest GTest::gtest_main)
target_link_libraries(binary_test PRIVATE libbinary GTest::gtest GTest::gtest_main)
target_link_libraries(grad_test PRIVATE libgrad GTest::gtest GTest::gtest_main)
target_link_libraries(bias_test PRIVATE libbias GTest::gtest GTest::gtest_main)
target_link_libraries(weight_test PRIVATE libweight GTest::gtest GTest::gtest_main)
target_link_libraries(softplus_test PRIVATE libsoftplus GTest::gtest GTest::gtest_main)

# Discover tests
gtest_discover_tests(model_test)
# add_test(NAME SigmoidTest)
# add_test(NAME ModelTest)

# ctestの設定
add_test(NAME SigmoidTests COMMAND sigmoid_test)
add_test(NAME BinaryTests COMMAND binary_test)
add_test(NAME GradTests COMMAND grad_test)
add_test(NAME ModelTests COMMAND model_test)
add_test(NAME BiasTest COMMAND bias_test)
add_test(NAME WeightTest COMMAND weight_test)
add_test(NAME SoftplusTest COMMAND softplus_test)
