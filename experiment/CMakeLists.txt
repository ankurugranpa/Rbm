cmake_minimum_required(VERSION 3.10)
add_compile_options(-std=c++17 -O2 -Wall)


# Main
add_executable(gibbs_test gibbssampling/gibbs_test.cpp)

target_link_libraries(gibbs_test
  PRIVATE libmodel
  PRIVATE libsampling
  PRIVATE libbinary
  PRIVATE libcsv)
