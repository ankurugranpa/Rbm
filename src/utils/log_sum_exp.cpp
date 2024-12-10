#include<iostream>
#include <algorithm>
#include <cmath>

#include "log_sum_exp.h"

double rbm_utils::log_sum_exp(std::vector<double> vec_a){
  auto a_max = std::max_element(vec_a.begin(), vec_a.end());
  // std::cout << "a_max:" <<*a_max << std::endl;

  double exp_a_i_sum=0;
  for(const auto& a_i :vec_a){
    exp_a_i_sum += exp(a_i - *a_max);
  }

  return *a_max + std::log(exp_a_i_sum);
}
