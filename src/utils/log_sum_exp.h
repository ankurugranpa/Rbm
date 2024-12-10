/**
* @file log_sum_exp.h 
* @brief log_sum_exp関数
* @author ankuru
* @date 2024/12/09
* @details log sum expの計算 ユーティリティ
*/
#ifndef LOG_SUM_EXP_H
#define LOG_SUM_EXP_H 

#include <vector>


namespace rbm_utils{

  /**
  *  @brief calc log_sum_exp
  *  @param[in] vec_a a_i..N
  *  @return double log_sum_expの計算結果
  */
  double log_sum_exp(std::vector<double> vec_a);
} 

#endif
