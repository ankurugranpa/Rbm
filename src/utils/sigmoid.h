#ifndef SIGMOID_H
#define SIGMOID_H 
#include<Eigen/Dense>

#include<bias.h>


namespace rbm_utils{
  /**
  * @file sigmoid.h 
  * @brief 
  * @author ankuru
  * @date 2024/9/11
  *
  * @details シグモイド関数util
  */

  /**
  *  @brief calc sigmoid 
  *  @param[in] x x
  *  @return double シグモイド関数の計算結果
  *  @details sig(x) = 1/(1+exp(-x)
  */
  double sig(double x);
  // Eigen::VectorXd Sig(Eigen::VectorXd x);

  /**
  *  @brief calc sigmoid 
  *  @param[in] x x
  *  @return double シグモイド関数の計算結果
  *  @details sig(x) = 1/(1+exp(-x)
  */
  rbm::Bias sig(rbm::Bias x);

  /**
  *  @brief calc sigmoid 
  *  @param[in] x x
  *  @return double シグモイド関数の計算結果
  *  @details sig(x) = 1/(1+exp(-x)
  */
  Eigen::VectorXd sig(Eigen::VectorXd x);
} 

#endif
