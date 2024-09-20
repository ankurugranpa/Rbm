/**
* @file sigmoid.h 
* @brief sigmoid関数
* @author ankuru
* @date 2024/9/13
* @details sigmoid ユーティリティ
*/
#ifndef SIGMOID_H
#define SIGMOID_H 
#include<Eigen/Dense>

#include<bias.h>
using namespace rbm_types;


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
  Bias sig(Bias x);

  /**
  *  @brief calc sigmoid 
  *  @param[in] x x
  *  @return double シグモイド関数の計算結果
  *  @details sig(x) = 1/(1+exp(-x)
  */
  Eigen::VectorXd sig(Eigen::VectorXd x);
} 

#endif
