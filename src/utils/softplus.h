/**
* @file softplus.h 
* @brief softplus関数
* @author ankuru
* @date 2024/12/07
* @details sigmoid ユーティリティ
*/
#ifndef SOFTPLUS_H
#define SOFTPLUS_H 


namespace rbm_utils{

  /**
  *  @brief calc softplus
  *  @param[in] x x
  *  @return double  ソフトプラス関数の計算結果
  */
  double softplus(double x);
} 

#endif
