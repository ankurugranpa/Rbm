/**
* @file grad.h
* @brief 勾配のデータ型
* @author ankuru
* @date 2024/9/13
*
* @details 勾配のデータ型
*/
#ifndef GRAD_H
#define GRAD_H 

#include<Eigen/Dense>

namespace rbm_types{
  /*! @struct Data
    @brief  Dataの型定義
  */
  class Grad{
    public:
      Eigen::VectorXd visible_grad;
      Eigen::VectorXd hidden_grad;
      Eigen::MatrixXd weight_grad;
      Grad(int visible_dim, int hidden_dim);
      
      Grad operator+(const Grad& other) const;

      Grad operator-(const Grad& other) const;

      Grad operator*(const Grad& other) const;

      Grad operator/(const Grad& other) const;
  };
}
#endif
