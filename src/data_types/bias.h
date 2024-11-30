/**
* @file bias.h 
* @brief RBMのバイアスのデータ型
* @author ankuru
* @date 2024/9/12
*
* @details RBMのバイアスのデータ型
*/
#ifndef BIAS_H
#define BIAS_H 
#include"data.h"

#include<Eigen/Dense>

namespace rbm_types{
  /*! @class Bias
    @brief  Biasの型定義
  */
  class Bias : public Eigen::VectorXd {
    public:
        Bias() : Eigen::VectorXd() {}
        Bias(int size) : Eigen::VectorXd(size) {}

        // コピーコンストラクタと代入演算子の定義
        Bias(const Eigen::VectorXd& other) : Eigen::VectorXd(other) {}
        Bias& operator=(const Eigen::VectorXd& other) {
            this->Eigen::VectorXd::operator=(other);
            return *this;
        }

        double dot(const Eigen::VectorXd& other) const {
            return Eigen::VectorXd::dot(other);
        }

        double dot(const Eigen::VectorXi& other) const {
            return Eigen::VectorXd::dot(other.cast<double>());
        }

        double dot(const Data& other) const {
            return Eigen::VectorXd::dot(other.cast<double>());
        }
        
        // block()メソッドをラップするメソッドを追加
        Bias block(int start, int size) const {
            return Bias(Eigen::VectorXd::block(start, 0, size, 1));
        }
  };
}
#endif
