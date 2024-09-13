/**
* @file rbm.h 
* @brief RBMの重みのデータ型
* @author ankuru
* @date 2024/9/12
*
* @details RBMの重みのデータ型
*/

#ifndef WEIGHT_H
#define WEIGHT_H 

#include<Eigen/Dense>

namespace rbm_types{
  /*! @class Weight
    @brief  Weight(重み)の型定義
  */
  class Weight : public Eigen::MatrixXd {
    public:
        // デフォルトコンストラクタ
        Weight() : Eigen::MatrixXd() {}

        // パラメータ付きコンストラクタ
        Weight(int rows, int cols) : Eigen::MatrixXd(rows, cols) {}

        // コピーコンストラクタ
        Weight(const Eigen::MatrixXd& other) : Eigen::MatrixXd(other) {}

        // 代入演算子
        Weight& operator=(const Eigen::MatrixXd& other) {
            this->Eigen::MatrixXd::operator=(other);
            return *this;
        }
  };
}
#endif
