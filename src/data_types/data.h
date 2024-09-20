/**
* @file data.h 
* @brief バイナリのRBMで扱うデータ型
* @author ankuru
* @date 2024/9/12
*
* @details RBMのバイアスのデータ型
*/
#ifndef DATA_H
#define DATA_H 
#include<vector>

#include<Eigen/Dense>

namespace rbm_types{
  /*! @class Data 
    @brief  Data型定義
  */
  class Data : public Eigen::VectorXi {
    public:
        Data() : Eigen::VectorXi() {}
        Data(int size) : Eigen::VectorXi(size) {}

        // コピーコンストラクタと代入演算子の定義
        Data(const Eigen::VectorXi& other) : Eigen::VectorXi(other) {}
        Data& operator=(const Eigen::VectorXi& other) {
            this->Eigen::VectorXi::operator=(other);
            return *this;
        }
  };

  /*! @struct Data
    @brief  Dataの型定義
  */
  struct DataSet : public std::vector<Data> {
      using std::vector<Data>::vector;
  };
}
#endif
