/**
* @file rbm.h 
* @brief RBMクラス
* @author ankuru
* @date 2024/9/11
*
* @details RBMクラス
*/

#ifndef RBM_H
#define RBM_H
#include<Eigen/Dense>

#include<parametar.h>




namespace rbm{

  /*! @class Model
    @brief  Modelの型定義
  */
  class Model{
    public:
      int visible_dim, hidden_dim;
      // Parametar parametar;

      Model();
      /**
       *  @brief Init Rbm
       *  @param[in] visible_dim 可視変数の次元
       *  @param[in] hidden_dim 隠れ変数の次元
       */
      Model(int visible_dim, int hidden_dim);
      ~Model();

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_hidden 隠れ変数の確率変数
       */
      Eigen::VectorXd lambda_visible(Parametar parametar, Eigen::VectorXi rand_hidden);

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       */
      Eigen::VectorXd lambda_hidden(Parametar parametar, Eigen::VectorXi rand_visible);

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @param[in] rand_hidden 隠れ変数の確率変数
       */
      double cost_func(Parametar parametar, Eigen::VectorXi rand_visible, Eigen::VectorXi rand_hidden);
  };

}



#endif
