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
      //! 可視変数の次元
      int visible_dim;
        
      //! 隠れ変数の次元
      int hidden_dim;
       
      //! パラメータ
      Parametar parametar;



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
      Eigen::VectorXd lambda_visible(const Parametar& parametar, const Eigen::VectorXi& rand_hidden) const;

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       */
      Eigen::VectorXd lambda_hidden(const Parametar& parametar, const Eigen::VectorXi& rand_visible) const;

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @param[in] rand_hidden 隠れ変数の確率変数
       */
      double cost_func(const Parametar& parametar, const Eigen::VectorXi& rand_visible, const Eigen::VectorXi& rand_hidden) const;

      /**
       *  @brief calc cost function 
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @return double calced cost func C_v(v;\theta) 
       *  @details C_v = - ( sum_i b_i x_i + sum_j log(1+e^lamdbda_hidden) 
       */
      double cost_v(const Parametar& parametar, const Eigen::VectorXi& rand_visible) const;

  };
}



#endif
