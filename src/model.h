/**
* @file model.h 
* @brief Modelクラス
* @author ankuru
* @date 2024/9/11
*
* @details RbmのModle定義
*/

#ifndef MODEL_H
#define MODEL_H
#include<Eigen/Dense>

#include<parametar.h>
#include<data.h>


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
       *  @return Model 定義されたモデル
       *  @details パラメータを自動で初期化する 
       */
      Model(int visible_dim, int hidden_dim);

      /**
       *  @brief Init Rbm
       *  @param[in] visible_dim 可視変数の次元
       *  @param[in] hidden_dim 隠れ変数の次元
       *  @param[in] parametar 初期パラメータ 
       *  @return Model 定義されたモデル
       *  @details 任意のパラメーターを初期値の演算が行われないため初期値を自分で設定する必要がある
       */
      Model(Parametar parametar);

      /**
       *  @brief デストラクタ
       */
      ~Model();

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_hidden 隠れ変数の確率変数
       *  @return Eigen::VectorXd lambda_visibleの値
       */
      virtual Eigen::VectorXd lambda_visible(const Parametar& parametar, const Eigen::VectorXi& rand_hidden) const;

      /**
       *  @brief Init Rbm
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @return Eigen::VectorXd lambda_hiddenの値
       */
      virtual Eigen::VectorXd lambda_hidden(const Parametar& parametar, const Eigen::VectorXi& rand_visible) const;

      /**
       *  @brief Rbmのコスト関数
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @param[in] rand_hidden 隠れ変数の確率変数
       *  @return double Rbmのエネルギー
       */
      double cost_func(const Parametar& parametar, const Eigen::VectorXi& rand_visible, const Eigen::VectorXi& rand_hidden) const;

      /**
       *  @brief calc Rbmのコスト関数を周辺化したもの
       *  @param[in] parametar パラメーター
       *  @param[in] rand_visible 可視変数の確率変数
       *  @return double calced cost func C_v(v;\theta) 
       *  @details C_v = - ( sum_i b_i x_i + sum_j log(1+e^lamdbda_hidden) 
       */
      double cost_v(const Parametar& parametar, const Eigen::VectorXi& rand_visible) const;


      /**
       *  @brief パラメータを再設定
       *  @param[in] parametar パラメーター
       */
      void set_parameter(Parametar parametar);

      /**
       *  @brief KLダイバージェンス
       *  @param[in] model 比較対象のモデル
       *  @return double kl距離を返す
       *  @return double klダイバージェンス
       *  @details 自分と引き数とのKL距離を算出する
       */
      double kl_divergence(const Model& model);

      /**
       *  @brief 分布
       *  @return std::vector<double> 各状態の分布
       *  @details 各状態の分布を算出する関数
       */
      std::vector<double> distribution();

      /**
       *  @brief すべての状態
       *  @return DataSet 状態
       *  @details モデルがとりうるすべての状態を算出する
       */
      DataSet all_status();
  };
}



#endif
