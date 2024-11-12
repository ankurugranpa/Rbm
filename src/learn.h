/**
* @file learn.h 
* @brief Learn クラス
* * @author ankuru
* @date 2024/9/11
*
* @details RBMの学習の定義
*/

#ifndef LEARN_H
#define LEARN_H 

#include<model.h>
#include<grad.h>
#include<data.h>
#include<file.h>
#include<sampling.h>
using namespace rbm_types;

namespace rbm{
  /*! @class Learn
    @brief  rbmの学習
  */
  class Learn{
    public:
      //! 学習結果や過程の保存先
      std::string work_dir;
      double epsilon = 1e-10;

      Learn();
      
      /**
       *  @brief Rbm Learn
       *  @param[in] work_dir 学習結果などを保存するディレクトリ 
       */
      Learn(std::string work_dir);

      /**
       *  @brief デストラクタ
       */
      ~Learn();

      /**
       *  @brief 厳密計算で学習
       *  @param[in] before_learn_model 学習に使用するするモデル
       *  @param[in] data_set 学習に使用するデータセット
       *  @param[in] max_step step数の最大値(100)
       *  @param[in] result_file_name 学習過程の保存ファイル
       *  @param[in] result_step パラメーター更新の最終step数
       *  @return Model 学習後のモデル
       *  @details 厳密計算を使用した学習, データの次元が多いと学習が終わらないので注意
       */
      Model exact_calculation(const Model& before_learn_model,
                              const DataSet& data_set,
                              const int max_step=100,
                              const double nabla=0.5,
                              std::string result_file_name="learn_process_exact_calc.csv",
                              int* result_step=nullptr);

      /**
       *  @brief Contrastive Divergence(CD法)
       *  @param[in] before_learn_model 学習に使用するするモデル
       *  @param[in] data_set 学習に使用するデータセット
       *  @param[in] max_step step数の最大値(100)
       *  @param[in] result_file_name 学習過程の保存ファイル
       *  @param[in] result_step パラメーター更新の最終step数
       *  @return Model 学習後のモデル
       *  @details CD法を使用した学習, ミニバッチ学習を行う
       */
      Model contrastive_divergence(const Model& before_learn_model,
                                   const DataSet& data_set,
                                   const double nabla=0.5,
                                   const int epoch=100,
                                   const int batch_size=100,
                                   std::string result_file_name = "learn_process_exact_calc.csv",
                                   int cd_k_num=1,
                                   int* result_epoch=nullptr,
                                   int* result_batch_time=nullptr);

      void set_epsilon(double value);

      /**
       *  @brief データ平均
       *  @param[in] model_object  パラメーター更新に使用するmodel
       *  @param[in] data_set 観測データ
       *  @details データ平均を計算する
       *  @return Grad データ平均
       */
      Grad calc_data_mean(const Model& model_object, const DataSet& data_set);

      /**
       *  @brief 厳密計算でのモデル平均
       *  @param[in] model_object  パラメーター更新に使用するmodel
       *  @param[in] data_set パラメーター更新の最終step数
       *  @return Grad モデル平均
       *  @details 厳密計算を使用したモデル平均の計算を行う, データの次元が多いと学習が終わらないので注意
       */
      Grad calc_model_mean(const Model& model_object);

      /**
       *  @brief Contrastive Divergence(CD法)でのモデル平均
       *  @param[in] model_object  パラメーター更新に使用するmodel
       *  @param[in] data_set パラメーター更新の最終step数
       *  @return Grad モデル平均
       *  @details cd法を使用したモデル平均の計算を行う
       */
      Grad calc_model_mean_cd(const Model& model_object, const DataSet& data_set, int sampring_rate=1);

      /**
       *  @brief 勾配計算
       *  @param[in] grad_data 計算する勾配の入力
       *  @details 各パラメータから勾配を計算する
       *  @return double 勾配
       */
      double calc_grad(const Grad& grad_data);

      /**
       *  @brief ゼロ判定
       *  @param[in] grad_data 計算する勾配の入力
       *  @return bool ゼロかどうか
       *  @details 各パラメータから勾配を計算する
       */
      bool is_zero(double value);
  };
}


#endif
