/**
* @file learn.h 
* @brief RBMの学習クラス
* @author ankuru
* @date 2024/9/11
*
* @details RBMクラス
*/

#ifndef LEARN_H
#define LEARN_H 

#include<model.h>
#include<grad.h>
#include<data.h>
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
      ~Learn();

      /**
       *  @brief 厳密計算で学習
       *  @param[in] before_learn_model 学習に使用するするモデル
       *  @param[in] data_set 学習に使用するデータセット
       *  @param[in] max_step step数の最大値(100)
       *  @param[in] result_file_name 学習過程の保存ファイル
       *  @param[in] result_step パラメーター更新の最終step数
       *  @details 厳密計算を使用した学習, データの次元が多いと学習が終わらないので注意
       */
      Model exact_calculation(const Model& before_learn_model,
                              const DataSet& data_set,
                              const int max_step=100,
                              const double nabla=0.5,
                              std::string result_file_name="learn_process_exact_calc.csv",
                              int* result_step = nullptr);
      void seg_epsilon(double value);
    private: 
      /**
       *  @brief 厳密計算で学習
       *  @param[in] model_object  パラメーター更新に使用するmodel
       *  @param[in] result_step パラメーター更新の最終step数
       *  @details 厳密計算を使用した学習, データの次元が多いと学習が終わらないので注意
       */
      Grad calc_data_mean(const Model& model_object, const DataSet& data_set);

      /**
       *  @brief 厳密計算で学習
       *  @param[in] model_object  パラメーター更新に使用するmodel
       *  @param[in] data_set パラメーター更新の最終step数
       *  @details 厳密計算を使用した学習, データの次元が多いと学習が終わらないので注意
       */
      Grad calc_model_mean(const Model& model_object, const DataSet& data_set);

      double calc_grad(const Grad& grad_data);

      bool is_zero(double value);
      

  };
}


#endif
