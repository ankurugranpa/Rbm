#ifndef SAMPLING_H
#define SAMPLING_H

#include<tuple>

#include<data_types/data.h>
#include<model.h>

namespace rbm{

  /*! @class Sampling
    @brief  Samplingを行うクラス
  */
  class Sampling{
    public:
      Sampling();
      ~Sampling();

      /**
       *  @brief  ブロック化ギブスサンプリング
       *  @param[in] observation_data 観測データ
       *  @param[in] model_object RBMのモデルデータ
       *  @param[in] data_num データの数
       *  @param[in] sampling_rate サンプリング周期(CK-k法のkのような奴)
       */
      std::tuple<DataSet, DataSet> block_gibbs_sampling(const DataSet& observation_data, const rbm::Model& model_object, int sampling_rate);

      /**
       *  @brief バイナリデータセットの作成
       *  @param[in] model_object モデルデータ
       *  @param[in] data_num データの数
       *  @param[in] sampling_rate サンプリングの周期
       */
      DataSet create_data_set(const rbm::Model& model_object,int data_num, int sampling_rate);

      /**
       *  @brief バイナリデータセットの作成
       *  @param[in] model_object モデルデータ
       *  @param[in] data_num データの数
       *  @param[in] sampling_rate サンプリングの周期
       */
      Data create_data(const rbm::Model& model_object, Data base_data, int sampling_rate);
  };
}

#endif
