/**
* @file csv.h 
* @brief csv ユーティリティ
* @author ankuru
* @date 2024/9/13
*
* @details csv ユーティリティ
 
*/

#ifndef CSV_H
#define CSV_H
#include<vector>

#include<data.h>
#include<parametar.h>

namespace rbm_utils{
  class Csv{
    public:
      std::string work_dir;
      Csv();
      Csv(std::string work_dir);
      ~Csv();

      void set_work_dir(std::string work_dir);

      /**
       *  @brief vector2csv 
       *  @param[in] vector 保存するベクトル
       *  @param[in] file_name ファイル名
       *  @details データ集合をcsvファイルに出力するツール
       */
      void vector2csv(std::vector<double> vector, std::string file_name);

      /**
       *  @brief dataset2csv
       *  @param[in] data_set 保存するデータ集合
       *  @param[in] file_name ファイル名
       *  @details データ集合をcsvファイルに出力するツール
       */
      void dataset2csv(rbm::DataSet data_set, std::string file_name);

      /**
       *  @brief csv2dataset
       *  @param[in] file_name ファイルの名前
       *  @return rbm::DataSet バイナリのデータセット.csv
       *  @details csvファイルからDataSet型のデータ集合を構築するツール
       *  */
      rbm::DataSet csv2dataset(std::string file_name);

      /**
       *  @brief parametar2csv
       *  @param[in] file_name ファイルを保存するディレクトリ
       *  @return rbm::DataSet バイナリのデータセット.csv
       *  @details パラメータを visible_bias.csv, hidden_bias.csv, weight.csvにそれぞれ保存するツール
       */
      void parametar2csv(rbm::Parametar parametar);

      /**
       *  @brief csv2parametar
       *  @param[in] file_name 読み込むファイル名
       *  @details csvファイルからパラメータを作成するツール
       */
      rbm::Parametar csv2parametar(std::string file_name);

  };
}
#endif
