/**
* @file csv.h 
* @brief csv ユーティリティ
* @author ankuru
* @date 2024/9/13
* @details csv ユーティリティ
*/

#ifndef CSV_H
#define CSV_H
#include<vector>

#include<data.h>
#include<parametar.h>
using namespace rbm_types;

namespace rbm_utils{
  /*! @class csv
    @brief  csvユーティリティ
  */
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
      void dataset2csv(DataSet data_set, std::string file_name);

      /**
       *  @brief data2csv
       *  @param[in] data 保存するデータ
       *  @param[in] file_name ファイル名
       *  @details データをcsvファイルに出力するツール
       */
      void data2csv(Data data, std::string file_name);

      /**
       *  @brief csv2dataset
       *  @param[in] file_name ファイルの名前
       *  @return DataSet バイナリのデータセット.csv
       *  @details csvファイルからDataSet型のデータ集合を構築するツール
       *  */
      DataSet csv2dataset(std::string file_name);

      /**
       *  @brief parametar2csv
       *  @param[in] file_name ファイルを保存するディレクトリ
       *  @return DataSet バイナリのデータセット.csv
       *  @details パラメータを visible_bias.csv, hidden_bias.csv, weight.csvにそれぞれ保存するツール
       */
      void parametar2csv(Parametar parametar);

      /**
       *  @brief csv2parametar
       *  @param[in] file_name 読み込むファイル名
       *  @details csvファイルからパラメータを作成するツール
       */
      Parametar csv2parametar(std::string file_name);

  };
}
#endif
