#ifndef FILE_H
#define FILE_H 
#include <filesystem>

#include<parametar.h>

namespace fs = std::filesystem;

namespace rbm_utils{
  /*! @class File
    @brief  Fileの操作
  */
  class File{
    public:
      fs::path work_dir;

      /**
      *  @brief 
      *  @param[in] bit_num ビット数
      */
      File();
      File(fs::path work_dir);
      ~File();

      /**
      *  @brief read_bias_file
      *  @param[in] parametar 保存するパラメーター
      *  @return bool 成功:失敗のステータス
      *  @details 数字をバイナリデータに変換する関数
      */
      std::vector<Bias> read_bias_file(fs::path file_name); 

      /**
      *  @brief read_weight_file
      *  @param[in] parametar 保存するパラメーター
      *  @return bool 成功:失敗のステータス
      *  @details 数字をバイナリデータに変換する関数
      */
      std::vector<Weight> read_weight_file(fs::path file_name); 

      /**
      *  @brief get_line_parametar
      *  @param[in] data_num 取得するファイル名
      *  @param[in] visible_file 取得するデータの場所(1つ目とか)
      *  @param[in] hidden_file 取得するデータの場所(1つ目とか)
      *  @param[in] weight_file 取得するデータの場所(1つ目とか)
      *  @return Parametar 取得したパラメーター
      *  @details ファイルからデータを1つ取り出す関数
      */
      Parametar get_line_parametar(int data_num,
                                   fs::path visible_file="visible_bias",
                                   fs::path hidden_file="hidden_bias",
                                   fs::path weight_file="weight_bias");

      /**
      *  @brief get_line_bias
      *  @param[in] file_name 取得するファイル名
      *  @param[in] data_num 取得するデータの場所(1つ目とか)
      *  @return Bias 取得したデータ
      *  @details ファイルからデータを1つ取り出す関数
      */
      Bias get_line_bias(fs::path file_name, int data_num);

      /**
      *  @brief get_line_weight
      *  @param[in] file_name 取得するファイル名
      *  @param[in] data_num 取得するデータの場所(1つ目とか)
      *  @return Weight 取得したデータ
      *  @details ファイルからデータを1つ取り出す関数
      */
      Weight get_line_weight(fs::path file_name, int data_num);
      

      /**
      *  @brief gen_file
      *  @param[in] parametar 保存するパラメーター
      *  @return bool 成功:失敗のステータス
      *  @details 可視変数, 隠れ変数, 重みのパラメーターの各ファイルを保存する
      */
      bool gen_file(Parametar parametar);

      /**
      *  @brief gen_file
      *  @param[in] parametar 保存するパラメーター
      *  @return bool 成功:失敗のステータス
      *  @details .biasファイルを生成する。(1,1)にベクトルの次元 すべてのデータはカンマ区切り, ファイルが存在すれば追加する
      */
      bool gen_file(Bias bias, fs::path file_name="bias.bias");

      /**
      *  @brief gen_file
      *  @param[in] parametar 保存するパラメーター
      *  @return bool 成功:失敗のステータス
      *  @details .biasファイルを生成する。(1,1)に行数、(1,2)に列数, すべてのデータはカンマ区切り
      */
      bool gen_file(Weight weight, fs::path file_name="weight.weight");

    private: 
      std::string extension_bias=".bias", extension_weight=".weight";
      /**
      *  @brief gen_path
      *  @param[in] dir ディレクトリ名
      *  @param[in] file_name ファイル名
      *  @return fs::path 絶対パス
      *  @details ディレクトリ名とファイル名を結合し絶対パスを作成する, 変な形の入力にも対応
      */
      fs::path absolute_path(fs::path dir, fs::path file_name);


      bool is_extension(const fs::path& file_path, const std::string& ext);
  };
}
#endif
