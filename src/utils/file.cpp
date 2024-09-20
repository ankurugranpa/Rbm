#include<iostream>
#include<fstream>

#include"file.h"

using namespace rbm_utils;


File::File(){};
File::~File(){};

std::vector<Bias> File::read_bias_file(fs::path file_name){
  std::vector<Bias> bias_set;
  fs::path full_path = absolute_path(work_dir, file_name);
  if (!is_extension(full_path, extension_bias)) full_path.replace_extension(extension_bias);

  if (!fs::exists(full_path)) {
      std::cerr << "ファイルが存在しません: " << full_path << std::endl;
      throw std::runtime_error("File not found");
  }
  
  // ファイルを開く
  std::ifstream file(full_path);
  if (!file.is_open()) {
      std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
      throw std::runtime_error("Unable to open file");
  }
  
  // 1行目から1行あたりの次元数を取得
  std::string first_line;
  std::getline(file, first_line);
  std::istringstream first_line_stream(first_line);
  size_t dimension_count;
  first_line_stream >> dimension_count;

  // Bias オブジェクトを作成
  Bias bias_data(static_cast<Eigen::Index>(dimension_count));

  // bias_num + 1 行目のデータを取得

  std::string line;

  while (std::getline(file, line)) {
       std::istringstream line_stream(line);
       std::string value;
       std::vector<double> values;
       Bias bias_data(dimension_count);

       while (std::getline(line_stream, value, ',')) {
           if (!value.empty()) {
               values.push_back(std::stod(value));
           }
       }
       if (values.size() ==  dimension_count) {
           for(int i=0; i<dimension_count; i++){
             bias_data(i) = values[i];
           }
       } else {
           std::cerr << "警告: 無効なデータ行をスキップします。期待される要素数: " 
                     << dimension_count << ", 実際の要素数: " << values.size() << std::endl;
       }
       bias_set.push_back(bias_data);
  }


  
  

  file.close();
  return bias_set;
}

std::vector<Weight> File::read_weight_file(fs::path file_name) {
    std::vector<Weight> weight_set;
    fs::path full_path = absolute_path(work_dir, file_name);
    if (!is_extension(full_path, extension_weight)) full_path.replace_extension(extension_weight);

    if (!fs::exists(full_path)) {
        std::cerr << "ファイルが存在しません: " << full_path << std::endl;
        throw std::runtime_error("File not found");
    }

    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    std::getline(file, line);
    std::istringstream dimensions_stream(line);
    size_t rows, cols;
    char comma;
    dimensions_stream >> rows >> comma >> cols;

    while (std::getline(file, line)) {
        if (line.empty()) continue; // 空行をスキップ

        std::vector<double> values;
        std::istringstream line_stream(line);
        std::string value;
        while (std::getline(line_stream, value, ',')) {
            if (!value.empty()) {
                values.push_back(std::stod(value));
            }
        }

        if (values.size() == rows * cols) {
            Weight weight_data(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    weight_data(i, j) = values[i * cols + j];
                }
            }
            weight_set.push_back(weight_data);
        } else {
            std::cerr << "警告: 無効なデータ行をスキップします。期待される要素数: " 
                      << rows * cols << ", 実際の要素数: " << values.size() << std::endl;
        }
    }

    file.close();
    return weight_set;
}


File::File(fs::path work_dir){
  this->work_dir  = work_dir;
  // ディレクトリが存在しなければ作成
  if (!fs::exists(work_dir)) { fs::create_directory(work_dir); }
};


Parametar File::get_line_parametar(int data_num,
                                   fs::path visible_file,
                                   fs::path hidden_file,
                                   fs::path weight_file){
  Parametar result;
  result.visible_bias = get_line_bias(visible_file, data_num);
  result.hidden_bias = get_line_bias(hidden_file, data_num);
  result.weight = get_line_weight(weight_file, data_num);

  return result;
}

Bias File::get_line_bias(fs::path file_name, int data_num){
  fs::path full_path = absolute_path(work_dir, file_name);
    if (!is_extension(full_path, extension_bias)) {
        full_path.replace_extension(extension_bias);
    }

    if (!fs::exists(full_path)) {
        std::cerr << "ファイルが存在しません: " << full_path << std::endl;
        throw std::runtime_error("File not found");
    }

    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
        throw std::runtime_error("Unable to open file");
    }

    // 1行目に次元数が書かれている
    std::string line;
    std::getline(file, line);
    std::istringstream dimensions_stream(line);
    size_t dimension_count;
    dimensions_stream >> dimension_count;

    // 指定された行番号までスキップ
    int current_line = 0;
    while (current_line < data_num && std::getline(file, line)) {
        current_line++;
    }

    // 指定された行がファイル内に存在しない場合
    if (current_line != data_num || line.empty()) {
        std::cerr << "指定された行番号がファイル内に存在しません: " << data_num << std::endl;
        throw std::out_of_range("Line number out of range");
    }

    // データを読み取る
    Bias bias_data(static_cast<Eigen::Index>(dimension_count));
    std::istringstream line_stream(line);
    std::string value;
    size_t index = 0;
    while (std::getline(line_stream, value, ',')) {
        if (!value.empty() && index < dimension_count) {
            bias_data(index) = std::stoi(value);
            ++index;
        }
    }

    if (index != dimension_count) {
        std::cerr << "警告: 行データのサイズが期待されたものと異なります。" << std::endl;
        throw std::runtime_error("Invalid data size");
    }

    file.close();
    return bias_data;
}
Weight File::get_line_weight(fs::path file_name, int data_num){

    std::vector<Weight> weight_set;
    fs::path full_path = absolute_path(work_dir, file_name);
    if (!is_extension(full_path, extension_weight)) full_path.replace_extension(extension_weight);

    if (!fs::exists(full_path)) {
        std::cerr << "ファイルが存在しません: " << full_path << std::endl;
        throw std::runtime_error("File not found");
    }

    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    std::getline(file, line);
    std::istringstream dimensions_stream(line);
    size_t rows, cols;
    char comma;
    dimensions_stream >> rows >> comma >> cols;

    // 指定された行番号までスキップ
    int current_line = 0;
    while (current_line < data_num && std::getline(file, line)) {
        current_line++;
    }

    // 指定された行がファイル内に存在しない場合
    if (current_line != data_num || line.empty()) {
        std::cerr << "指定された行番号がファイル内に存在しません: " << data_num << std::endl;
        throw std::out_of_range("Line number out of range");
    }
    // データを読み取る
    std::vector<double> values;
    std::istringstream line_stream(line);
    std::string value;
    while (std::getline(line_stream, value, ',')) {
        if (!value.empty()) {
            values.push_back(std::stod(value));
        }
    }

    Weight weight_data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            weight_data(i, j) = values[i * cols + j];
        }
    }

    file.close();
    return weight_data;
}

bool File::gen_file(Parametar parametar){
  if(!gen_file(parametar.visible_bias, "visible_bias")) return false;
  if(!gen_file(parametar.hidden_bias, "hidden_bias")) return false;
  if(!gen_file(parametar.weight, "weight_bias")) return false;
  return true;
}

bool File::gen_file(Bias bias, fs::path file_name){
  fs::path full_path = absolute_path(work_dir, file_name);
  full_path.replace_extension(extension_bias);
  
  bool file_exists = fs::exists(full_path);
  std::ofstream file_object;
  
  // ファイルが存在する場合は読み込みモード、存在しない場合は作成モードで開く
  if (file_exists) {
      file_object.open(full_path, std::ios::app);
  } else {
      file_object.open(full_path, std::ios::out);
      if (!file_object.is_open()) {
          std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
          return false;
      }
      // ファイルの1行目に1行当たりの文字数を書き込む
      file_object << bias.size() << "\n";
  }
  
  if (!file_object.is_open()) {
      std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
      return false;
  }

  // ファイルの末尾にデータを書き込む
  for (int i = 0; i < bias.size(); ++i) {
      if (i > 0) {
          file_object << ",";  // カンマで区切る
      }
      file_object << bias(i);  // 要素を書き込む
  }
  file_object << "\n";  // 行末の改行

  if (file_object.fail()) {
      std::cerr << "データの書き込みに失敗しました: " << full_path << std::endl;
      return false;
  }
  
  return true;
}

bool File::gen_file(Weight weight, fs::path file_name){
  fs::path full_path = absolute_path(work_dir, file_name);
  full_path.replace_extension(extension_weight);

  bool file_exists = fs::exists(full_path);
  std::ofstream file_object;

  // ファイルが存在する場合は追記モード、存在しない場合は作成モードで開く
  if (file_exists) {
      file_object.open(full_path, std::ios::app);
  } else {
      file_object.open(full_path, std::ios::out);
      if (!file_object.is_open()) {
          std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
          return false;
      }
      // ファイルの1行目に行数と列数を書き込む
      file_object << weight.rows() << "," << weight.cols() << "\n";
  }

  if (!file_object.is_open()) {
      std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
      return false;
  }

  // weight をカンマ区切りで書き込む
  for (int i = 0; i < weight.rows(); i++) {
      for (int j = 0; j < weight.cols(); j++) {
          if (i > 0 || j > 0) {
              file_object << ",";  // カンマで区切る
          }
          file_object << weight(i, j);  // 要素を書き込む
      }
  }
  file_object << "\n";  // 行末の改行

  if (file_object.fail()) {
      std::cerr << "データの書き込みに失敗しました: " << full_path << std::endl;
      return false;
  }

  return true;
}

bool File::gen_file(std::string text_data , fs::path file_name){
  fs::path full_path = absolute_path(work_dir, file_name);
  full_path.replace_extension(".txt");

  bool file_exists = fs::exists(full_path);
  std::ofstream file_object;

  if (file_exists) {
      file_object.open(full_path, std::ios::app);
  } else {
      file_object.open(full_path, std::ios::out);
      if (!file_object.is_open()) {
          std::cerr << "ファイルを開けませんでした: " << full_path << std::endl;
          return false;
      }
  }

  file_object << text_data << "\n";

  if (file_object.fail()) {
      std::cerr << "データの書き込みに失敗しました: " << full_path << std::endl;
      return false;
  }
  return true;
}

fs::path File::absolute_path(fs::path dir, fs::path file_name){
    return fs::absolute(dir / file_name);
}

bool File::is_extension(const fs::path& file_path, const std::string& ext){
  return file_path.extension() == ext;
}
