#include<fstream>
#include<sstream>

#include"csv.h"
using namespace rbm_utils;


Csv::Csv(){}

Csv::Csv(std::string work_dir){
  this->work_dir = work_dir;
};

Csv::~Csv(){};

void Csv::vector2csv(std::vector<double> vector, std::string file_name){
   std::ofstream data_file(work_dir+ "/" + file_name);
   for (auto &&item : vector) {
        data_file << item;
        data_file << '\n';
   }
}

void Csv::set_work_dir(std::string work_dir){
  this->work_dir = work_dir;
}

void Csv::dataset2csv(DataSet data_set, std::string file_name){
   std::ofstream data_file(work_dir+ "/" + file_name);
   for (auto &&item : data_set ) {
        data_file << item.transpose();
        data_file << '\n';
   }
}

DataSet Csv::csv2dataset(std::string file_name){
  DataSet result_data;
  std::ifstream file(work_dir+ "/" + file_name);
  std::string line;
  std::stringstream line_stream(line);

  if(!file.is_open()){
     throw std::invalid_argument( "Failed file open" );
  }
  while(std::getline(file, line)) {
    std::stringstream line_stream(line);
    std::string cell;
    std::vector<int> row;
    
    // カンマをデリミタとしてセルを読み込む
    while (std::getline(line_stream, cell, ',')) {
        row.push_back(std::stoi(cell));
    }

    Eigen::VectorXi buf_vec(row.size());
    for(int i=0; i<row.size(); i++){
      buf_vec(i) = row[i];
    }
    result_data.push_back(buf_vec);

  }
  file.close();

  return result_data;
}

void Csv::parametar2csv(Parametar parametar){
   std::ofstream visible_data_file(work_dir+ "/visible_bias.csv");
   visible_data_file << parametar.visible_bias;
   visible_data_file.close();

   std::ofstream hidden_data_file(work_dir+ "/hidden_bias.csv");
   hidden_data_file << parametar.hidden_bias;
   hidden_data_file.close();

   std::ofstream weight_data_file(work_dir+ "/weight_bias.csv");
   weight_data_file << parametar.weight;
   weight_data_file.close();
}

// Parametar Csv::csv2parametar(std::string file_name){
//   DataSet result_data;
//   std::ifstream file(work_dir+ "/" + file_name);
//   std::string line;
//   std::stringstream line_stream(line);
// 
//   if(!file.is_open()){
//      throw std::invalid_argument( "Failed file open" );
//   }
//   while(std::getline(file, line)) {
//     std::stringstream line_stream(line);
//     std::string cell;
//     std::vector<int> row;
//     
//     // カンマをデリミタとしてセルを読み込む
//     while (std::getline(line_stream, cell, ',')) {
//         row.push_back(std::stoi(cell));
//     }
// 
//     Eigen::VectorXi buf_vec(row.size());
//     for(int i=0; i<row.size(); i++){
//       buf_vec(i) = row[i];
//     }
//     result_data.push_back(buf_vec);
// 
//   }
//   file.close();
// 
//   return result_data;
// }
