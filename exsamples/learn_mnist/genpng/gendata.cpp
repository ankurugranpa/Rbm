#include "data.h"
#include<iostream>
#include<string>
#include <filesystem>
#include<random>
#include<fstream>
#include<tuple>

#include<boost/program_options.hpp>

#include<model.h>
#include<learn.h>
#include<csv.h>
#include<sampling.h>




using namespace rbm;
using namespace rbm_utils;
namespace po=boost::program_options;

int main(const int argc, const char* const * const argv){
  // 引き数の設定
  int EPOCH_NUM = 0,  SAMPLING_RATE = 1, NUM=1;
  std::string WORK_DIR = "./", TRAIN_DIR = "./", RESULT_DIR = "./";

  po::options_description description("実験設定");
  description.add_options()
      ("epoch,e", po::value<int>(), "エポック数")
      ("num,n", po::value<int>(), "数字の最大値")
      ("sampling_rate,r", po::value<int>()->default_value(10), "サンプリングレート")
      ("work_dir,w", po::value<std::string>()->default_value("./"), "読み込みデータ(パラメータ)のディレクトリパス")
      ("train_data_dir,t", po::value<std::string>()->default_value("./"), "テストデータのファイルパス")
      ("result_dir,e", po::value<std::string>()->default_value("./"), "結果のファイルパス")
      ("help,H", "ヘルプ");

  po::variables_map vm;
  store(po::parse_command_line(argc, argv, description), vm);

  if (vm.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }

  try {
    notify(vm);

    if (vm.count("epoch")) {
      EPOCH_NUM = vm["epoch"].as<int>();
    }

    if (vm.count("num")) {
      NUM = vm["num"].as<int>();
    }

    if (vm.count("sampling_rate")) {
      SAMPLING_RATE = vm["sampling_rate"].as<int>();
    }

    if (vm.count("work_dir")) {
      WORK_DIR = vm["work_dir"].as<std::string>();
    }

    if (vm.count("train_data_dir")) {
      TRAIN_DIR = vm["train_data_dir"].as<std::string>();
    }

    if (vm.count("result_dir")) {
      RESULT_DIR = vm["result_dir"].as<std::string>();
    }

    // デバッグ出力
    std::cout << "データ数 " << EPOCH_NUM << std::endl;
    std::cout << "SAMPLING_RATE: " << SAMPLING_RATE << std::endl;
    std::cout << "WORK_DIR: " << WORK_DIR << std::endl;
    std::cout << "TRAIN_DIR: " << TRAIN_DIR << std::endl;
    std::cout << "RESULT_DIR: " << RESULT_DIR << std::endl;

    // その他の処理...
    Csv csv(TRAIN_DIR);
    Csv out_csv(RESULT_DIR);
    File file_engine(WORK_DIR);
    Sampling sampler;



    for(int num = 0; num < NUM+1; num++) {
      DataSet data_set = csv.csv2dataset((std::to_string(num) + ".csv"));

      // std::ofstream data_file(WORK_DIR+ "/" + "learned_" + std::to_string(num) + ".csv");
      // std::ofstream original_data_file(WORK_DIR+ "/" + "original_" + std::to_string(num) + ".csv");

      std::vector<Bias> visible_set = file_engine.read_bias_file("visible_bias");
      std::vector<Bias> hidden_set = file_engine.read_bias_file("hidden_bias");
      std::vector<Weight> weight_set = file_engine.read_weight_file("weight_bias");

      std::vector<Parametar> parameter_set;
      for(auto i=0u; i<visible_set.size(); i++){
        Parametar buf;
        buf.visible_bias = visible_set[i];
        buf.hidden_bias = hidden_set[i];
        buf.weight=weight_set[i];
        parameter_set.push_back(buf);
      }


      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dist(1, data_set.size());
      int rand = dist(gen);
      // int rand = 0;

      // for(int i=0; i<data_set[rand].size(); i++){
      //   if(i != data_set[rand].size()-1){
      //    original_data_file << data_set[rand](i) << ",";
      //   }else{
      //    original_data_file << data_set[rand](i) << "\n";
      //   }
      // }
      out_csv.data2csv(data_set[rand], ("original_" + std::to_string(num) + ".csv"));

      for(int epoch = 0; epoch <EPOCH_NUM ; epoch++) {
        
        Model train_model(parameter_set[epoch]);
        DataSet buf_data_set;
        DataSet create_data_set;
        buf_data_set.push_back(data_set[rand]);
        std::tie(create_data_set, std::ignore) = sampler.block_gibbs_sampling(buf_data_set, train_model, 1);
        Data create_data = create_data_set[0];


        // for(int i=0; i<create_data.size(); i++){
        //   if(i != create_data.size()-1){
        //    data_file << create_data(i) << ",";
        //   }else{
        //    data_file << create_data(i) << "\n";
        //   }
        // }

        out_csv.data2csv(create_data, ("learned_" + std::to_string(num) + ".csv"));
      }
    }

  } catch (const boost::bad_any_cast& e) {
    std::cerr << "boost::bad_any_cast エラー: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "エラー: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

