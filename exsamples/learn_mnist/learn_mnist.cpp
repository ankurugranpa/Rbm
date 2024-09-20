#include<iostream>
#include<string>

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
  int VISIBLE_DIM, HIDDEN_DIM, EPOCH, SAMPLING_RATE, BATCH_SIZE;
  double NABLA;
  std::string WORK_DIR, DATA_FILE;
    
  po::options_description description("実験設定");
  description.add_options()
      ("v_dim,v", po::value<int>(), "可視変数の次元")
      ("h_dim,h", po::value<int>(), "隠れ変数の次元")
      ("nabla,n", po::value<double>(), "学習率")
      ("epoch,e", po::value<int>(), "epoch数")
      ("batch_size,b", po::value<int>(), "バッチサイズ")
      ("sampling_rate,r", po::value<int>()->default_value(10), "サンプリングレート")
      ("data_file,d", po::value<std::string>(), "観測データ")
      ("work_dir,w", po::value<std::string>()->default_value("test"), "作業ディレクトリ")

      ("help,H", "ヘルプ");

    po::variables_map vm;
    store(po::parse_command_line(argc, argv, description), vm);

    if (vm.count("help")) {
        std::cout << description << std::endl;
        std::exit(0);  // help表示後に終了
    }

    notify(vm);

    if (vm.count("v_dim")) {
      VISIBLE_DIM = vm["v_dim"].as<int>();
    }

    if (vm.count("h_dim")) {
        HIDDEN_DIM = vm["h_dim"].as<int>();
    }
    
    if (vm.count("nabla")) {
        NABLA = vm["nabla"].as<double>();
    }

    if (vm.count("epoch")) {
        EPOCH = vm["epoch"].as<int>();
    }

    if (vm.count("batch_size")) {
        BATCH_SIZE = vm["batch_size"].as<int>();
    }

    if (vm.count("sampling_rate")) {
        SAMPLING_RATE = vm["sampling_rate"].as<int>();
    }

    if (vm.count("work_dir")) {
        WORK_DIR = vm["work_dir"].as<std::string>();
    }
    if (vm.count("data_file")) {
      DATA_FILE = vm["data_file"].as<std::string>();
    }

  // Mnistのロード
  Csv csv;
  DataSet mnist_data = csv.csv2dataset(DATA_FILE);
  // DataSet mnist_one;
  // mnist_one.push_back(mnist_data[0]);
  

  // 実験設定
  int result_epoch, result_batch_time;
  Model model(VISIBLE_DIM, HIDDEN_DIM);
  Learn learn(WORK_DIR);
  File file_engine(WORK_DIR);
  std::cout << "データ数" <<  mnist_data.size() << std::endl;
  std::cout << "NABLA " << NABLA << "EPOCH " << EPOCH << "BATCH_SIZE" << BATCH_SIZE << "SAMPLING_RATE" << SAMPLING_RATE;
  learn.contrastive_divergence(model,
                               mnist_data,
                               NABLA,
                               EPOCH,
                               BATCH_SIZE,
                               "grad.csv",
                               SAMPLING_RATE,
                               &result_epoch,
                               &result_batch_time);
}

