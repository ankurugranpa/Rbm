#include<iostream>
#include<vector>

#include<boost/program_options.hpp>

#include<model.h>
#include<csv.h>
#include<binary.h>
#include<sampling.h>
#include<learn.h>
#include<file.h>

namespace po=boost::program_options;

int main(const int argc, const char* const * const argv){
  // 引き数の設定
  int VISIBLE_DIM=4, HIDDEN_DIM, STEP, SAMPLING_RATE, DATA_NUM;
  double NABLA;
  std::string WORK_DIR, RESULT_FILE;
    
  po::options_description description("実験設定");
  description.add_options()
      ("data_num,d", po::value<int>(), "可視変数の次元")
      ("h_dim,h", po::value<int>(), "隠れ変数の次元")
      ("nabla,n", po::value<double>(), "学習率")
      ("step,s", po::value<int>(), "epoch数")
      ("sampling_rate,r", po::value<int>()->default_value(10), "サンプリングレート")
      ("work_dir,w", po::value<std::string>()->default_value("test"), "作業ディレクトリ")
      ("result_file,f", po::value<std::string>(), "結果のファイル名")

      ("help,H", "ヘルプ");

    po::variables_map vm;
    store(po::parse_command_line(argc, argv, description), vm);

    if (vm.count("help")) {
        std::cout << description << std::endl;
        std::exit(0);  // help表示後に終了
    }

    notify(vm);

    if (vm.count("data_num")) {
        DATA_NUM = vm["data_num"].as<int>();
    }

    if (vm.count("h_dim")) {
        HIDDEN_DIM = vm["h_dim"].as<int>();
    }
    
    if (vm.count("nabla")) {
        NABLA = vm["nabla"].as<double>();
    }

    if (vm.count("step")) {
        STEP = vm["step"].as<int>();
    }

    if (vm.count("sampling_rate")) {
        SAMPLING_RATE = vm["sampling_rate"].as<int>();
    }

    if (vm.count("work_dir")) {
        WORK_DIR = vm["work_dir"].as<std::string>();
    }

    if (vm.count("result_file")) {
        RESULT_FILE = vm["result_file"].as<std::string>();
    }

  rbm::Model befor_model(VISIBLE_DIM, HIDDEN_DIM);
  DataSet train_data_set;
  rbm_utils::File result_file(WORK_DIR);
  std::string setting = "可視変数の次元:" + std::to_string(VISIBLE_DIM) + "\n"+ "隠れ変数の次元:" + std::to_string(HIDDEN_DIM) + 
    "\n" + "学習データ数:" + std::to_string(DATA_NUM) + "\n" + "ステップ数:" + std::to_string(STEP) + "\n" +
    "学習率:" + std::to_string(NABLA) + "\n" + "サンプリングレート:" + std::to_string(SAMPLING_RATE) ;
  result_file.gen_file(setting, RESULT_FILE);
  
  // すべての状態を準備
  rbm_utils::Csv csv(WORK_DIR);
  rbm_utils::Binary calc_binary(VISIBLE_DIM);

  std::cout << "サンプリングの開始" << std::endl;
  rbm::Sampling sampler;
  train_data_set = sampler.create_data_set(befor_model, DATA_NUM, SAMPLING_RATE); // 学習に使用するデータセット
  std::cout << "サンプリング終了" << std::endl;
  csv.dataset2csv(train_data_set, "sampling_data.csv");




  DataSet all_status;


  for(int i=0; i<std::pow(2, VISIBLE_DIM); i++){
    all_status.push_back(calc_binary.num2binary(i));
  }
  csv.dataset2csv(all_status, "status.csv");


  
  std::cout << "学習開始" << std::endl;

  // 学習
  rbm::Learn learner(WORK_DIR);
  rbm::Model after_model(befor_model.visible_dim, befor_model.hidden_dim);
  rbm_utils::File read_test(WORK_DIR);
  Parametar test_parameter(after_model.visible_dim, after_model.hidden_dim);
  
  int buf;
  rbm::Model buf_afuter_model = after_model;
  learner.exact_calculation(after_model, train_data_set, STEP, NABLA, "procece.csv", &buf);
  std::cout << "学習終了" << std::endl;


  test_parameter.visible_bias =  read_test.read_bias_file("visible_bias.bias")[999];
  test_parameter.hidden_bias = read_test.read_bias_file("hidden_bias.bias")[999];
  test_parameter.weight = read_test.read_weight_file("weight_bias.weight")[999];

  after_model.set_parameter(test_parameter);

  // 学習前の分布の生成
  std::vector<double> p_v_list(std::pow(2, VISIBLE_DIM)); // 各状態毎の確率の保存list
  double z=0;

  int i=0;
  for(const auto& item: all_status){
    double p_v;
    p_v = std::exp(-befor_model.cost_v(befor_model.parametar, item));
    // std::cout << "p_v:" <<p_v << std::endl;
    p_v_list[i] = p_v;
    i++;
    z += p_v;
  }
  for(auto& item:p_v_list){
    item = item/z;
  }
  csv.vector2csv(p_v_list, "befor_p_v_distribution.csv");
  
  //
  // 学習後の分布の生成
  std::vector<double> p_v_list_a(std::pow(2, VISIBLE_DIM)); // 各状態毎の確率の保存list
  double z_a=0;

  int i_a=0;
  for(const auto& item: all_status){
    double p_v;
    p_v = std::exp(-after_model.cost_v(after_model.parametar, item));
    // std::cout << "p_v:" << p_v << std::endl;
    p_v_list_a[i_a] = p_v;
    i_a++;
    z_a += p_v;
  }
  for(auto& item:p_v_list_a){
    item = item/z_a;
    // std::cout << item << std::endl;
  }
  // std::cout << "p_v:" << p_v << std::endl;
  csv.vector2csv(p_v_list_a, "after_p_v_distribution.csv");

// bufの生成
  std::vector<double> p_v_list_b(std::pow(2,VISIBLE_DIM)); // 各状態毎の確率の保存list
  double z_b=0;

  int i_b=0;
  for(const auto& item: all_status){
    double p_v;
    p_v = std::exp(-buf_afuter_model.cost_v(buf_afuter_model.parametar, item));
    // std::cout << "p_v:" <<p_v << std::endl;
    p_v_list_b[i_b] = p_v;
    i_b++;
    z_b += p_v;
  }
  for(auto& item:p_v_list_b){
    item = item/z_b;
  }
  csv.vector2csv(p_v_list_b, "after_buf_p_v_distribution.csv");
  double befor_kl = befor_model.kl_divergence(buf_afuter_model);
  double after_kl = befor_model.kl_divergence(after_model);


  std::string kl_d = ("befor_kld:" + std::to_string(befor_kl) + "\n" + "after_kld:" + std::to_string(after_kl));
  result_file.gen_file(kl_d, RESULT_FILE);

  // result_file.gen_file(

  std::cout << "befor_kl" << std::endl;
  std::cout << befor_kl << std::endl;
  std::cout << "after_kl" << std::endl;
  std::cout << after_kl << std::endl;
  std::cout << "end" << std::endl;
}
