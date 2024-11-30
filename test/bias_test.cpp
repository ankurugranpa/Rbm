#include<gtest/gtest.h>
#include<iostream>

#include<bias.h>
#include<data.h>


namespace {
  class BiasTest: public testing::Test{
    protected:
      rbm_types::Bias bias_test;
      BiasTest() : bias_test(17) {} // コンストラクタで初期化
  };
};



TEST_F(BiasTest, block){
  bias_test << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17;
  rbm_types::Bias seikai(4);
  seikai << 14,15,16,17;
  rbm_types::Bias buf_bias = bias_test.block(13, 4);
  // std::cout << buf_bias.transpose() << std::endl;
  ASSERT_EQ(seikai, buf_bias);
}

TEST_F(BiasTest, dot_data){
  rbm_types::Bias bias_1(2);
  bias_1 << 2, 3;

  rbm_types::Data data_1(2);
  data_1 << -5, 4;

  double seikai = 2;
  
  
  ASSERT_EQ(seikai, bias_1.dot(data_1));
}

TEST_F(BiasTest, dot_VectorXd){
  Eigen::VectorXd bias_1(2);
  bias_1 << 2, 3;

  Eigen::VectorXd data_1(2);
  data_1 << -5, 4;

  double seikai = 2;
  std::cout << "dot_VectorXd" << std::endl;

  ASSERT_EQ(seikai, bias_1.dot(data_1));
}
