#include<gtest/gtest.h>

#include<bias.h>

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

