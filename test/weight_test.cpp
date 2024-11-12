#include<gtest/gtest.h>

#include<weight.h>


namespace {
  class WeightTest: public testing::Test{
    protected:
      rbm_types::Weight weight_test;
      WeightTest() : weight_test(3, 5) {} // コンストラクタで初期化
  };
};



TEST_F(WeightTest, block){
  weight_test << 1, 2, 3, 4, 5,
                 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15;
  rbm_types::Weight weight_test_buf(1, 5);
  weight_test_buf << 11, 12, 13, 14, 15;
  ASSERT_EQ(weight_test.block(2, 0, 1, 5), weight_test_buf);
}


