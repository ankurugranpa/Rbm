#include<gtest/gtest.h>

#include<binary.h>
#include<data.h>

namespace {
  class BinaryTest: public testing::Test{
    protected:
      // rbm::Model calc_rbm;
      rbm_utils::Binary binary_test;

      virtual void SetUp() {
        binary_test = rbm_utils::Binary(4);
    }
  };
};


TEST_F(BinaryTest, num1binary){
  rbm::Data data_test = binary_test.num2binary(15);
  for(int i=0; i<4; i++){
    ASSERT_EQ(data_test(i), 1);
  }
}
