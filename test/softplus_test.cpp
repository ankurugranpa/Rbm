#include<gtest/gtest.h>

#include<softplus.h>

namespace {
  class SoftplusTest: public testing::Test{
    protected:

      virtual void SetUp() {
    }
  };
};


TEST_F(SoftplusTest, double){
  ASSERT_DOUBLE_EQ(0.313261687518222834049, rbm_utils::softplus(-1.0));
  ASSERT_DOUBLE_EQ(1.313261687518222834049, rbm_utils::softplus(1.0));
  ASSERT_DOUBLE_EQ(0.6931471805599453094172, rbm_utils::softplus(0));
}

