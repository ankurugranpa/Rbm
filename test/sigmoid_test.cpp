#include<gtest/gtest.h>

#include<sigmoid.h>

namespace {
  class SigmoidTest: public testing::Test{
    protected:
      // Model calc_rbm;

      virtual void SetUp() {
        // calc_rbm = Rbm(2, 2);
        // calc_rbm(2, 2);
    }
  };
};


TEST_F(SigmoidTest, double){
  ASSERT_DOUBLE_EQ(0.9933071490757153, rbm_utils::sig(5.0));
  ASSERT_DOUBLE_EQ(0.5, rbm_utils::sig(0.0));
  ASSERT_DOUBLE_EQ(0.006692850924284856, rbm_utils::sig(-5.0));
}


TEST_F(SigmoidTest, Bias){
  Bias in(3);
  in << 5.0,
        0.0,
       -5.0;

  Bias out(3);
  out << 0.9933071490757153,
         0.5,
         0.006692850924284856;

  Eigen::VectorXd answer;
  answer = rbm_utils::sig(in);

  for(int i=0; i<in.size(); i++){
    ASSERT_DOUBLE_EQ(out(i), answer(i));
  }
}

TEST_F(SigmoidTest, Sigmoid_VECTOR_D){
  Eigen::Vector3d in;
  in << 5.0,
        0.0,
       -5.0;

  Eigen::Vector3d out;
  out << 0.9933071490757153,
         0.5,
         0.006692850924284856;

  Eigen::VectorXd answer;
  answer = rbm_utils::sig(in);

  for(int i=0; i<in.size(); i++){
    ASSERT_DOUBLE_EQ(out(i), answer(i));
  }
}

