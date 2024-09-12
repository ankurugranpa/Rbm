#include<typeinfo>
#include<iostream>

#include<gtest/gtest.h>
#include<rbm.h>

namespace {
  class RbmTest: public testing::Test{
    protected:
      rbm::Model calc_rbm;
      virtual void SetUp() {
        calc_rbm = rbm::Model(4, 2);
      // calc_rbm(2, 2);
    }
  };
};


TEST_F(RbmTest, Constructor){
  ASSERT_EQ(4, calc_rbm.visible_dim);
  ASSERT_EQ(2, calc_rbm.hidden_dim);
}


TEST_F(RbmTest, LambdaVisible){
  rbm::Parametar test_parametaer(3, 2);
  rbm::Bias b(3);
  b << 1,
       2,
       3;

  rbm::Weight w(3, 2);
  w << 1, 4,
       2, 5,
       3, 6;
  Eigen::VectorXi h(2);
  h << 1,
       1;

  Eigen::Vector3d answer;
  answer << 6,
            9,
            12;
  test_parametaer.visible_bias = b;
  test_parametaer.weight = w;

  for(int i=0; i<answer.size(); i++){
    ASSERT_DOUBLE_EQ(answer(i), calc_rbm.lambda_visible(test_parametaer, h)(i));
  }
}


TEST_F(RbmTest, LambdaHidenn){
  rbm::Parametar test_parametaer(3, 2);
  rbm::Bias c(2);
  c << 1,
       2;
  rbm::Weight w(3, 2);
  w << 1, 4,
       2, 5,
       3, 6;
  Eigen::VectorXi v(3);
  v << 1,
       1,
       1;
  Eigen::Vector2d answer;
  answer << 7,
            17;

  test_parametaer.hidden_bias = c;
  test_parametaer.weight = w;
  for(int i=0; i<answer.size(); i++){
    ASSERT_DOUBLE_EQ(answer(i), calc_rbm.lambda_hidden(test_parametaer, v)(i));
  }
}
TEST_F(RbmTest, Cost){
  rbm::Parametar test_parametaer(2, 2);

  test_parametaer.visible_bias << 1,
                                  2;

  Eigen::VectorXi rand_visible(2);
  rand_visible << 3,
                  4;

  test_parametaer.hidden_bias << 5,
                                 6;

  Eigen::VectorXi rand_hidden(2);
  rand_hidden << 7,
                 8;

  test_parametaer.weight(0, 0) = 9;
  test_parametaer.weight(0, 1) = 10;
  test_parametaer.weight(1, 0) = 11;
  test_parametaer.weight(1, 1) = 12;

  ASSERT_DOUBLE_EQ(-1215, calc_rbm.cost_func(test_parametaer, rand_visible, rand_hidden));
  ;
}
