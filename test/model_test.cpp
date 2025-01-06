#include<gtest/gtest.h>
#include<model.h>
#include<binary.h>

namespace {
  class ModelTest: public testing::Test{
    protected:
      rbm::Model rbm_model;
      virtual void SetUp() {
        rbm_model = rbm::Model(4, 2);
      // rbm_model(2, 2);
    }
  };
};


TEST_F(ModelTest, Constructor){
  ASSERT_EQ(4, rbm_model.visible_dim);
  ASSERT_EQ(2, rbm_model.hidden_dim);

  for(int i=0; i<rbm_model.visible_dim; i++){
    ASSERT_TRUE(rbm_model.parametar.visible_bias[i] >= -1 || rbm_model.parametar.visible_bias[i] <= 1);
  }

  for(int i=0; i<rbm_model.hidden_dim; i++){
    ASSERT_TRUE(rbm_model.parametar.hidden_bias[i] >= -1 || rbm_model.parametar.hidden_bias[i] <= 1);
  }

  for(int i=0; i<rbm_model.visible_dim; i++){
    for(int j=0; j<rbm_model.hidden_dim; j++){
      ASSERT_TRUE(rbm_model.parametar.weight(i, j) >= -1 || rbm_model.parametar.weight(i, j) <= 1);
    }
  }
}


TEST_F(ModelTest, LambdaVisible){
  Parametar test_parametaer(3, 2);
  Bias b(3);
  b << 1,
       2,
       3;

  Weight w(3, 2);
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
    ASSERT_DOUBLE_EQ(answer(i), rbm_model.lambda_visible(test_parametaer, h)(i));
  }
}


TEST_F(ModelTest, LambdaHidenn){
  Parametar test_parametaer(3, 2);
  Bias c(2);
  c << 1,
       2;
  Weight w(3, 2);
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
    ASSERT_DOUBLE_EQ(answer(i), rbm_model.lambda_hidden(test_parametaer, v)(i));
  }
}
TEST_F(ModelTest, CostFunc){
  Parametar test_parametaer(2, 2);

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

  ASSERT_DOUBLE_EQ(-1215, rbm_model.cost_func(test_parametaer, rand_visible, rand_hidden));
}

TEST_F(ModelTest, AllStatus){
  DataSet status = rbm_model.all_status();
  rbm_utils::Binary gen_bit(rbm_model.visible_dim+rbm_model.hidden_dim);

  // std::cout << status.size() << std::endl;
  for(auto i=0; i<64; i++){
    std::cout << status[i].transpose() << std::endl;
    ASSERT_EQ(status[i], gen_bit.num2binary(i));
  }
}
