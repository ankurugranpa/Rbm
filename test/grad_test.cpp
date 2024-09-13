#include<gtest/gtest.h>

#include<grad.h>
using namespace rbm_types;

namespace {
  class GradTest: public testing::Test{
    protected:
      rbm_types::Grad grad_test;

      GradTest() : grad_test(4, 2) {} // コンストラクタで初期化
  };
};


TEST_F(GradTest, plus){
  std::cout << grad_test.visible_grad.transpose() << std::endl;
  std::cout << grad_test.hidden_grad.transpose() << std::endl;
  std::cout << grad_test.weight_grad << std::endl;

  grad_test.visible_grad << 5.0, 5.0, 5.0, 5.0;

  grad_test.hidden_grad << 4.0, 4.0;

  grad_test.weight_grad << 4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0;
  Grad test_num(4, 2);
  test_num.visible_grad << 5.0, 5.0, 5.0, 5.0;

  test_num.hidden_grad << 4.0, 4.0;

  test_num.weight_grad << 4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0;

  Grad result_num(4, 2);
  result_num.visible_grad << 10.0, 10.0, 10.0, 10.0;

  result_num.hidden_grad << 8.0, 8.0;

  result_num.weight_grad << 8.0, 8.0,
                            8.0, 8.0,
                            8.0, 8.0,
                            8.0, 8.0;
  // std::cout << (test_num+grad_test).visible_grad.transpose() << std::endl;
  // std::cout << "##" << std::endl;
  // std::cout << (test_num+grad_test).hidden_grad.transpose() << std::endl;
  // std::cout << "##" << std::endl;
  // std::cout << (test_num+grad_test).weight_grad << std::endl;

  ASSERT_EQ(result_num.visible_grad, (test_num+grad_test).visible_grad);
  ASSERT_EQ(result_num.hidden_grad, (test_num+grad_test).hidden_grad);
  ASSERT_EQ(result_num.weight_grad, (test_num+grad_test).weight_grad);

}

TEST_F(GradTest, minus){
  grad_test.visible_grad << 5.0, 5.0, 5.0, 5.0;

  grad_test.hidden_grad << 4.0, 4.0;

  grad_test.weight_grad << 4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0;
  Grad test_num(4, 2);
  test_num.visible_grad << 5.0, 5.0, 5.0, 5.0;

  test_num.hidden_grad << 4.0, 4.0;

  test_num.weight_grad << 4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0;

  Grad result_num(4, 2);
  result_num.visible_grad << 0, 0, 0, 0;

  result_num.hidden_grad << 0, 0;

  result_num.weight_grad << 0, 0,
                            0, 0,
                            0, 0,
                            0, 0;

  ASSERT_EQ(result_num.visible_grad, (test_num-grad_test).visible_grad);
  ASSERT_EQ(result_num.hidden_grad, (test_num-grad_test).hidden_grad);
  ASSERT_EQ(result_num.weight_grad, (test_num-grad_test).weight_grad);
}

TEST_F(GradTest, product){
  grad_test.visible_grad << 5.0, 5.0, 5.0, 5.0;

  grad_test.hidden_grad << 4.0, 4.0;

  grad_test.weight_grad << 4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0;
  Grad test_num(4, 2);
  test_num.visible_grad << 5.0, 5.0, 5.0, 5.0;

  test_num.hidden_grad << 4.0, 4.0;

  test_num.weight_grad << 4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0;

  Grad result_num(4, 2);
  result_num.visible_grad << 25, 25, 25, 25;

  result_num.hidden_grad << 16, 16;

  result_num.weight_grad << 16, 16,
                            16, 16,
                            16, 16,
                            16, 16;

  ASSERT_EQ(result_num.visible_grad, (test_num*grad_test).visible_grad);
  ASSERT_EQ(result_num.hidden_grad, (test_num*grad_test).hidden_grad);
  ASSERT_EQ(result_num.weight_grad, (test_num*grad_test).weight_grad);
}
TEST_F(GradTest, division){
  grad_test.visible_grad << 5.0, 5.0, 5.0, 5.0;

  grad_test.hidden_grad << 4.0, 4.0;

  grad_test.weight_grad << 4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0,
                           4.0, 4.0;
  Grad test_num(4, 2);
  test_num.visible_grad << 5.0, 5.0, 5.0, 5.0;

  test_num.hidden_grad << 4.0, 4.0;

  test_num.weight_grad << 4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0,
                          4.0, 4.0;

  Grad result_num(4, 2);
  result_num.visible_grad << 1, 1, 1, 1;

  result_num.hidden_grad << 1, 1;

  result_num.weight_grad << 1, 1,
                            1, 1,
                            1, 1,
                            1, 1;

  ASSERT_EQ(result_num.visible_grad, (test_num/grad_test).visible_grad);
  ASSERT_EQ(result_num.hidden_grad, (test_num/grad_test).hidden_grad);
  ASSERT_EQ(result_num.weight_grad, (test_num/grad_test).weight_grad);
}
