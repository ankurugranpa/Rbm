#include<Eigen/Dense>
#include<random>

namespace rbm_utils{
  class Distribuition{
    public:
      std::random_device rd;
      std::mt19937 gen;
      std::uniform_real_distribution<> dis;

      Distribuition(double range_start, double range_end);
      ~Distribuition();
      double random_num();
  };
}
