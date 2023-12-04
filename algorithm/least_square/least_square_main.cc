#include <random>

#include "common.h"
#include "problem.h"

class CurveFittingVertex : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CurveFittingVertex() : Vertex(3) {}  // abc: 三个参数， Vertex 是 3 维的
  virtual std::string TypeInfo() const { return "abc"; }
};

class CurveFittingEdge : public Edge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CurveFittingEdge(double x, double y)
      : Edge(1, 1, std::vector<std::string>{"abc"}) {
    x_ = x;
    y_ = y;
  }

  virtual void ComputeResidual() override {
    Vec3 abc = verticies_.at(0)->Parameters();
    // residual_(0) = my_type{
    //     std::exp(static_cast<double>(abc(0) * my_type{x_} * my_type{x_} +
    //                                  abc(1) * my_type{x_} + abc(2))) -
    //     y_};  // 构建残差
    
    // TODO: gxt
    // 注意这里也很容易溢出
    residual_(0) = my_type{std::exp(static_cast<double>(abc(0)) * x_ * x_ +
                                    static_cast<double>(abc(1)) * x_ +
                                    static_cast<double>(abc(2))) -
                           y_};  // 构建残差
  }

  virtual void ComputeJacobians() override {
    Vec3 abc = verticies_.at(0)->Parameters();
    double exp_y = std::exp(static_cast<double>(
        abc(0) * my_type{x_} * my_type{x_} + abc(1) * my_type{x_} + abc(2)));

    Eigen::Matrix<my_type, 1, 3> jaco_abc;
    jaco_abc << my_type{x_} * my_type{x_} * my_type{exp_y},
        my_type{x_} * my_type{exp_y}, 1 * my_type{exp_y};
    jacobians_.at(0) = jaco_abc;
  }

  virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }

  double x_, y_;
};

int main() {
  gDebug() << G_FILE;

  double a = 1.0, b = 2.0, c = 1.0;  // 真实参数值

  int N = 100;         // 数据点
  double w_sigma = 1;  // 噪声Sigma值

  std::default_random_engine generator;
  std::normal_distribution<double> noise(0, w_sigma);

  // 构建 problem
  Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
  std::shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());

  // 设定待估计参数 a, b, c初始值
  // vertex->SetParameters(Eigen::Vector3d(0, 0, 0));
  vertex->SetParameters(
      Eigen::Matrix<my_type, 3, 1>(my_type{0}, my_type{0}, my_type{0}));
  // 将待估计的参数加入最小二乘问题
  problem.AddVertex(vertex);

  for (int i = 0; i < N; ++i) {
    double x = i / 100.0;
    double n = noise(generator);

    // 观测 y
    double y = std::exp(a * x * x + b * x + c) + n;
    // double y = std::exp( a*x*x + b*x + c );

    // 每个观测对应的残差函数
    std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
    std::vector<std::shared_ptr<Vertex>> edge_vertex;
    edge_vertex.push_back(vertex);
    edge->SetVertex(edge_vertex);

    // 把这个残差添加到最小二乘问题
    problem.AddEdge(edge);
  }

  gDebugCol3() << "\nTest CurveFitting start...";

  /// 使用 LM 求解
  problem.Solve(30);

  // gDebugCol5(problem);

  std::cout << "-------After optimization, we got these parameters :"
            << std::endl;
  std::cout << vertex->Parameters().transpose() << std::endl;
  std::cout << "-------ground truth: " << std::endl;
  std::cout << "1.0,  2.0,  1.0" << std::endl;

  // std
  return 0;
}
