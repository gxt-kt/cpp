#include <random>

#include "common.h"
#include "problem.h"

// class CurveFittingVertex : public Vertex {
//  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//   CurveFittingVertex() : Vertex(3) {}  // abc: 三个参数， Vertex 是 3 维的
//   virtual std::string TypeInfo() const { return "abc"; }
// };

class CurveFittingVertex1 : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CurveFittingVertex1() : Vertex(1) {}  // a
  virtual std::string TypeInfo() const { return "a"; }
};
class CurveFittingVertex2 : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CurveFittingVertex2() : Vertex(1) {}  // a
  virtual std::string TypeInfo() const { return "b"; }
};
class CurveFittingVertex3 : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CurveFittingVertex3() : Vertex(1) {}  // a
  virtual std::string TypeInfo() const { return "c"; }
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
    Vec1 a = verticies_.at(0)->Parameters();
    Vec1 b = verticies_.at(1)->Parameters();
    Vec1 c = verticies_.at(2)->Parameters();
    // residual_(0) = my_type{
    //     std::exp(static_cast<double>(abc(0) * my_type{x_} * my_type{x_} +
    //                                  abc(1) * my_type{x_} + abc(2))) -
    //     y_};  // 构建残差

    // TODO: gxt
    // 注意这里也很容易溢出
    residual_(0) = my_type{std::exp(static_cast<double>(a(0)) * x_ * x_ +
                                    static_cast<double>(b(0)) * x_ +
                                    static_cast<double>(c(0))) -
                           y_};  // 构建残差
  }

  virtual void ComputeJacobians() override {
    // Vec3 abc = verticies_.at(0)->Parameters();
    Vec1 a = verticies_.at(0)->Parameters();
    Vec1 b = verticies_.at(1)->Parameters();
    Vec1 c = verticies_.at(2)->Parameters();
    double exp_y = std::exp(static_cast<double>(
        a(0) * my_type{x_} * my_type{x_} + b(0) * my_type{x_} + c(0)));

    Eigen::Matrix<my_type, 1, 3> jaco_abc;
    jaco_abc << my_type{x_} * my_type{x_} * my_type{exp_y},
        my_type{x_} * my_type{exp_y}, 1 * my_type{exp_y};
    jacobians_.resize(3);
    jacobians_.at(0) = Vec1(jaco_abc(0,0));
    jacobians_.at(1) = Vec1(jaco_abc(0,1));
    jacobians_.at(2) = Vec1(jaco_abc(0,2));
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
  // std::shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());

  std::shared_ptr<CurveFittingVertex1> vertex1(new CurveFittingVertex1());
  std::shared_ptr<CurveFittingVertex2> vertex2(new CurveFittingVertex2());
  std::shared_ptr<CurveFittingVertex3> vertex3(new CurveFittingVertex3());

  // 设定待估计参数 a, b, c初始值
  // vertex->SetParameters(Eigen::Vector3d(0, 0, 0));
  vertex1->SetParameters(Eigen::Matrix<my_type, 1, 1>(my_type{0}));
  vertex2->SetParameters(Eigen::Matrix<my_type, 1, 1>(my_type{0}));
  vertex3->SetParameters(Eigen::Matrix<my_type, 1, 1>(my_type{0}));

  // 将待估计的参数加入最小二乘问题
  problem.AddVertex(vertex1);
  problem.AddVertex(vertex2);
  problem.AddVertex(vertex3);

  for (int i = 0; i < N; ++i) {
    double x = i / 100.0;
    double n = noise(generator);

    // 观测 y
    double y = std::exp(a * x * x + b * x + c) + n;
    // double y = std::exp( a*x*x + b*x + c );

    // 每个观测对应的残差函数
    std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
    std::vector<std::shared_ptr<Vertex>> edge_vertex;
    edge_vertex.push_back(vertex1);
    edge_vertex.push_back(vertex2);
    edge_vertex.push_back(vertex3);
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
  std::cout << vertex1->Parameters().transpose() << std::endl;
  std::cout << vertex2->Parameters().transpose() << std::endl;
  std::cout << vertex3->Parameters().transpose() << std::endl;
  std::cout << "-------ground truth: " << std::endl;
  std::cout << "1.0,  2.0,  1.0" << std::endl;

  // std
  return 0;
}
