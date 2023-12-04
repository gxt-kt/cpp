#pragma once

#include <cmath>

#include "edge.h"
#include "vertex.h"

class Problem {
 public:
  /**
   * 问题的类型
   * SLAM问题还是通用的问题
   *
   * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
   * SLAM问题只接受一些特定的Vertex和Edge
   * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
   */
  enum class ProblemType { SLAM_PROBLEM, GENERIC_PROBLEM };

  using ulong = unsigned long;

  using HashVertex = std::map<unsigned long, std::shared_ptr<Vertex>>;
  using HashEdge = std::unordered_map<unsigned long, std::shared_ptr<Edge>>;
  using HashVertexIdToEdge =
      std::unordered_multimap<unsigned long, std::shared_ptr<Edge>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Problem(ProblemType problem_type) : problem_type_(problem_type) {
    // LogoutVectorSize();
    verticies_marg_.clear();
  };

  ~Problem() {}

  bool AddVertex(std::shared_ptr<Vertex> vertex) {
    // 已经存在了就报错
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
      gDebugWarn("AddVertex is exist");
      return false;
    }
    verticies_.insert({vertex->Id(), vertex});
    return true;
  };

  // gxt: not use
  // bool RemoveVertex(std::shared_ptr<Vertex> vertex);

  bool AddEdge(std::shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) != edges_.end()) {
      gDebugWarn("AddEdge is exist");
      return false;
    }

    edges_.insert({edge->Id(), edge});

    for (auto& vertex : edge->Verticies()) {
      vertexToEdge_.insert({vertex->Id(), edge});
    }
    return true;
  };

  // gxt: not use
  // bool RemoveEdge(std::shared_ptr<Edge> edge);

  /**
   * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
   * @param outlier_edges
   */
  // gxt: not use
  // void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

  /**
   * 求解此问题
   * @param iterations
   * @return
   */
  bool Solve(int iterations) {
    if (edges_.size() == 0 || verticies_.size() == 0) {
      gDebugWarn() << "\nCannot solve problem without edges or verticies";
      return false;
    }

    TIME_BEGIN();

    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H = J^T * J 矩阵
    MakeHessian();

    // LM 初始化
    ComputeLambdaInitLM();

    bool stop = false;
    int iter = 0;

    while (!stop && iter < iterations) {
      gDebugCol1() << "iter: " << iter << " , chi= " << currentChi_
                   << " , Lambda= " << currentLambda_;
      bool one_step_success{false};
      int false_cnt = 0;
      while (!one_step_success) {  // 不断尝试 Lambda, 直到成功迭代一步
        // 更新Hx=b为(H+uI)x=b也就是H变为H+uI
        AddLambdatoHessianLM();

        // 解线性方程 H =b x=H(-1)b
        SolveLinearSystem();

        // 把H+uI恢复到原来的H
        RemoveLambdaHessianLM();

        // 优化退出条件1： delta_x_ 很小则退出
        if (delta_x_.squaredNorm() <= my_type{1e-6} || false_cnt > 10) {
          gDebug("stop=true");
          stop = true;
          break;
        }

        // 更新状态量 X = X+ delta_x
        UpdateStates();

        // 判断当前步是否可行以及 LM 的 lambda 怎么更新
        one_step_success = IsGoodStepInLM();
        gDebugCol2(one_step_success);

        // 后续处理，
        if (one_step_success) {
          // 在新线性化点 构建 hessian
          MakeHessian();
          // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12
          // 很难达到，这里的阈值条件不应该用绝对值，而是相对值
          //                double b_max = 0.0;
          //                for (int i = 0; i < b_.size(); ++i) {
          //                    b_max = max(fabs(b_(i)), b_max);
          //                }
          //                // 优化退出条件2： 如果残差 b_max
          //                已经很小了，那就退出 stop = (b_max <= 1e-12);
          false_cnt = 0;
        } else {
          false_cnt++;
          RollbackStates();  // 误差没下降，回滚
        }
      }
      iter++;

      if (std::sqrt(currentChi_) <= stopThresholdLM_) {
        stop = true;
      }
    }

    return true;
  }

  /// 边缘化一个frame和以它为host的landmark
  bool Marginalize(
      std::shared_ptr<Vertex> frameVertex,
      const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);

  bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

  // test compute prior
  void TestComputePrior();

 private:
  /// Solve的实现，解通用问题
  bool SolveGenericProblem(int iterations);

  /// Solve的实现，解SLAM问题
  bool SolveSLAMProblem(int iterations);

  // gxt: 把变量总数给到ordering_generic_
  /// 设置各顶点的ordering_index
  void SetOrdering() {
    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    // 统计带估计的所有变量的总维度
    for (auto vertex : verticies_) {
      ordering_generic_ += vertex.second->LocalDimension();
    }
  }

  /// set ordering for new vertex in slam problem
  void AddOrderingSLAM(std::shared_ptr<Vertex> v);

  /// 构造大H矩阵
  void MakeHessian() {
    // 代优化变量总数
    unsigned long size = ordering_generic_;

    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
    for (auto& edge : edges_) {
      edge.second->ComputeResidual();
      edge.second->ComputeJacobians();

      std::vector<MatXX> jacobians = edge.second->Jacobians();
      std::vector<std::shared_ptr<Vertex>> verticies = edge.second->Verticies();
      assert(jacobians.size() == verticies.size());

      for (size_t i = 0; i < verticies.size(); ++i) {
        auto v_i = verticies.at(i);
        if (v_i->IsFixed()) {
          continue;  // Hessian 里不需要添加它的信息，也就是它的雅克比为 0
        }

        MatXX jacobian_i = jacobians.at(i);
        unsigned long index_i = v_i->OrderingId();
        unsigned long dim_i = v_i->LocalDimension();

        MatXX JtW = jacobian_i.transpose() * edge.second->Information();

        for (size_t j = i; j < verticies.size(); ++j) {
          auto v_j = verticies.at(j);
          if (v_j->IsFixed()) {
            continue;
          }

          MatXX jacobian_j = jacobians[j];
          unsigned long index_j = v_j->OrderingId();
          unsigned long dim_j = v_j->LocalDimension();
          assert(v_j->OrderingId() != -1);

          MatXX hessian = JtW * jacobian_j;
          // 所有的信息矩阵叠加起来
          H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
          if (j != i) {
            // 对称的下三角
            H.block(index_j, index_i, dim_j, dim_i).noalias() +=
                hessian.transpose();
          }
        }
        b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
      }
    }
    Hessian_ = H;
    b_ = b;
    // t_hessian_cost_;// gxt:时间貌似不重要在这里

    // gDebug(H);
    gDebug(Hessian_);
    // gDebug(b);
    gDebug(b_);

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
  }

  /// schur求解SBA
  void SchurSBA();

  /// 解线性方程
  void SolveLinearSystem() {
    delta_x_ = Hessian_.inverse() * b_;
    gDebug(delta_x_);
    // delta_x_ = H.ldlt().solve(b_);
  }

  /// 更新状态变量
  void UpdateStates() {
    for (auto vertex : verticies_) {
      unsigned long idx = vertex.second->OrderingId();
      unsigned long dim = vertex.second->LocalDimension();
      VecX delta = delta_x_.segment(idx, dim);

      // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
      vertex.second->Plus(delta);
    }
  }

  // 有时候 update 后残差会变大，需要退回去，重来
  void RollbackStates() {
    for (const auto& vertex : verticies_) {
      ulong idx = vertex.second->OrderingId();
      ulong dim = vertex.second->LocalDimension();
      VecX delta = delta_x_.segment(idx, dim);

      // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
      vertex.second->Plus(-delta);
    }
  }

  /// 计算并更新Prior部分
  void ComputePrior();

  /// 判断一个顶点是否为Pose顶点
  bool IsPoseVertex(std::shared_ptr<Vertex> v);

  /// 判断一个顶点是否为landmark顶点
  bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

  /// 在新增顶点后，需要调整几个hessian的大小
  void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

  /// 检查ordering是否正确
  bool CheckOrdering();

  // void LogoutVectorSize();

  /// 获取某个顶点连接到的边
  std::vector<std::shared_ptr<Edge>> GetConnectedEdges(
      std::shared_ptr<Vertex> vertex);

  /// Levenberg
  /// 计算LM算法的初始Lambda
  void ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    // 计算出当前的总残差
    for (const auto& edge : edges_) {
      currentChi_ += edge.second->Chi2();
    }

    // 计算先验的参数（如果有先验的话）
    if (err_prior_.rows() > 0) {
      currentChi_ += static_cast<double>(err_prior_.norm());
    }

    // 1. 第一步计算停止迭代条件stopThresholdLM_
    stopThresholdLM_ = 1e-6 * currentChi_;  // 迭代条件为 误差下降 1e-6 倍

    // 取出H矩阵对角线的最大值
    double maxDiagonal = 0.;
    unsigned long size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (unsigned long i = 0; i < size; ++i) {
      maxDiagonal =
          std::max(std::fabs(static_cast<double>(Hessian_(i, i))), maxDiagonal);
    }
    double tau = 1e-5;
    // 2. 根据对角线最大值计算出currentLambda_
    currentLambda_ = tau * maxDiagonal;  // 给到u0的初值
  }

  /// Hessian 对角线加上或者减去  Lambda
  void AddLambdatoHessianLM() {
    unsigned int size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (unsigned long i = 0; i < size; ++i) {
      Hessian_(i, i) += my_type{currentLambda_};
    }
  }

  void RemoveLambdaHessianLM() {
    unsigned long size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO::
    // 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (unsigned int i = 0; i < size; ++i) {
      Hessian_(i, i) -= my_type{currentLambda_};
    }
  }

  /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
  bool IsGoodStepInLM() {
    double scale = 0;
    // scale = delta_x_.transpose() * (my_type{currentLambda_} * delta_x_ + b_);
    scale =
        static_cast<double>((delta_x_.transpose() *
                             (my_type{currentLambda_} * delta_x_ + b_))(0, 0));
    // my_type scale_tmp = delta_x_.transpose() * (my_type{currentLambda_} *
    // delta_x_ + b_);
    // gDebugCol3(delta_x_);
    // gDebugCol3(currentLambda_);
    // gDebugCol3(b_);
    // gDebugCol3(scale);
    // gDebugCol3(scale_tmp);
    // gDebugCol4() << G_SPLIT_LINE;
    // gDebugCol4(my_type{currentLambda_} * delta_x_ + b_);
    // gDebugCol4(delta_x_.transpose());
    // gDebugCol4(delta_x_.transpose() *
    //            (my_type{currentLambda_} * delta_x_ + b_));
    scale += 1e-3;  // make sure it's non-zero :)

    // recompute residuals after update state
    // 统计所有的残差
    double tempChi = 0.0;
    for (auto edge : edges_) {
      edge.second->ComputeResidual();
      tempChi += edge.second->Chi2();
    }

    gDebugCol5(tempChi);
    gDebugCol5(currentChi_);

    double rho = (currentChi_ - tempChi) / scale;
    gDebugCol5(rho);

    // std::terminate();

    if (rho > 0 && std::isfinite(tempChi)) {  // last step was good, 误差在下降
      double alpha = 1. - pow((2 * rho - 1), 3);
      alpha = std::min(alpha, 2.0 / 3.0);
      double scaleFactor = std::max(1.0 / 3.0, alpha);
      currentLambda_ *= scaleFactor;
      ;
      ni_ = 2;
      currentChi_ = tempChi;
      return true;
    } else {
      currentLambda_ *= ni_;
      ni_ *= 2;
      return false;
    }
  }

  /// PCG 迭代线性求解器
  VecX PCGSolver(const MatXX& A, const VecX& b, int max_iter);

  double currentLambda_;
  double currentChi_;
  double stopThresholdLM_;  // LM 迭代退出阈值条件
  double ni_;               // 控制 Lambda 缩放大小

  ProblemType problem_type_;

  /// 整个信息矩阵
  MatXX Hessian_;
  VecX b_;
  VecX delta_x_;

  /// 先验部分信息
  MatXX H_prior_;
  VecX b_prior_;
  MatXX Jt_prior_inv_;
  VecX err_prior_;

  /// SBA的Pose部分
  MatXX H_pp_schur_;
  VecX b_pp_schur_;
  // Heesian 的 Landmark 和 pose 部分
  MatXX H_pp_;
  VecX b_pp_;
  MatXX H_ll_;
  VecX b_ll_;

  /// all vertices
  HashVertex verticies_;

  /// all edges
  HashEdge edges_;//std::unordered_map<unsigned long, std::shared_ptr<Edge>>

  /// 由vertex id查询edge
  HashVertexIdToEdge vertexToEdge_;

  /// Ordering related
  unsigned long ordering_poses_ = 0;
  unsigned long ordering_landmarks_ = 0;
  unsigned long ordering_generic_ = 0;

  std::map<unsigned long, std::shared_ptr<Vertex>>
      idx_pose_vertices_;  // 以ordering排序的pose顶点
  std::map<unsigned long, std::shared_ptr<Vertex>>
      idx_landmark_vertices_;  // 以ordering排序的landmark顶点

  HashVertex verticies_marg_;

  bool bDebug = false;
  double t_hessian_cost_{0};
  double t_PCGsolve_cost{0};
};
