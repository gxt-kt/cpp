#pragma once

#include <cmath>

// 主要用在函数SelfAdjointEigenSolver
#include <eigen3/Eigen/Dense>

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

  void TestMarginalize() {
    // Add marg test
    int idx = 1;           // marg 中间那个变量
    int dim = 1;           // marg 变量的维度
    int reserve_size = 3;  // 总共变量的维度
    double delta1 = 0.1 * 0.1;
    double delta2 = 0.2 * 0.2;
    double delta3 = 0.3 * 0.3;

    int cols = 3;
    MatXX H_marg(MatXX::Zero(cols, cols));
    H_marg << my_type{1. / delta1}, my_type{-1. / delta1}, my_type{0},
        my_type{-1. / delta1}, my_type{1. / delta1 + 1. / delta2 + 1. / delta3},
        my_type{-1. / delta3}, my_type{0.}, my_type{-1. / delta3},
        my_type{1 / delta3};
    std::cout << "---------- TEST Marg: before marg------------" << std::endl;
    std::cout << H_marg << std::endl;

    // TODO:: home work. 将变量移动到右下角
    /// 准备工作： move the marg pose to the Hmm bottown right
    // 将 row i 移动矩阵最下面
    MatXX temp_rows = H_marg.block(idx, 0, dim, reserve_size);
    MatXX temp_botRows =
        H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
    // H_marg.block(?,?,?,?) = temp_botRows;
    // H_marg.block(?,?,?,?) = temp_rows;
    // NOTE: gxt: 完成TODO
    H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
    H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

    // 将 col i 移动矩阵最右边
    MatXX temp_cols = H_marg.block(0, idx, reserve_size, dim);
    MatXX temp_rightCols =
        H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
    H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) =
        temp_rightCols;
    H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

    std::cout << "---------- TEST Marg: 将变量移动到右下角------------"
              << std::endl;
    std::cout << H_marg << std::endl;

    /// 开始 marg ： schur
    double eps = 1e-8;
    int m2 = dim;
    int n2 = reserve_size - dim;  // 剩余变量的维度
    // Eigen::MatrixXd Amm = my_type{0.5} * (H_marg.block(n2, n2, m2, m2) +
    //                              H_marg.block(n2, n2, m2, m2).transpose());
    MatXX Amm = my_type{0.5} * (H_marg.block(n2, n2, m2, m2) +
                                H_marg.block(n2, n2, m2, m2).transpose());

    // Eigen::SelfAdjointEigenSolver<MatXX> saes;
    // MatXX Amm_inv;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> saes(Amm.cast<double>());
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Amm_inv_double = saes.eigenvectors() *
                    Eigen::VectorXd((saes.eigenvalues().array() > eps)
                             .select(saes.eigenvalues().array().inverse(),
                             0))
                        .asDiagonal() *
                    saes.eigenvectors().transpose();
    MatXX Amm_inv=Amm_inv_double.cast<my_type>();


    // TODO:: home work. 完成舒尔补操作
    // Eigen::MatrixXd Arm = H_marg.block(?,?,?,?);
    // Eigen::MatrixXd Amr = H_marg.block(?,?,?,?);
    // Eigen::MatrixXd Arr = H_marg.block(?,?,?,?);
    // NOTE: gxt 完成TODO
    MatXX Arm = H_marg.block(0, n2, n2, m2);
    MatXX Amr = H_marg.block(n2, 0, m2, n2);
    MatXX Arr = H_marg.block(0, 0, n2, n2);

    MatXX tempB = Arm * Amm_inv;
    MatXX H_prior = Arr - tempB * Amr;

    std::cout << "---------- TEST Marg: after marg------------" << std::endl;
    std::cout << H_prior << std::endl;
  }

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
    int debug = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    // 统计带估计的所有变量的总维度
    gDebugWarn(verticies_.size());
    for (auto vertex : verticies_) {
      ordering_generic_ += vertex.second->LocalDimension();

      if (IsPoseVertex(vertex.second)) {
        debug += vertex.second->LocalDimension();
      }

      if (problem_type_ ==
          ProblemType::SLAM_PROBLEM)  // 如果是 slam 问题，还要分别统计 pose 和
                                      // landmark 的维数，后面会对他们进行排序
      {
        AddOrderingSLAM(vertex.second);
      }
      if (IsPoseVertex(vertex.second)) {
        std::cout << vertex.second->Id()
                  << " order: " << vertex.second->OrderingId() << std::endl;
      }
    }
    gDebugWarn(ordering_generic_);

    std::cout << "\n ordered_landmark_vertices_ size : "
              << idx_landmark_vertices_.size() << std::endl;
    if (problem_type_ == ProblemType::SLAM_PROBLEM) {
      // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark
      // 在后,而 pose 在前
      ulong all_pose_dimension = ordering_poses_;
      for (auto landmarkVertex : idx_landmark_vertices_) {
        landmarkVertex.second->SetOrderingId(
            landmarkVertex.second->OrderingId() + all_pose_dimension);
      }
    }
  }

  /// set ordering for new vertex in slam problem
  void AddOrderingSLAM(std::shared_ptr<Vertex> v) {
    if (IsPoseVertex(v)) {
      v->SetOrderingId(ordering_poses_);
      idx_pose_vertices_.insert(
          std::pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
      ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
      v->SetOrderingId(ordering_landmarks_);
      ordering_landmarks_ += v->LocalDimension();
      idx_landmark_vertices_.insert(
          std::pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
  }

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
    // gDebug(Hessian_);
    // gDebug(b);
    // gDebug(b_);

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
  }

  /// schur求解SBA
  void SchurSBA();

  /// 解线性方程
  void SolveLinearSystem() {
    // gxt:
    // 主要是求解出delta_x_
    // 其中分成普通问题和SLAM问题
    // 其中普通问题直接求解就好了
    // SLAM由于矩阵比较稀疏，可以使用舒尔补的方式求解

    // 非 SLAM 问题直接求解
    if (problem_type_ == ProblemType::GENERIC_PROBLEM) {
      delta_x_ = Hessian_.inverse() * b_;
      gDebug(delta_x_);
    } else {
      // SLAM 问题采用舒尔补的计算方式

      // step1: schur marginalization --> Hpp, bpp
      int reserve_size = ordering_poses_;
      int marg_size = ordering_landmarks_;

      // TODO:: home work. 完成矩阵块取值，Hmm，Hpm，Hmp，bpp，bmm
      // MatXX Hmm = Hessian_.block(?,?, ?, ?);
      // MatXX Hpm = Hessian_.block(?,?, ?, ?);
      // MatXX Hmp = Hessian_.block(?,?, ?, ?);
      // VecX bpp = b_.segment(?,?);
      // VecX bmm = b_.segment(?,?);
      // NOTE: gxt 完成TODO
      MatXX Hmm =
          Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
      MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
      MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
      VecX bpp = b_.segment(0, reserve_size);
      VecX bmm = b_.segment(reserve_size, marg_size);

      // Hmm
      // 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
      MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
      for (auto landmarkVertex : idx_landmark_vertices_) {
        int idx = landmarkVertex.second->OrderingId() - reserve_size;
        int size = landmarkVertex.second->LocalDimension();
        Hmm_inv.block(idx, idx, size, size) =
            Hmm.block(idx, idx, size, size).inverse();
      }

      // TODO:: home work. 完成舒尔补 Hpp, bpp 代码
      MatXX tempH = Hpm * Hmm_inv;
      // H_pp_schur_ = Hessian_.block(?,?,?,?) - tempH * Hmp;
      // b_pp_schur_ = bpp - ? * ?;
      // NOTE: gxt 完成TODO
      H_pp_schur_ =
          Hessian_.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
      b_pp_schur_ = bpp - tempH * bmm;

      // step2: solve Hpp * delta_x = bpp
      VecX delta_x_pp(VecX::Zero(reserve_size));
      // PCG Solver
      for (ulong i = 0; i < ordering_poses_; ++i) {
        H_pp_schur_(i, i) += my_type{currentLambda_};
      }

      int n = H_pp_schur_.rows() * 2;  // 迭代次数
      delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_,
                             n);  // 哈哈，小规模问题，搞 pcg 花里胡哨
      delta_x_.head(reserve_size) = delta_x_pp;
      //        std::cout << delta_x_pp.transpose() << std::endl;

      // TODO:: home work. step3: solve landmark
      VecX delta_x_ll(marg_size);
      // delta_x_ll = ???;
      // NOTE: gxt 完成TODO
      delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
      delta_x_.tail(marg_size) = delta_x_ll;
    }
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
  bool IsPoseVertex(std::shared_ptr<Vertex> v) {
    std::string type = v->TypeInfo();
    return type == std::string("VertexPose");
  }

  /// 判断一个顶点是否为landmark顶点
  bool IsLandmarkVertex(std::shared_ptr<Vertex> v) {
    std::string type = v->TypeInfo();
    return type == std::string("VertexPointXYZ") ||
           type == std::string("VertexInverseDepth");
  }

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
  /// 用在SLAM舒尔补求解中
  VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter = -1) {
    assert(A.rows() == A.cols() &&
           "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = static_cast<double>(r0.dot(z0));
    double alpha = r0z0 / static_cast<double>(p.dot(w));
    VecX r1 = r0 - my_type{alpha} * w;
    int i = 0;
    double threshold = 1e-6 * static_cast<double>(r0.norm());
//    while (static_cast<double>(r1.norm()) > threshold && i < n) {
    while (true) {
      // NOTE: gxt: 注意这里有个大坑，直接用my_type的norm()会触发assert因为数字变小了？不懂，哈哈
      my_type r1_norm=my_type{r1.cast<double>().norm()};
      double r1_norm_double=static_cast<double>(r1_norm);
      if(r1_norm_double<=threshold) {break;}
      if(i>=n) {break;}

      i++;
      VecX z1 = M_inv * r1;
      double r1z1 = static_cast<double>(r1.dot(z1));
      double belta = r1z1 / r0z0;
      z0 = z1;
      r0z0 = r1z1;
      r0 = r1;
      p = my_type{belta} * p + z1;
      w = A * p;
      alpha = r1z1 / static_cast<double>(p.dot(w));
      x += my_type{alpha} * p;
      r1 -= my_type{alpha} * w;
    }
    return x;
  }

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
  HashEdge edges_;  // std::unordered_map<unsigned long, std::shared_ptr<Edge>>

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
