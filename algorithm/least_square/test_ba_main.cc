#include <random>

#include "common.h"
#include "edge_reprojection.h"
#include "problem.h"
#include "se3.hpp"
#include "so3.hpp"

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(
    const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;
  // gDebug(TYPET(Derived));
  // gDebug(TYPET(Scalar_t));
  // gDebug() << __PRETTY_FUNCTION__;

  Eigen::Quaternion<Scalar_t> dq;
  Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
  half_theta /= static_cast<Scalar_t>(2.0);
  dq.w() = static_cast<Scalar_t>(1.0);
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

/**
 * Pose vertex
 * parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
 * optimization is perform on manifold, so update is 6 DoF, left multiplication
 *
 * pose is represented as Twb in VIO case
 */
class VertexPose : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexPose() : Vertex(7, 6) {}

  /// 加法，可重定义
  /// 默认是向量加
  virtual void Plus(const VecX &delta) override {
    VecX &parameters = Parameters();

    VecX para_tmp = Parameters();

    parameters.head<3>() += delta.head<3>();
    Qd q(parameters[6], parameters[3], parameters[4], parameters[5]);
    q = q * (Sophus::SO3d::exp(Eigen::Vector3d(static_cast<double>(delta[3]),
                                               static_cast<double>(delta[4]),
                                               static_cast<double>(delta[5])))
                 .unit_quaternion())
                .cast<my_type>();  // right multiplication with so3
    q.normalized();
    parameters[3] = q.x();
    parameters[4] = q.y();
    parameters[5] = q.z();
    parameters[6] = q.w();
    // gDebug() << VAR(parameters.transpose());

    // 上面用的是Sophus进行四元数的相加
    // 下面用Eigen库进行相加，然后对结果进行对比
    // {
    //   Eigen::Map<const Eigen::Vector3d> p(&para_tmp(0));
    //   Eigen::Map<const Eigen::Quaterniond> q(&para_tmp(0) + 3);

    //   Eigen::Map<const Eigen::Vector3d> dp(&delta(0));
    //   Eigen::Quaterniond dq=deltaQ(Eigen::Map<const Eigen::Vector3d>(&delta(0) + 3));

    //   Eigen::Vector3d p_plus_delta;
    //   Eigen::Quaterniond q_plus_delta;

    //   p_plus_delta = p + dp;
    //   para_tmp.head<3>() = p_plus_delta;

    //   q_plus_delta = (q * dq).normalized();
    //   para_tmp[3] = q_plus_delta.x();
    //   para_tmp[4] = q_plus_delta.y();
    //   para_tmp[5] = q_plus_delta.z();
    //   para_tmp[6] = q_plus_delta.w();

    //   gDebug() << VAR(para_tmp.transpose());
    //   gDebugCol3() << G_SPLIT_LINE;
    // }

    // 下面这个演示的是数值分析判断Pulse是否计算正确
    //    Qd test = Sophus::SO3d::exp(Vec3(0.2, 0.1, 0.1)).unit_quaternion() *
    //    Sophus::SO3d::exp(-Vec3(0.2, 0.1, 0.1)).unit_quaternion(); std::cout
    //    << test.x()<<" "<< test.y()<<" "<<test.z()<<" "<<test.w() <<std::endl;
  }

  std::string TypeInfo() const override { return "VertexPose"; }

  /**
   * 需要维护[H|b]矩阵中的如下数据块
   * p: pose, m:mappoint
   *
   *     Hp1_p2
   *     Hp2_p2    Hp2_m1    Hp2_m2    Hp2_m3     |    bp2
   *
   *                         Hm2_m2               |    bm2
   *                                   Hm2_m3     |    bm3
   * 1. 若该Camera为source camera，则维护vHessionSourceCamera；
   * 2. 若该Camera为measurement camera, 则维护vHessionMeasurementCamera；
   * 3. 并一直维护m_HessionDiagonal；
   */
};

/**
 * 以逆深度形式存储的顶点
 */
class VertexInverseDepth : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexInverseDepth() : Vertex(1) {}

  virtual std::string TypeInfo() const { return "VertexInverseDepth"; }
};

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
  Frame(Mat33 R, Vec3 t) : Rwc(R), qwc(R), twc(t){};
  Mat33 Rwc;
  Qd qwc;
  Vec3 twc;

  std::unordered_map<int, Vec3> featurePerId;  // 该帧观测到的特征以及特征id
};

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(std::vector<Frame> &cameraPoses,
                           std::vector<Vec3> &points) {
  int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
  int poseNums = 3;  // 相机数目

  double radius = 8;
  for (int n = 0; n < poseNums; ++n) {
    double theta = n * 2 * M_PI / (poseNums * 4);  // 1/4 圆弧
    // 绕 z轴 旋转
    Mat33 R;
    R = Eigen::AngleAxis<my_type>(my_type{theta}, Vec3::UnitZ());
    Vec3 t = Vec3(my_type{radius * cos(theta) - radius},
                  my_type{radius * sin(theta)}, my_type{1 * sin(2 * theta)});
    cameraPoses.push_back(Frame(R, t));
  }

  // 随机数生成三维特征点
  std::default_random_engine generator;
  std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
  for (int j = 0; j < featureNums; ++j) {
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(4., 8.);

    Vec3 Pw(my_type{xy_rand(generator)}, my_type{xy_rand(generator)},
            my_type{z_rand(generator)});
    points.push_back(Pw);

    // 在每一帧上的观测量
    for (int i = 0; i < poseNums; ++i) {
      Vec3 Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
      Pc = Pc / Pc.z();  // 归一化图像平面
      Pc[0] += my_type{noise_pdf(generator)};
      Pc[1] += my_type{noise_pdf(generator)};
      cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
    }
  }
}

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */

int main(int argc, char *argv[]) {
  gDebugCol4() << G_FILE;

  std::vector<Frame> cameras;
  std::vector<Vec3> points;
  GetSimDataInWordFrame(cameras, points);
  Eigen::Quaternion<my_type> qic(my_type{1}, my_type{0}, my_type{0},
                                 my_type{0});
  Vec3 tic(my_type{0}, my_type{0}, my_type{0});

  // 构建 problem
  Problem problem(Problem::ProblemType::SLAM_PROBLEM);

  // 所有 Pose
  std::vector<std::shared_ptr<VertexPose> > vertexCams_vec;
  for (size_t i = 0; i < cameras.size(); ++i) {
    std::shared_ptr<VertexPose> vertexCam(new VertexPose());
    VecX pose(7);
    pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(),
        cameras[i].qwc.z(), cameras[i].qwc.w();  // 平移和四元数
    vertexCam->SetParameters(pose.cast<my_type>());

    //        if(i < 2)
    //            vertexCam->SetFixed();

    problem.AddVertex(vertexCam);
    vertexCams_vec.push_back(vertexCam);
  }

  // 所有 Point 及 edge
  std::default_random_engine generator;
  std::normal_distribution<double> noise_pdf(0, 1.);
  double noise = 0;
  std::vector<double> noise_invd;
  std::vector<std::shared_ptr<VertexInverseDepth> > allPoints;
  for (size_t i = 0; i < points.size(); ++i) {
    // 假设所有特征点的起始帧为第0帧， 逆深度容易得到
    Vec3 Pw = points[i];
    Vec3 Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
    noise = noise_pdf(generator);
    double inverse_depth =
        static_cast<double>(my_type{1.} / (Pc.z() + my_type{noise}));
    //        double inverse_depth = 1. / Pc.z();
    noise_invd.push_back(inverse_depth);

    // 初始化特征 vertex
    std::shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
    VecX inv_d(1);
    inv_d << my_type{inverse_depth};
    verterxPoint->SetParameters(inv_d);
    problem.AddVertex(verterxPoint);
    allPoints.push_back(verterxPoint);

    // 每个特征对应的投影误差, 第 0 帧为起始帧
    for (size_t j = 1; j < cameras.size(); ++j) {
      Vec3 pt_i = cameras[0].featurePerId.find(i)->second;
      Vec3 pt_j = cameras[j].featurePerId.find(i)->second;
      std::shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
      edge->SetTranslationImuFromCamera(qic, tic);

      std::vector<std::shared_ptr<Vertex> > edge_vertex;
      edge_vertex.push_back(verterxPoint);
      edge_vertex.push_back(vertexCams_vec[0]);
      edge_vertex.push_back(vertexCams_vec[j]);
      edge->SetVertex(edge_vertex);

      problem.AddEdge(edge);
    }
  }

  gDebug() << "begin";
  problem.Solve(5);  // 一共迭代5次

  std::cout << "\nCompare MonoBA results after opt..." << std::endl;
  for (size_t k = 0; k < allPoints.size(); k += 1) {
    std::cout << "after opt, point " << k << " : gt "
              << my_type{1.} / points[k].z() << " ,noise " << noise_invd[k]
              << " ,opt " << allPoints[k]->Parameters() << std::endl;
  }
  std::cout << "------------ pose translation ----------------" << std::endl;
  for (int i = 0; i < vertexCams_vec.size(); ++i) {
    std::cout << "translation after opt: " << i << " :"
              << vertexCams_vec[i]->Parameters().head(3).transpose()
              << " || gt: " << cameras[i].twc.transpose() << std::endl;
  }

  /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0.
  /// 说明向零空间发生了漂移。 解决办法： fix 第一帧和第二帧，固定 7 自由度。
  /// 或者加上非常大的先验值。
  problem.TestMarginalize();

  return 0;
}
