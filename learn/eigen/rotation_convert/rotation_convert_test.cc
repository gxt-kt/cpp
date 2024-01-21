#include "common.h"
#include "rotaion_convert.hpp"


// clang-format off

// Axis Angle
Eigen::Vector3d axis_temp_(1, 2, 3);
Eigen::Vector3d axis=axis_temp_.normalized();
double angle = M_PI / 2;
Eigen::AngleAxisd axis_angle(angle, axis);

// Rotation Matrix
Eigen::Matrix3d rotation_matrix = (Eigen::Matrix3d() << 
                                   0.0714286, -0.658927, 0.748808,
                                   0.944641,  0.285714,  0.16131,
                                   -0.320237, 0.695833,  0.642857
                                  ).finished();

// Quaternion
Eigen::Quaterniond quaternion(0.707107, 0.188982, 0.377964, 0.566947);

// Euler Angle 
// 我们定义的欧拉角顺序是外旋定轴RPY，就是先转X，再转Y，再转Z
// Ref: https://stackoverflow.com/questions/54125208/eigen-eulerangles-returns-incorrect-values
// euler(0)是z轴的值
// euler(1)是y轴的值
// euler(2)是x轴的值
Eigen::Vector3d euler(1.49533, 0.32598, 0.82495);
// euler_tran(0)是x轴的值
// euler_tran(1)是y轴的值
// euler_tran(2)是z轴的值
Eigen::Vector3d euler_tran(0.82495, 0.32598, 1.49533);
// 注意区分以下代码中的euler和euler_tran


// clang-format on

// template <typename Derived>
// void Assert(const Eigen::MatrixBase<Derived>& matrix1,const Eigen::MatrixBase<Derived>& matrix2) {
//   assert(matrix1.isApprox(matrix2,1e-8));
// }

#define ASSERT(matrix1, matrix2) assert(matrix1.isApprox(matrix2, 1e-4));

int main(int argc, char* argv[]) {
  // clang-format off
  gDebug(quaternion);
  gDebug(quaternion.coeffs().transpose());
  gDebug(rotation_matrix);
  gDebug(axis_angle.matrix());
  gDebug(euler.transpose());
  gDebug(euler_tran.transpose());
  gDebug(rotation_matrix.eulerAngles(2,1,0));

  ASSERT(rotation_matrix,quaternion.matrix());

  {
    gDebugCol2() << G_SPLIT_LINE;
    Eigen::Matrix3d rotation_matrix1 = Eigen::AxisAngleToRotationMatrix(axis, angle);
    Eigen::Matrix3d rotation_matrix2 = Eigen::AxisAngleToRotationMatrix(axis_angle);
    gDebug(rotation_matrix1);
    gDebug(rotation_matrix2);
    ASSERT(rotation_matrix,rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix2);

    Eigen::Quaterniond quaternion1 = Eigen::AxisAngleToQuaternion(axis, angle);
    Eigen::Quaterniond quaternion2 = Eigen::AxisAngleToQuaternion(axis_angle);
    gDebug(quaternion1);
    gDebug(quaternion2);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());
    ASSERT(quaternion.coeffs(),quaternion2.coeffs());

    Eigen::Vector3d euler1 = Eigen::AxisAngleToEuler(axis, angle);
    Eigen::Vector3d euler2 = Eigen::AxisAngleToEuler(axis_angle);
    gDebug(euler1);
    gDebug(euler2);
    ASSERT(euler,euler1);
    ASSERT(euler,euler2);
  }
  {
    gDebugCol2() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = Eigen::RotationMatrixToAxisAngle(rotation_matrix);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Quaterniond quaternion1 = Eigen::RotationMatrixToQuaternion(rotation_matrix);
    gDebug(quaternion1);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());

    Eigen::Vector3d euler1 = Eigen::RotationMatrixToEuler(rotation_matrix);
    gDebug(euler1);
    ASSERT(euler,euler1);
  }
  {
    gDebugCol2() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = Eigen::QuaternionToAxisAngle(quaternion);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Matrix3d rotation_matrix1 = Eigen::QuaternionToRotationMatrix(quaternion);
    gDebug(rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix1);

    Eigen::Vector3d euler1 = Eigen::QuaternionToEuler(quaternion);
    gDebug(euler1);
    ASSERT(euler,euler1);
  }
  {
    gDebugCol2() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = Eigen::EulerToAxisAngle(euler_tran);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Matrix3d rotation_matrix1 = Eigen::EulerToRotationMatrix(euler_tran);
    gDebug(rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix1);

    Eigen::Quaterniond quaternion1 = Eigen::EulerToQuaternion(euler_tran);
    gDebug(quaternion1);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());
  }
  

  {
    gDebugCol4() << G_SPLIT_LINE;
    Eigen::Matrix3d rotation_matrix1 = gxt::AxisAngleToRotationMatrix(axis, angle);
    Eigen::Matrix3d rotation_matrix2 = gxt::AxisAngleToRotationMatrix(axis_angle);
    gDebug(rotation_matrix1);
    gDebug(rotation_matrix2);
    ASSERT(rotation_matrix,rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix2);

    Eigen::Quaterniond quaternion1 = gxt::AxisAngleToQuaternion(axis, angle);
    Eigen::Quaterniond quaternion2 = gxt::AxisAngleToQuaternion(axis_angle);
    gDebug(quaternion1);
    gDebug(quaternion2);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());
    ASSERT(quaternion.coeffs(),quaternion2.coeffs());

    Eigen::Vector3d euler1 = gxt::AxisAngleToEuler(axis, angle);
    Eigen::Vector3d euler2 = gxt::AxisAngleToEuler(axis_angle);
    gDebug(euler1);
    gDebug(euler2);
    ASSERT(euler_tran,euler1);
    ASSERT(euler_tran,euler2);
  }
  {
    gDebugCol4() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = gxt::RotationMatrixToAxisAngle(rotation_matrix);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Quaterniond quaternion1 = gxt::RotationMatrixToQuaternion(rotation_matrix);
    gDebug(quaternion1);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());

    Eigen::Vector3d euler1 = gxt::RotationMatrixToEuler(rotation_matrix);
    gDebug(euler1);
    ASSERT(euler_tran,euler1);
  }
  {
    gDebugCol4() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = gxt::QuaternionToAxisAngle(quaternion);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Matrix3d rotation_matrix1 = gxt::QuaternionToRotationMatrix(quaternion);
    gDebug(rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix1);

    Eigen::Vector3d euler1 = gxt::QuaternionToEuler(quaternion);
    gDebug(euler1);
    ASSERT(euler_tran,euler1);
  }
  {
    gDebugCol4() << G_SPLIT_LINE;
    Eigen::AngleAxisd axis_angle1 = gxt::EulerToAxisAngle(euler_tran);
    gDebug(axis_angle1.matrix());
    ASSERT(axis_angle.matrix(),axis_angle1.matrix());

    Eigen::Matrix3d rotation_matrix1 = gxt::EulerToRotationMatrix(euler_tran);
    gDebug(rotation_matrix1);
    ASSERT(rotation_matrix,rotation_matrix1);

    Eigen::Quaterniond quaternion1 = gxt::EulerToQuaternion(euler_tran);
    gDebug(quaternion1);
    ASSERT(quaternion.coeffs(),quaternion1.coeffs());
  }

  // clang-format on
  return 0;
}
