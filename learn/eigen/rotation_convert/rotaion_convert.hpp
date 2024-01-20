#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"

namespace Eigen {

// clang-format off

inline Eigen::Matrix3d AxisAngleToRotationMatrix( const Eigen::AngleAxisd& axis_angle) {
  Eigen::AngleAxisd rotation(axis_angle);
  return rotation.toRotationMatrix();
}
inline Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd rotation(angle, axis);
  return rotation.toRotationMatrix();
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::AngleAxisd& axis_angle) {
  Eigen::Matrix3d rotation_matrix = axis_angle.toRotationMatrix();
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(0, 1, 2);
  return euler_angles;
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd axis_angle(angle, axis);
  Eigen::Matrix3d rotation_matrix = axis_angle.toRotationMatrix();
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(0, 1, 2);
  return euler_angles;
}
inline Eigen::Quaterniond AxisAngleToQuaternion( const Eigen::AngleAxisd& axis_angle) {
  return Eigen::Quaterniond(axis_angle);
}
inline Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd rotation(angle, axis);
  return Eigen::Quaterniond(rotation);
}


inline Eigen::Vector3d RotationMatrixToEulerAngles( const Eigen::Matrix3d& rotationMatrix) {
  return rotationMatrix.eulerAngles(0, 1, 2);
}
inline Eigen::Quaterniond RotationMatrixToQuaternion( const Eigen::Matrix3d& rotationMatrix) {
  return Eigen::Quaterniond(rotationMatrix);
}
inline Eigen::AngleAxisd RotationMatrixToAxisAngle( const Eigen::Matrix3d& rotation_matrix) {
  Eigen::AngleAxisd axis_angle(rotation_matrix);
  return axis_angle;
}

inline Eigen::Matrix3d QuaternionToRotationMatrix( const Eigen::Quaterniond& quaternion) {
  return quaternion.toRotationMatrix();
}
inline Eigen::Vector3d QuaternionToEulerAngles( const Eigen::Quaterniond& quaternion) {
  return quaternion.toRotationMatrix().eulerAngles(0, 1, 2);
}
inline Eigen::AngleAxisd QuaternionToAxisAngle( const Eigen::Quaterniond& quaternion) {
  return Eigen::AngleAxisd(quaternion);
}


inline Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& euler) {
  Eigen::Quaterniond quaternion;
  quaternion = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX()) *
               Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ());
  return quaternion;
}
inline Eigen::Matrix3d EulerToRotationMatrix( const Eigen::Vector3d& euler) {
  return EulerToQuaternion(euler).toRotationMatrix();
}
inline Eigen::AngleAxisd EulerToAxisAngle(const Eigen::Vector3d& euler) {
  return Eigen::AngleAxisd(EulerToQuaternion(euler));
}

// clang-format on

}  // namespace Eigen
