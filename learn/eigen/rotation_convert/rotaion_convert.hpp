#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

#include "common.h"

namespace Eigen {

// clang-format off

inline Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::AngleAxisd& axis_angle) {
  Eigen::AngleAxisd rotation(axis_angle);
  return rotation.toRotationMatrix();
}
inline Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd rotation(angle, axis);
  return rotation.toRotationMatrix();
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::AngleAxisd& axis_angle) {
  Eigen::Matrix3d rotation_matrix = axis_angle.toRotationMatrix();
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
  return euler_angles;
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd axis_angle(angle, axis);
  Eigen::Matrix3d rotation_matrix = axis_angle.toRotationMatrix();
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
  return euler_angles;
}
inline Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::AngleAxisd& axis_angle) {
  return Eigen::Quaterniond(axis_angle);
}
inline Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& axis, double angle) {
  Eigen::AngleAxisd rotation(angle, axis);
  return Eigen::Quaterniond(rotation);
}


inline Eigen::Vector3d RotationMatrixToEuler(const Eigen::Matrix3d& rotationMatrix) {
  return rotationMatrix.eulerAngles(2, 1, 0);
}
inline Eigen::Quaterniond RotationMatrixToQuaternion(const Eigen::Matrix3d& rotationMatrix) {
  return Eigen::Quaterniond(rotationMatrix);
}
inline Eigen::AngleAxisd RotationMatrixToAxisAngle(const Eigen::Matrix3d& rotation_matrix) {
  Eigen::AngleAxisd axis_angle(rotation_matrix);
  return axis_angle;
}

inline Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Quaterniond& quaternion) {
  return quaternion.toRotationMatrix();
}
inline Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond& quaternion) {
  return quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
}
inline Eigen::AngleAxisd QuaternionToAxisAngle(const Eigen::Quaterniond& quaternion) {
  return Eigen::AngleAxisd(quaternion);
}


inline Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& euler) {
  Eigen::Quaterniond quaternion;
  quaternion = Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX());
  return quaternion;
}
inline Eigen::Matrix3d EulerToRotationMatrix(const Eigen::Vector3d& euler) {
  return EulerToQuaternion(euler).toRotationMatrix();
}
inline Eigen::AngleAxisd EulerToAxisAngle(const Eigen::Vector3d& euler) {
  return Eigen::AngleAxisd(EulerToQuaternion(euler));
}

// clang-format on

}  // namespace Eigen

// clang-format off
namespace gxt {

namespace {
inline Eigen::Matrix3d VectorHat(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix=Eigen::Matrix3d::Zero();
  matrix(0,1)=-vector(2);
  matrix(1,0)=vector(2);
  matrix(0,2)=vector(1);
  matrix(2,0)=-vector(1);
  matrix(1,2)=-vector(0);
  matrix(2,1)=vector(0);
  return matrix;
}
}

inline Eigen::Vector3d RotationMatrixToEuler(const Eigen::Matrix3d& rotation_matrix) {
  Eigen::Vector3d ret;
  ret(0)=std::atan2(rotation_matrix(2,1),rotation_matrix(2,2));
  ret(1)=std::asin(-rotation_matrix(2,0));
  ret(2)=std::atan2(rotation_matrix(1,0),rotation_matrix(0,0));
  return ret;
}

inline Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::AngleAxisd& axis_angle) {
  Eigen::Vector3d axis = axis_angle.axis();
  double angle = axis_angle.angle();
  Eigen::Matrix3d ret;
  ret = Eigen::Matrix3d::Identity() + std::sin(angle) * VectorHat(axis) +
        (1 - std::cos(angle)) * VectorHat(axis) * VectorHat(axis);
  return ret;
}
inline Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::Vector3d& axis, double angle) {
  Eigen::Matrix3d ret;
  ret = Eigen::Matrix3d::Identity() + std::sin(angle) * VectorHat(axis) +
        (1 - std::cos(angle)) * VectorHat(axis) * VectorHat(axis);
  return ret;
}
inline Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::AngleAxisd& axis_angle) {
  Eigen::Vector3d axis = axis_angle.axis();
  double angle = axis_angle.angle();
  Eigen::Quaterniond ret;
  ret.w() = std::cos(angle / 2);
  ret.x() = std::sin(angle / 2) * axis(0);
  ret.y() = std::sin(angle / 2) * axis(1);
  ret.z() = std::sin(angle / 2) * axis(2);
  return ret;
}
inline Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& axis,double angle) {
  Eigen::Quaterniond ret;
  ret.w() = std::cos(angle / 2);
  ret.x() = std::sin(angle / 2) * axis(0);
  ret.y() = std::sin(angle / 2) * axis(1);
  ret.z() = std::sin(angle / 2) * axis(2);
  return ret;
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::AngleAxisd& axis_angle) {
  return gxt::RotationMatrixToEuler(gxt::AxisAngleToRotationMatrix(axis_angle));
}
inline Eigen::Vector3d AxisAngleToEuler(const Eigen::Vector3d& axis,double angle) {
  return gxt::RotationMatrixToEuler(gxt::AxisAngleToRotationMatrix(axis,angle));
}


inline Eigen::AngleAxisd RotationMatrixToAxisAngle(const Eigen::Matrix3d& rotation_matrix) {
  Eigen::AngleAxisd ret;
  double tr_R=rotation_matrix(0,0)+rotation_matrix(1,1)+rotation_matrix(2,2);
  ret.angle() = std::acos((tr_R - 1) / 2);
  ret.axis()(0) = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (2 * std::sin(ret.angle()));
  ret.axis()(1) = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (2 * std::sin(ret.angle()));
  ret.axis()(2) = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (2 * std::sin(ret.angle()));
  return ret;
}
inline Eigen::Quaterniond RotationMatrixToQuaternion(const Eigen::Matrix3d& rotation_matrix) {
  Eigen::Quaterniond ret;
  double tr_R=rotation_matrix(0,0)+rotation_matrix(1,1)+rotation_matrix(2,2);
  ret.w()=std::sqrt((1+tr_R)/4);
  ret.x()=(rotation_matrix(2,1)-rotation_matrix(1,2))/(4*ret.w());
  ret.y()=(rotation_matrix(0,2)-rotation_matrix(2,0))/(4*ret.w());
  ret.z()=(rotation_matrix(1,0)-rotation_matrix(0,1))/(4*ret.w());
  return ret;
}


inline Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Quaterniond& quaternion) {
  Eigen::Matrix3d ret=Eigen::Matrix3d::Zero();
  double a=quaternion.w();
  double b=quaternion.x();
  double c=quaternion.y();
  double d=quaternion.z();
  ret << 
    1-2*c*c-2*d*d, 2*b*c-2*a*d,   2*a*c+2*b*d,
    2*b*c+2*a*d,   1-2*b*b-2*d*d, 2*c*d-2*a*b,
    2*b*d-2*a*c,   2*a*b+2*c*d,   1-2*b*b-2*c*c ;
  return ret;
}
inline Eigen::AngleAxisd QuaternionToAxisAngle(const Eigen::Quaterniond& quaternion) {
  Eigen::AngleAxisd ret;
  double a=quaternion.w();
  double b=quaternion.x();
  double c=quaternion.y();
  double d=quaternion.z();
  double angle=2*std::acos(a);
  ret.angle()=angle;
  double axis_x=b/std::sin(angle/2);
  double axis_y=c/std::sin(angle/2);
  double axis_z=d/std::sin(angle/2);
  ret.axis()(0)=axis_x;
  ret.axis()(1)=axis_y;
  ret.axis()(2)=axis_z;
  return ret;
}
inline Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond& quaternion) {
  Eigen::Vector3d ret;
  double w=quaternion.w();
  double x=quaternion.x();
  double y=quaternion.y();
  double z=quaternion.z();

  ret(0)=std::atan2(2*(w*x+y*z),1-2*(x*x+y*y));
  ret(1)=-M_PI/2+2*std::atan2(std::sqrt(1+2*(w*y-x*z)),std::sqrt(1-2*(w*y-x*z)));
  ret(2)=std::atan2(2*(w*z+x*y),1-2*(y*y+z*z));
  return ret;
}

inline Eigen::Matrix3d EulerToRotationMatrix(const Eigen::Vector3d& euler) {
  Eigen::Matrix3d ret=Eigen::Matrix3d::Zero();
  Eigen::Matrix3d R_x=Eigen::Matrix3d::Zero();
  Eigen::Matrix3d R_y=Eigen::Matrix3d::Zero();
  Eigen::Matrix3d R_z=Eigen::Matrix3d::Zero();

  double x=euler(0);
  double y=euler(1);
  double z=euler(2);

  R_x(0,0)=1;
  R_x(1,1)=std::cos(x);
  R_x(1,2)=-std::sin(x);
  R_x(2,1)=std::sin(x);
  R_x(2,2)=std::cos(x);

  R_y(0,0)=std::cos(y);
  R_y(0,2)=std::sin(y);
  R_y(1,1)=1;
  R_y(2,0)=-std::sin(y);
  R_y(2,2)=std::cos(y);

  R_z(0,0)=std::cos(z);
  R_z(0,1)=-std::sin(z);
  R_z(1,0)=std::sin(z);
  R_z(1,1)=std::cos(z);
  R_z(2,2)=1;

  ret=R_z*R_y*R_x;
  return ret;
}
inline Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& euler) {
  Eigen::Quaterniond quaternion;
  double x=euler(0);
  double y=euler(1);
  double z=euler(2);

  double cx=std::cos(x/2); double cy=std::cos(y/2); double cz=std::cos(z/2);
  double sx=std::sin(x/2); double sy=std::sin(y/2); double sz=std::sin(z/2);
  quaternion.w()=cx*cy*cz+sx*sy*sz;
  quaternion.x()=sx*cy*cz-cx*sy*sz;
  quaternion.y()=cx*sy*cz+sx*cy*sz;
  quaternion.z()=cx*cy*sz-sx*sy*cz;

  return quaternion;
}
inline Eigen::AngleAxisd EulerToAxisAngle(const Eigen::Vector3d& euler) {
  return gxt::QuaternionToAxisAngle(gxt::EulerToQuaternion(euler));
}

}  // namespace gxt

// clang-format on
