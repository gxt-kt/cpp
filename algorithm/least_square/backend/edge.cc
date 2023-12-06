#include "edge.h"

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string>& verticies_types) {
  residual_.resize(residual_dimension, 1);

  //   TODO::
  //  这里可能会存在问题，比如这里resize了3个空,后续调用edge->addVertex.
  //  使得vertex前面会存在空元素
  //  verticies_.resize(num_verticies);

  if (!verticies_types.empty()) {
    verticies_types_ = verticies_types;
  }
  jacobians_.resize(num_verticies);
  id_ = global_edge_id++;

  MatXX information(residual_dimension, residual_dimension);
  information.setIdentity();
  information_ = information;

  //    cout<<"Edge construct residual_dimension="<<residual_dimension
  //            << ", num_verticies="<<num_verticies<<", id_="<<id_<<endl;
}

double Edge::Chi2() {
  // TODO::  we should not Multiply information here, because we have computed
  // Jacobian = sqrt_info * Jacobian

  // TODO: gxt
  // 因为这里乘出来的数都很大，所以肯定不能再用定点数了,先用浮点数做一下替换吧
  //
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> info;
  info.resize(information_.rows(), information_.cols());
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> resi;
  resi.resize(residual_.rows(), residual_.cols());

  for (int i = 0; i < resi.rows(); i++) {
    resi(i, 0) = static_cast<double>(residual_(i, 0));
  }
  for (int i = 0; i < info.rows(); i++) {
    for (int j = 0; j < info.rows(); j++) {
      info(i, j) = static_cast<double>(information_(i, j));
    }
  }
  double res{0};
  res=(resi.transpose() * info * resi)(0,0);

  // gDebug(residual_.transpose());
  // gDebug(information_);
  // gDebug(residual_.transpose() * information_ * residual_);
  // gDebug(res);
  return res;

  // return static_cast<double>(
  //     (residual_.transpose() * information_ * residual_)(0, 0));

  //    return residual_.transpose() * residual_;   // 当计算 residual
  //    的时候已经乘以了 sqrt_info, 这里不要再乘
}

bool Edge::CheckValid() {
  if (!verticies_types_.empty()) {
    // check type info
    for (size_t i = 0; i < verticies_.size(); ++i) {
      if (verticies_types_.at(i) != verticies_.at(i)->TypeInfo()) {
        std::cout << "Vertex type does not match, should be "
                  << verticies_types_[i] << ", but set to "
                  << verticies_[i]->TypeInfo() << std::endl;
        return false;
      }
    }
  }
  assert(information_.rows() == information_.cols());
  assert(residual_.rows() == information_.rows());
  assert(residual_.rows() == observation_.cols());
  // check jacobians
  for (size_t i = 0; i < jacobians_.size(); ++i) {
    assert(jacobians_.at(i).rows() == residual_.rows());
    assert(jacobians_.at(i).cols() == verticies_.at(i)->LocalDimension());
  }
  return true;
}
