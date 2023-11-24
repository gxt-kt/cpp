#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>


#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "fpm/math.hpp"

#include "common.h"


// template<typename _Scalar, int _Rows, int _Cols>
// gxt::DebugStream& operator<<(gxt::DebugStream& os, const Eigen::Matrix<_Scalar,_Rows,_Cols> matrix) {
//   gDebug() << "???";
//   for(int i=0;i<_Rows;i++) {
//     for(int j=0;j<_Cols;j++) {
//       // os << matrix(i,j);
//     }
//     os << gxt::endl;
//   }
//   return os;
// }


// template <typename BaseType, typename IntermediateType, unsigned int FractionBits, bool EnableRounding = true>
// gxt::DebugStream& operator<<(gxt::DebugStream& os, const fpm::fixed<BaseType,IntermediateType,FractionBits> n) {
//   os << static_cast<double>(n);
//   return os;
// }

int main(int argc, char* argv[]) {
  gDebug() << G_FILE;

  {
    fpm::fixed_16_16 x;
    fpm::fixed<std::int32_t, std::int64_t, 16> a(0.11);
    fpm::fixed<std::int32_t, std::int64_t, 16> b(0.14);

    gDebug(static_cast<double>(a + b));
    gDebug(static_cast<double>(a - b));
    gDebug(static_cast<double>(a * b));
    gDebug(static_cast<double>(a / b));
  }
  {
    Eigen::Matrix<double, 2, 2> a;
    a << 1, 2, 3, 4;
    Eigen::Matrix<double, 2, 2> b;
    b << 1, 2, 3, 4;
    auto c = a * b;
    gDebug(c);
  }
  {
    gDebugCol2() << G_SPLIT_LINE;
    using f16 = fpm::fixed_16_16;
    Eigen::Matrix<f16, 2, 2> a;
    a(0, 0) = f16(1);
    a(0, 1) = f16(2);
    a(1, 0) = f16(3);
    a(1, 1) = f16(4);
    Eigen::Matrix<f16, 2, 2> b=a;
    // b(0, 0) = f16(1);
    // b(0, 1) = f16(2);
    // b(1, 0) = f16(3);
    // b(1, 1) = f16(4);
    // a<<1,2,3,4;
    // Eigen::Matrix<double,2,2> b;
    // b<<1,2,3,4;
    // auto c=a*b;
    Eigen::Matrix<f16,2,2> c=a*b;
    // auto c=a*b;
    gDebug(TYPE(c));

    std::cout << c(0,0) << std::endl;
    std::cout << c << std::endl;
    gDebugCol1() << c(0,0);
    gDebugCol1(c);
  }
}
