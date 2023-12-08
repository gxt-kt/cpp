#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "common.h"

// template<typename _Scalar, int _Rows, int _Cols>
// gxt::DebugStream& operator<<(gxt::DebugStream& os, const
// Eigen::Matrix<_Scalar,_Rows,_Cols> matrix) {
//   gDebug() << "???";
//   for(int i=0;i<_Rows;i++) {
//     for(int j=0;j<_Cols;j++) {
//       // os << matrix(i,j);
//     }
//     os << gxt::endl;
//   }
//   return os;
// }

// template <typename BaseType, typename IntermediateType, unsigned int
// FractionBits, bool EnableRounding = true> gxt::DebugStream&
// operator<<(gxt::DebugStream& os, const
// fpm::fixed<BaseType,IntermediateType,FractionBits> n) {
//   os << static_cast<double>(n);
//   return os;
// }

template <std::size_t N_INT, std::size_t N_FRAC>
class FixedPointNumber {
 private:
  using IntType = std::int64_t;  // 内部整数类型

  static constexpr IntType SCALE = static_cast<IntType>(1)
                                   << N_FRAC;  // 缩放因子

  IntType value_;

 public:
  // 构造函数
  FixedPointNumber() : value_(0) {}
  FixedPointNumber(IntType value) : value_(value * SCALE) {}
  FixedPointNumber(double value)
      : value_(static_cast<IntType>(value * SCALE)) {}

  // 获取定点数的整数部分
  IntType integerPart() const { return value_ / SCALE; }

  // 获取定点数的小数部分
  double fractionalPart() const {
    return static_cast<double>(value_ % SCALE) / SCALE;
  }

  // 加法运算
  FixedPointNumber operator+(const FixedPointNumber& other) const {
    return FixedPointNumber(value_ + other.value_);
  }

  // 减法运算
  FixedPointNumber operator-(const FixedPointNumber& other) const {
    return FixedPointNumber(value_ - other.value_);
  }

  // 乘法运算
  FixedPointNumber operator*(const FixedPointNumber& other) const {
    return FixedPointNumber((value_ * other.value_) / SCALE);
  }

  // 除法运算
  FixedPointNumber operator/(const FixedPointNumber& other) const {
    return FixedPointNumber((value_ * SCALE) / other.value_);
  }

  // 输出定点数的值
  friend std::ostream& operator<<(std::ostream& os,
                                  const FixedPointNumber& number) {
    os << number.integerPart() << "." << number.fractionalPart();
    return os;
  }
};

int main(int argc, char* argv[]) {
  gDebugCol3() << G_FILE;
  FixedPointNumber<8, 8> a(1.25);
  FixedPointNumber<8, 8> b(2.5);

  FixedPointNumber<8, 8> sum = a + b;
  FixedPointNumber<8, 8> difference = a - b;
  FixedPointNumber<8, 8> product = a * b;
  FixedPointNumber<8, 8> quotient = a / b;

  std::cout << "Sum: " << sum << std::endl;
  std::cout << "Difference: " << difference << std::endl;
  std::cout << "Product: " << product << std::endl;
  std::cout << "Quotient: " << quotient << std::endl;
}

