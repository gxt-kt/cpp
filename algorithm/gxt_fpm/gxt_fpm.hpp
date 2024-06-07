#pragma once

#include <type_traits>

#include "common.h"

namespace gxt {

// clang-format off
//! Fixed-point number type
//! \tparam BaseType         the base integer type used to store the fixed-point number. This can be a signed or unsigned type.
//! \tparam IntermediateType the integer type used to store intermediate results during calculations.
//! \tparam FractionBits     the number of bits of the BaseType used to store the fraction
//! \tparam EnableRounding   enable rounding of LSB for multiplication, division, and type conversion
// clang-format on
template <typename BaseType, typename IntermediateType,
          unsigned int FractionBits, bool EnableRounding = true>
class FixN {
 public:
  FixN() { gDebug("construct NULL"); }
  // Converts an integral number to the fixed-point type.
  // Like static_cast, this truncates bits that don't fit.
  template <typename T, typename std::enable_if<
                            std::is_integral<T>::value>::type* = nullptr>
  FixN(T value) {
    gDebug("integral");
  }
  // construct for floating number
  template <typename T, typename std::enable_if<
                            std::is_floating_point<T>::value>::type* = nullptr>
  FixN(T value) {
    gDebug("floating");
  }

 private:
  BaseType value_;
};

}  // namespace gxt
