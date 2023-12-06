#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "fixed.hpp"
#include "ios.hpp"
#include "math.hpp"

using my_double = double;
using my_float = float;

#define MY_TYPE 0
#if !MY_TYPE
using my_type = double;
#else
// using my_type = fpm::fixed<std::int32_t, std::int64_t, 12>;

// 当前的定点数的库实现不了64bit的，也就是Base的类型需要128或更大，这里就需要__int128_t仅支持gcc
// Ref：https://github.com/MikeLankamp/fpm/issues/57
// 
using my_type = fpm::fixed<std::int64_t, __int128_t, 20>;
// using my_type = fpm::fixed_16_16;
// fpm::fixed_16_16 x;
#endif

using MatXX = Eigen::Matrix<my_type, Eigen::Dynamic, Eigen::Dynamic>;
using Mat1010 = Eigen::Matrix<my_type, 10, 10>;
using Mat1313 = Eigen::Matrix<my_type, 13, 13>;
using Mat810 = Eigen::Matrix<my_type, 8, 10>;
using Mat83 = Eigen::Matrix<my_type, 8, 3>;
using Mat66 = Eigen::Matrix<my_type, 6, 6>;
using Mat53 = Eigen::Matrix<my_type, 5, 3>;
using Mat43 = Eigen::Matrix<my_type, 4, 3>;
using Mat42 = Eigen::Matrix<my_type, 4, 2>;
using Mat33 = Eigen::Matrix<my_type, 3, 3>;
using Mat22 = Eigen::Matrix<my_type, 2, 2>;
using Mat23 = Eigen::Matrix<my_type, 2, 3>;
using Mat88 = Eigen::Matrix<my_type, 8, 8>;
using Mat77 = Eigen::Matrix<my_type, 7, 7>;
using Mat49 = Eigen::Matrix<my_type, 4, 9>;
using Mat89 = Eigen::Matrix<my_type, 8, 9>;
using Mat94 = Eigen::Matrix<my_type, 9, 4>;
using Mat98 = Eigen::Matrix<my_type, 9, 8>;
using Mat99 = Eigen::Matrix<my_type, 9, 9>;
using Mat96 = Eigen::Matrix<my_type, 9, 6>;
using Mat81 = Eigen::Matrix<my_type, 8, 1>;
using Mat18 = Eigen::Matrix<my_type, 1, 8>;
using Mat91 = Eigen::Matrix<my_type, 9, 1>;
using Mat19 = Eigen::Matrix<my_type, 1, 9>;
using Mat84 = Eigen::Matrix<my_type, 8, 4>;
using Mat48 = Eigen::Matrix<my_type, 4, 8>;
using Mat44 = Eigen::Matrix<my_type, 4, 4>;
using Mat1414 = Eigen::Matrix<my_type, 14, 14>;
using Mat1515 = Eigen::Matrix<my_type, 15, 15>;

// float matricies
using Mat33f = Eigen::Matrix<float, 3, 3>;
using Mat103f = Eigen::Matrix<float, 10, 3>;
using Mat22f = Eigen::Matrix<float, 2, 2>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Mat18f = Eigen::Matrix<float, 1, 8>;
using Mat66f = Eigen::Matrix<float, 6, 6>;
using Mat88f = Eigen::Matrix<float, 8, 8>;
using Mat84f = Eigen::Matrix<float, 8, 4>;
using Mat66f = Eigen::Matrix<float, 6, 6>;
using Mat44f = Eigen::Matrix<float, 4, 4>;
using Mat1212f = Eigen::Matrix<float, 12, 12>;
using Mat1313f = Eigen::Matrix<float, 13, 13>;
using Mat1010f = Eigen::Matrix<float, 10, 10>;
using Mat99f = Eigen::Matrix<float, 9, 9>;
using Mat42f = Eigen::Matrix<float, 4, 2>;
using Mat62f = Eigen::Matrix<float, 6, 2>;
using Mat12f = Eigen::Matrix<float, 1, 2>;
using MatXXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Mat1414f = Eigen::Matrix<float, 14, 14>;

// double vectors
using Vec15 = Eigen::Matrix<my_type, 15, 1>;
using Vec14 = Eigen::Matrix<my_type, 14, 1>;
using Vec13 = Eigen::Matrix<my_type, 13, 1>;
using Vec10 = Eigen::Matrix<my_type, 10, 1>;
using Vec9 = Eigen::Matrix<my_type, 9, 1>;
using Vec8 = Eigen::Matrix<my_type, 8, 1>;
using Vec7 = Eigen::Matrix<my_type, 7, 1>;
using Vec6 = Eigen::Matrix<my_type, 6, 1>;
using Vec5 = Eigen::Matrix<my_type, 5, 1>;
using Vec4 = Eigen::Matrix<my_type, 4, 1>;
using Vec3 = Eigen::Matrix<my_type, 3, 1>;
using Vec2 = Eigen::Matrix<my_type, 2, 1>;
using Vec1 = Eigen::Matrix<my_type, 1, 1>;
using VecX = Eigen::Matrix<my_type, Eigen::Dynamic, 1>;

// float vectors
using Vec12f = Eigen::Matrix<float, 12, 1>;
using Vec8f = Eigen::Matrix<float, 8, 1>;
using Vec10f = Eigen::Matrix<float, 10, 1>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec12f = Eigen::Matrix<float, 12, 1>;
using Vec13f = Eigen::Matrix<float, 13, 1>;
using Vec9f = Eigen::Matrix<float, 9, 1>;
using VecXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using Vec14f = Eigen::Matrix<float, 14, 1>;

// Quaternions
using Qd = Eigen::Quaternion<my_type>;
using Qf = Eigen::Quaternion<float>;

// Vector of Eigen vectors
using VecVec2 = std::vector<Vec2, Eigen::aligned_allocator<Vec2>>;
using VecVec3 = std::vector<Vec3, Eigen::aligned_allocator<Vec3>>;
using VecVec2f = std::vector<Vec2f, Eigen::aligned_allocator<Vec2f>>;
using VecVec3f = std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>>;

// Map of Eigen matrix
using MapMatXX = std::map<unsigned long, MatXX, std::less<unsigned long>,
                          Eigen::aligned_allocator<MatXX>>;
