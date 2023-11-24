#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>


#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "fpm/math.hpp"


#include "ceres/ceres.h"

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


// 比如下面这个问题要求的是
// 拟合曲线 y=exp{a*x*x+b*x+c} // 其中a=1,b=2,c=1

std::vector<double> data = {0.0,  3.1225102616837117, 0.01, 2.9114834899554345,
                            0.02, 2.431764205031237,  0.03, 3.1517178815159252,
                            0.04, 3.1762813582094616, 0.05, 2.8494401928967186,
                            0.06, 2.2004302658800383, 0.07, 2.9394803587703695,
                            0.08, 3.563335745410244,  0.09, 2.8082429257631207,
                            0.1,  2.8839318743485998, 0.11, 4.197733094121994,
                            0.12, 3.5074913527685956, 0.13, 3.643307179737193,
                            0.14, 3.2734661218318073, 0.15, 3.9197074802912097,
                            0.16, 3.5292172278774294, 0.17, 3.711967173999904,
                            0.18, 3.5884516240959403, 0.19, 5.120739360872787,
                            0.2,  4.586269912434171,  0.21, 4.076542642286447,
                            0.22, 5.537563925214148,  0.23, 5.916571261307799,
                            0.24, 6.834219653028663,  0.25, 5.0634570719299985,
                            0.26, 4.188784725539535,  0.27, 4.667852531748672,
                            0.28, 6.044226909544067,  0.29, 3.823608014703326,
                            0.3,  4.423102279661729,  0.31, 7.059830979641762,
                            0.32, 6.07789805037402,   0.33, 7.1846289878078045,
                            0.34, 4.228787587609053,  0.35, 5.879087564344235,
                            0.36, 5.955476766951016,  0.37, 7.457527934375881,
                            0.38, 6.860280359965479,  0.39, 5.713628886049559,
                            0.4,  6.531558397442931,  0.41, 5.993284511623399,
                            0.42, 6.460377605528751,  0.43, 5.9084845716605425,
                            0.44, 6.47307841922458,   0.45, 8.315884904932926,
                            0.46, 8.24899605439105,   0.47, 8.859371877240449,
                            0.48, 8.955379417658612,  0.49, 8.57563583706674,
                            0.5,  10.43249605747935,  0.51, 7.776152096476593,
                            0.52, 9.745005891837716,  0.53, 9.87894292180396,
                            0.54, 10.197103555786224, 0.55, 11.90211320873156,
                            0.56, 12.389501054209928, 0.57, 12.253244251582299,
                            0.58, 11.327561160315943, 0.59, 12.128062924218886,
                            0.6,  13.096323725221238, 0.61, 13.756307331577789,
                            0.62, 13.685055564960154, 0.63, 14.13606399960257,
                            0.64, 14.294747711487231, 0.65, 15.79683396947187,
                            0.66, 14.15895033046316,  0.67, 15.94284443587406,
                            0.68, 18.01802958664646,  0.69, 16.622474568994644,
                            0.7,  20.568982597423926, 0.71, 18.48452914509631,
                            0.72, 18.76802129471154,  0.73, 19.41201103060666,
                            0.74, 20.025525471248393, 0.75, 21.44443839104401,
                            0.76, 23.227686720666682, 0.77, 22.88564157408132,
                            0.78, 23.772537857952468, 0.79, 24.332892499829814,
                            0.8,  25.967861311696748, 0.81, 26.963067960816524,
                            0.82, 27.413697219443772, 0.83, 29.402252152133507,
                            0.84, 29.38910202704645,  0.85, 31.703184437006364,
                            0.86, 31.383499428298023, 0.87, 33.24303523747431,
                            0.88, 35.13949486699716,  0.89, 35.95026507322022,
                            0.9,  37.8700855562006,   0.91, 38.96574656474007,
                            0.92, 41.13805704410948,  0.93, 41.5336964636455,
                            0.94, 44.48150997682299,  0.95, 45.49884409932172,
                            0.96, 44.20442520005128,  0.97, 50.11333249129726,
                            0.98, 50.33158501899404,  0.99, 53.590103886971576};

// 定义一个残差函数
struct MyResidual {
  MyResidual(double x_, double y_) : x(x_), y(y_) {}
  template <typename T>
  bool operator()(const T* const parameters, T* residuals) const {
    // 将残差的值赋给residuals数组
    residuals[0] =
        y -
        (ceres::exp(parameters[0] * x * x + parameters[1] * x + parameters[2]));
    return true;
  }
  double x;
  double y;
};

int main(int argc, char** argv) {
  double abc[3] = {2, -1, 5};

  fpm::fixed_16_16 aabbcc[3];
  aabbcc[0]=fpm::fixed_16_16{2};
  aabbcc[1]=fpm::fixed_16_16{-1};
  aabbcc[2]=fpm::fixed_16_16{5};

  // 构建寻优问题
  ceres::Problem problem;
  for (int i = 0; i < data.size(); i += 2) {
    // std::cout << data[i] << " " << data[i + 1] << std::endl;
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<MyResidual, 1, 3>(
            new MyResidual(data.at(i), data.at(i + 1)));
    problem.AddResidualBlock(cost_function, NULL, abc);
  }


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;  // 确定求解方法（Ax=b）
  options.minimizer_progress_to_stdout = true;   // 输出到cout
  ceres::Solver::Summary summary;                // 优化信息
  Solve(options, &problem, &summary);            // 求解

  std::cout << summary.BriefReport() << std::endl;  // 输出优化的简要信息
  std::cout << "abc[0]=" << abc[0] << std::endl;
  std::cout << "abc[1]=" << abc[1] << std::endl;
  std::cout << "abc[2]=" << abc[2] << std::endl;
  return 0;
}
