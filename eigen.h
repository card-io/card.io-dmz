//
//  eigen.h
//  See the file "LICENSE.md" for the full license governing this code.
//

// Convenience one-stop-shop for Eigen imports

#ifndef DMZ_EIGEN_H
#define DMZ_EIGEN_H

#define EIGEN_MPL2_ONLY 1

#define EIGEN_NO_MALLOC 1 // disable all heap allocation of matrices

#if !DEBUG
  // turn off range checking, asserts, and anything else that could slow us down
  // NB: turning off asserts is important! When EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO is 0,
  // as it is with llvm-gcc4.2 (which is our current iOS compiler), then a very slow
  // assertion macro replacement is used. As of time of writing, it cuts the 4S from
  // ~22fps to ~17fps.
  #define NDEBUG 1
  #define EIGEN_NO_DEBUG 1
#endif

#include "Eigen/Core"
#include "Eigen/Dense"

#endif
