//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "geometry.h"
#include "eigen.h"
#include "dmz_debug.h"

DMZ_INTERNAL bool is_parametric_line_none(ParametricLine line_to_test) {
  return ((line_to_test).theta == FLT_MAX);
}

DMZ_INTERNAL bool parametricIntersect(ParametricLine line1, ParametricLine line2, float *x, float *y) {
  if(is_parametric_line_none(line1) || is_parametric_line_none(line2)) {
    return false;
  }

  Eigen::Matrix2f t;
  Eigen::Vector2f r;
  t << cosf(line1.theta), sinf(line1.theta), cosf(line2.theta), sinf(line2.theta);
  r << line1.rho, line2.rho;

  if(t.determinant() < 1e-10) {
    return false;
  }
  
  Eigen::Vector2f intersection = t.inverse() * r;
  *x = intersection(0);
  *y = intersection(1);
  return true;
}

DMZ_INTERNAL ParametricLine lineByShiftingOrigin(ParametricLine oldLine, int xOffset, int yOffset) {
  ParametricLine newLine;
  newLine.theta = oldLine.theta;
  double offsetAngle = xOffset == 0 ? CV_PI / 2.0f : atan((float)yOffset / (float)xOffset);
  double deltaAngle = oldLine.theta - offsetAngle + CV_PI / 2.0f; // because we're working with the line *normal* to theta
  double offsetMagnitude = sqrt(xOffset * xOffset + yOffset * yOffset);
  double delta_rho = offsetMagnitude * cos(CV_PI / 2 - deltaAngle);
  newLine.rho = (float)(oldLine.rho + delta_rho);
  return newLine;
}

#endif // COMPILE_DMZ
