//  See the file "LICENSE.md" for the full license governing this code.

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "dmz_macros.h"
#include "dmz.h"
#include "opencv2/core/types_c.h"

static inline CvRect cvInsetRect(CvRect originalRect, int horizontalInset, int verticalInset) {
  return cvRect(originalRect.x + horizontalInset,
                originalRect.y + verticalInset,
                originalRect.width - 2 * horizontalInset,
                originalRect.height - 2 * verticalInset);
}


DMZ_INTERNAL bool is_parametric_line_none(ParametricLine line_to_test);

// Find point (x, y) where two parameterized lines intersect. Returns true iff the lines intersect.
DMZ_INTERNAL bool parametricIntersect(ParametricLine line1, ParametricLine line2, float *x, float *y);
DMZ_INTERNAL ParametricLine lineByShiftingOrigin(ParametricLine oldLine, int xOffset, int yOffset);

DMZ_INTERNAL inline ParametricLine ParametricLineNone() {
  ParametricLine l;
  l.rho = FLT_MAX;
  l.theta = FLT_MAX;
  return l;
}

#endif // GEOMETRY_H
