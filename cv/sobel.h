//  See the file "LICENSE.md" for the full license governing this code.

#ifndef SOBEL_H
#define SOBEL_H

#include "opencv2/core/core_c.h" // for IplImage
#include "opencv2/imgproc/imgproc_c.h"
#include "dmz_macros.h"

// Convolve with a sobel kernel of size 7.
// src must be of type 8UC1; dst must be of type 16SC1.
// (dx, dy) must be either (0, 1) or (1, 0).
// scratch space may be NULL (in which it will be allocated internally), or it may be
// an image of size (src_height, src_width) -- yes, transposed! -- of type 16SC1.
DMZ_INTERNAL void llcv_sobel7(IplImage *src, IplImage *dst, IplImage *scratch, bool dx, bool dy);

// Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
#if DMZ_DEBUG
void llcv_scharr3_dx_abs(IplImage *src, IplImage *dst);
void llcv_scharr3_dy_abs(IplImage *src, IplImage *dst);
#else
DMZ_INTERNAL_UNLESS_CYTHON void llcv_scharr3_dx_abs(IplImage *src, IplImage *dst);
DMZ_INTERNAL_UNLESS_CYTHON void llcv_scharr3_dy_abs(IplImage *src, IplImage *dst);
#endif

void llcv_scharr3_dx(IplImage *src, IplImage *dst);

#endif
