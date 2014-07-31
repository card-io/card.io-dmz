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

DMZ_INTERNAL void llcv_sobel3_dx_dy(IplImage *src, IplImage *dst);

void llcv_scharr3_dx(IplImage *src, IplImage *dst);

#endif
