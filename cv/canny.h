#ifndef CANNY_H
#define CANNY_H

#include "opencv2/core/core_c.h" // for IplImage
#include "dmz_macros.h"

// Canny on an image, with aperature 7.
DMZ_INTERNAL void llcv_canny7(IplImage *src, IplImage *dst, double low_thresh, double high_thresh);
DMZ_INTERNAL void llcv_adaptive_canny7_precomputed_sobel(IplImage *src, IplImage *dst, IplImage *dx, IplImage *dy);

#endif
