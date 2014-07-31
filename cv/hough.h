#ifndef HOUGH_H
#define HOUGH_H

#include "opencv2/core/core_c.h" // for IplImage
#include "dmz_macros.h"

typedef struct CvLinePolar {
    float rho;
    float angle;
    bool is_null;
} CvLinePolar;

DMZ_INTERNAL CvLinePolar llcv_hough(const CvArr *src_image, IplImage *dx, IplImage *dy, float rho, float theta, int threshold, float theta_min, float theta_max, bool vertical, float gradient_angle_threshold);

#endif
