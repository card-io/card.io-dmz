//  See the file "LICENSE.md" for the full license governing this code.

#ifndef STATS_H
#define STATS_H

#include "opencv2/core/core_c.h" // for IplImage
#include "dmz_macros.h"

// NOTE: For performance reasons, this function may alter the contents of image!
DMZ_INTERNAL float llcv_stddev_of_abs(IplImage *image);
DMZ_INTERNAL void llcv_equalize_hist(const IplImage *srcimg, IplImage *dstimg);

#endif
