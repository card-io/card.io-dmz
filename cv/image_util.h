//
//  image_util.h
//  See the file "LICENSE.md" for the full license governing this code.


#ifndef LLCV_UTIL_H
#define LLCV_UTIL_H

#include "opencv2/core/core_c.h"
#include "dmz_macros.h"

DMZ_INTERNAL void* llcv_get_data_origin(IplImage *image);
DMZ_INTERNAL uint8_t llcv_get_pixel_step(IplImage *image);

#endif
