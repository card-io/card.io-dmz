//  See the file "LICENSE.md" for the full license governing this code.

#ifndef CONVERT_H
#define CONVERT_H

#include "opencv2/core/core_c.h"
#include "dmz_macros.h"

DMZ_INTERNAL void llcv_split_u8(IplImage *interleaved, IplImage *channel1, IplImage *channel2);
DMZ_INTERNAL void llcv_lineardown2_1d_u8(IplImage *src, IplImage *dst);
DMZ_INTERNAL void llcv_norm_convert_1d_u8_to_f32(IplImage *src, IplImage *dst);
DMZ_INTERNAL void llcv_YCbCr2RGB_u8(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst);

#endif
