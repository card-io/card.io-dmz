//
//  morph.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef MORPH_H
#define MORPH_H

#include "opencv2/imgproc/imgproc_c.h"
#include "dmz_macros.h"

DMZ_INTERNAL void llcv_morph_grad3_1d_u8(IplImage *src, IplImage *dst);
DMZ_INTERNAL void llcv_morph_grad3_2d_cross_u8(IplImage *src, IplImage *dst);

#endif
