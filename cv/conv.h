//
//  conv.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_CV_CONV_H
#define DMZ_CV_CONV_H

#include "dmz_macros.h"

// Runs a 3x3 convolution kernel to generate a single row of output
// Accepts three (presumably consecutive) input row pointers.
// The kernel should be in row major order, and be PADDED WITH TRAILING ZEROS at the end of each row to be 3x4 in shape.
// The length is the number of output entries to calculate and write, and MUST be greater than 0 AND MUST be divisible by 4.
// Note that since we will be reading 4 values at a time from the inrows, you should pass a length ONE LESS than what you might think,
// and handle the last element in a safe, scalar way.
// All bounds checking and kernel alignment must be done by the caller.
// Only implemented when NEON is available. Will crash if called when NEON not available.
DMZ_INTERNAL void llcv_conv_3x3_f32_row(const float *inrow0, const float *inrow1, const float *inrow2, float *kernel3x4, float *outrow, uint16_t length);

#endif
