//
//  cv_ios.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef icc_cv_ios_h
#define icc_cv_ios_h

#include "dmz.h"
#include "dmz_macros.h"

// Skews input image from 4 dmz_points to the given dmz_rect.
// Results are rendered to output IplImage, which should already be created.
// from_points should have the following ordering: top-left, top-right, bottom-left, bottom-right
void ios_gpu_unwarp(dmz_context *dmz, IplImage *input, const dmz_point from_points[4], IplImage *output);

#if DMZ_DEBUG
void ios_save_file(char *filename, IplImage *image);
#endif

#endif
