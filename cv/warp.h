//
//  warp.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef WARP_H_
#define WARP_H_

#include "dmz_macros.h"
#include "dmz.h"

// Returns true if textures are input automatically stretched to fit
// Useful for determining if points should be normalized or altered 
// prior to llcv_unwarp.
DMZ_INTERNAL bool llcv_warp_auto_upsamples();


// unwarps input image, interpolating image such that src_points map to dst_rect coordinates.
// Image is written to output IplImage.
void llcv_unwarp(dmz_context *dmz, IplImage *input, const dmz_point src_points[4], const dmz_rect dst_rect, IplImage *output);

// Solves and writes perpsective matrix to the matrixData buffer. 
// If matrixDataSize >= 16, uses a 4x4 matrix. Otherwise a 3x3. 
// Specifying rowMajor true writes to the buffer in row major format.
void llcv_calc_persp_transform(float *matrixData, int matrixDataSize, bool rowMajor, const dmz_point sourcePoints[], const dmz_point destPoints[]);


#endif /* WARP_H_ */
