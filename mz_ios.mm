//
//  dmz_ios.m
//  See the file "LICENSE.md" for the full license governing this code.
//

#if USE_CAMERA

#include "mz_ios.h"
#import "CardIOIplImage.h"
#import "CardIOGPUTransformFilter.h"
#include "warp.h"

const dmz_point kWarpGLDestPoints[4] = {
  dmz_create_point(-1, -1), // bottom-left GL -> top-left in image
  dmz_create_point( 1, -1), // bottom-right GL -> top-right in image
  dmz_create_point(-1,  1), // top-left GL -> bottom-left in image
  dmz_create_point( 1,  1), // top-right GL -> bottom-right in image
};

// this is where we'll pre-allocate and setup OpenGL textures, compile our program, 
// and maintain a reference to all this. Return a reference to some allocated object or struct 
// that tracks this information, for use by later dmz functions.
void *mz_create(void) {
  void *filter = (__bridge_retained void *)[[CardIOGPUTransformFilter alloc] initWithSize:CGSizeMake(kLandscapeSampleWidth, kLandscapeSampleHeight)];
  return filter;
}

// free up any persistent references to OpenGL textures, programs, shaders, etc, as well 
// as any wrapping objects.
void mz_destroy(void *mz) {
  CardIOGPUTransformFilter *filter = (__bridge_transfer CardIOGPUTransformFilter *)mz; // needed to "release" filter
#pragma unused(filter)
}

// tell the filter to call glFinish() on its context
void mz_prepare_for_backgrounding(void *mz) {
  [(__bridge CardIOGPUTransformFilter *)mz finish];
}

void ios_gpu_unwarp(dmz_context *dmz, IplImage *input, const dmz_point from_points[4], IplImage *output) {
  // Create filter if necessary
  CardIOGPUTransformFilter *filter = (__bridge CardIOGPUTransformFilter *) dmz->mz;
  if (filter) {
    // Filter dimensions should be an integral multiple (usually 1 or 2) of the input dimensions.
    // If that's not the current case, then let's re-make our filter.
    int widthRatio = (int)filter.size.width / input->width;
    int heightRatio = (int)filter.size.height / input->height;
    if (widthRatio != heightRatio
        || input->width * widthRatio != filter.size.width
        || input->height * heightRatio != filter.size.height) {
      filter = nil;
    }
  }
  if (filter == nil) {
    filter = [[CardIOGPUTransformFilter alloc] initWithSize:CGSizeMake(input->width, input->height)];
    dmz->mz = (__bridge_retained void *)filter;
  }
  
  // Calculate and set perspective matrix, then process the image.
  float perspMat[16];
  llcv_calc_persp_transform(perspMat, 16, false, from_points, kWarpGLDestPoints);  
  [filter setPerspectiveMat:perspMat];

  [filter processIplImage:input dstIplImg:output];
}

#endif // USE_CAMERA
