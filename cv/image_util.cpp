//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "image_util.h"

DMZ_INTERNAL uint8_t llcv_get_pixel_step(IplImage *image) {
  uint8_t pixel_step = 0;
  switch(image->depth) {
    case IPL_DEPTH_8S:
    case IPL_DEPTH_8U:
      pixel_step = sizeof(uint8_t);
      break;
    case IPL_DEPTH_16U:
    case IPL_DEPTH_16S:
      pixel_step = sizeof(uint16_t);
      break;
    case IPL_DEPTH_32F:
    case IPL_DEPTH_32S:
      pixel_step = sizeof(uint32_t);
      break;
    case IPL_DEPTH_64F:
      pixel_step = sizeof(uint64_t);
      break;
  }
  return pixel_step;
}

DMZ_INTERNAL void* llcv_get_data_origin(IplImage *image) {
  uint8_t pixel_step = llcv_get_pixel_step(image);
  uint8_t *data_origin = (uint8_t *)image->imageData;
  if(NULL != image->roi) {
    data_origin += image->roi->yOffset * image->widthStep + image->roi->xOffset * pixel_step;
  }
  return data_origin;
}

#endif