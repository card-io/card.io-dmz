//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "stdint.h"
#include "mz.h"

#if CYTHON_DMZ

// this is where we'll pre-allocate and setup OpenGL textures, compile our program,
// and maintain a reference to all this. Return a reference to some allocated object or struct
// that tracks this information, for use by later dmz functions.
void *mz_create(void) {
  return NULL;
}

// free up any persistent references to OpenGL textures, programs, shaders, etc, as well
// as any wrapping objects.
void mz_destroy(void *mz) {
}

// Perform any necessary operations prior to app backgrounding (e.g., calling glFinish() on any OpenGL contexts)
void mz_prepare_for_backgrounding(void *mz) {
}

IplImage *py_mz_create_from_cv_image_data(char *image_data, int image_size,
                                          int width, int height,
                                          int64_t depth, int n_channels,
                                          int roi_x_offset, int roi_y_offset, int roi_width, int roi_height) {
  int bits_per_pixel = 8;
  switch(depth) {
    case IPL_DEPTH_1U:
      bits_per_pixel = 1;
      break;
    case IPL_DEPTH_8S:
    case IPL_DEPTH_8U:
      bits_per_pixel = 8;
      break;
    case IPL_DEPTH_16S:
    case IPL_DEPTH_16U:
      bits_per_pixel = 16;
      break;
    case IPL_DEPTH_32S:
    case IPL_DEPTH_32F:
      bits_per_pixel = 32;
      break;
    case IPL_DEPTH_64F:
      bits_per_pixel = 64;
      break;
    default:
      bits_per_pixel = 8;
      break;
  }
  
  int step = (bits_per_pixel * n_channels * width) / 8;  // bits_per_pixel is in bits, step is in bytes
  //  step += (4 - (step % 4)) % 4; // pad to multiple of 4 bytes
  //
  // Some people claim that IplImage row-padding can be to either 4 or 8 bytes. But according to
  // http://opencv.willowgarage.com/documentation/c/core_basic_structures.html#CvMat it's always 4 bytes.
  //
  // For the Python version of IplImage, http://opencv.willowgarage.com/documentation/python/core_basic_structures.html#CvMat
  // does not say anything at all about padding.
  //
  // For now, let's simply assert if there appears to be any padding:
  if (image_size > 0 && step * height != image_size) {
    fprintf(stderr, "\nstep: %d, height: %d, image_size: %d => padding: %d\n", step, height, image_size, image_size/height % step);
    assert(FALSE);
  }

  IplImage *image = cvCreateImageHeader(cvSize(width, height), depth, n_channels);
  cvSetData(image, image_data, step);
  cvSetImageROI(image, cvRect(roi_x_offset, roi_y_offset, roi_width, roi_height));
  return image;
}

void py_mz_release_ipl_image(IplImage *image) {
  cvReleaseImageHeader(&image);
}

void py_mz_get_cv_image_data(IplImage *source,
                             char **image_data, int *image_size,
                             int *width, int *height,
                             int64_t *depth, int *n_channels,
                             int *roi_x_offset, int *roi_y_offset, int *roi_width, int *roi_height
                            ) {
    *image_data = source->imageData;
    *image_size = source->imageSize;
    *width = source->width;
    *height = source->height;
    *depth = source->depth;
    *n_channels = source->nChannels;
    CvRect roi = cvGetImageROI(source);
    *roi_x_offset = roi.x;
    *roi_y_offset = roi.y;
    *roi_width = roi.width;
    *roi_height = roi.height;
}

void py_mz_cvSetImageROI(IplImage* image, int left, int top, int width, int height) {
  cvSetImageROI(image, cvRect(left, top, width, height));
}

void py_mz_cvResetImageROI(IplImage* image) {
  cvResetImageROI(image);
}

//void py_mz_expiry_fill_borders(IplImage *character_image_float, int char_top, int char_left) {
//  expiry_fill_borders(character_image_float, char_top, char_left);
//}


#endif


#endif // COMPILE_DMZ
