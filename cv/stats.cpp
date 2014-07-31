//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "dmz_macros.h"
#include "stats.h"
#include "processor_support.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"  // used in llcv_equalize_hist

#if DMZ_HAS_NEON_COMPILETIME
  #include <arm_neon.h>
#endif

DMZ_INTERNAL float llcv_stddev_of_abs_neon(IplImage *image) {
#if DMZ_HAS_NEON_COMPILETIME
#define kVectorSize 8
  CvSize image_size = cvGetSize(image);
  
  const uint8_t *image_origin = (uint8_t *)(image->imageData);
  if(dmz_likely(NULL != image->roi)) {
    image_origin += image->roi->yOffset * image->widthStep + image->roi->xOffset * sizeof(int16_t);
  }
  uint16_t image_width_step = (uint16_t)image->widthStep;

  uint16_t scalar_cols = (uint16_t)(image_size.width % kVectorSize);
  uint16_t vector_chunks = (uint16_t)(image_size.width / kVectorSize);
  
  uint16_t vector_cols = vector_chunks * kVectorSize;
  int32x4_t vector_sum = {0, 0, 0, 0};
  int64x2_t vector_sum_squared = {0, 0};
  int32_t scalar_sum = 0;
  int64_t scalar_sum_squared = 0;
  
  for(uint16_t row_index = 0; row_index < image_size.height; row_index++) {
    const int16_t *image_row_origin = (int16_t *)(image_origin + row_index * image_width_step);
    
    for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
      uint16_t col_index = vector_index * kVectorSize;
      const int16_t *image_data = image_row_origin + col_index;
      
      int16x8_t image_vector = vld1q_s16(image_data);
      
      int16x4_t image_d0 = vget_low_s16(image_vector);
      int16x4_t image_d1 = vget_high_s16(image_vector);
      
      int32x4_t d0_squared = vmull_s16(image_d0, image_d0);
      int32x4_t d1_squared = vmull_s16(image_d1, image_d1);

      image_vector = vqabsq_s16(image_vector); // saturating absolute value
      
      vector_sum_squared = vpadalq_s32(vector_sum_squared, d0_squared); // pairwise add and accumulate
      vector_sum_squared = vpadalq_s32(vector_sum_squared, d1_squared);
      vector_sum = vpadalq_s16(vector_sum, image_vector);
    }
    
    for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
      uint16_t col_index = scalar_index + vector_cols;
      int16_t pixel_val = image_row_origin[col_index];
      scalar_sum += abs(pixel_val);
      scalar_sum_squared += pixel_val * pixel_val;
    }
  }
  
  scalar_sum += vgetq_lane_s32(vector_sum, 0);
  scalar_sum += vgetq_lane_s32(vector_sum, 1);
  scalar_sum += vgetq_lane_s32(vector_sum, 2);
  scalar_sum += vgetq_lane_s32(vector_sum, 3);

  scalar_sum_squared += vgetq_lane_s64(vector_sum_squared, 0);
  scalar_sum_squared += vgetq_lane_s64(vector_sum_squared, 1);

  float32_t n_elements = image_size.width * image_size.height;

  float32_t mean = scalar_sum / n_elements;
  float32_t stddev = sqrtf(scalar_sum_squared / n_elements - mean * mean);
  
  return stddev;
#undef kVectorSize
#else
  return 0.0f;
#endif
}

DMZ_INTERNAL float llcv_stddev_of_abs_c(IplImage *image) {
  cvAbs(image, image);
  CvScalar stddev;
  cvAvgSdv(image, NULL, &stddev, NULL);
  return (float)stddev.val[0];
}

#define TEST_STDDEV_NEON 0

DMZ_INTERNAL float llcv_stddev_of_abs(IplImage *image) {
  assert(image->depth == IPL_DEPTH_16S);
  assert(image->nChannels == 1);

  if(dmz_has_neon_runtime()) {
    // NB for testing: must do neon calc before c calc, because c calc changes the image!
    float neon_ret = llcv_stddev_of_abs_neon(image);
#if TEST_STDDEV_NEON
    double c_ret = llcv_stddev_of_abs_c(image);
    fprintf(stderr, "llcv_stddev C: %f, NEON: %f, DELTA: %f (%f %%)\n", c_ret, neon_ret, c_ret - neon_ret, 100.0f * (c_ret - neon_ret) / c_ret);
#endif
    return neon_ret;
  } else {
    return llcv_stddev_of_abs_c(image);
  }
}


// This implementation copied directly from OpenCV's cvEqualizeHist, as
// part of an effort to remove dependencies on libopencv_imgproc.a.
DMZ_INTERNAL void llcv_equalize_hist(const IplImage *srcimg, IplImage *dstimg) {
  CvMat sstub, *src = cvGetMat(srcimg, &sstub);
  CvMat dstub, *dst = cvGetMat(dstimg, &dstub);
  
  CV_Assert( CV_ARE_SIZES_EQ(src, dst) && CV_ARE_TYPES_EQ(src, dst) &&
            CV_MAT_TYPE(src->type) == CV_8UC1 );
  CvSize size = cvGetMatSize(src);
  if( CV_IS_MAT_CONT(src->type & dst->type) )
  {
    size.width *= size.height;
    size.height = 1;
  }
  int x, y;
  const int hist_sz = 256;
  int hist[hist_sz];
  memset(hist, 0, sizeof(hist));
  
  for( y = 0; y < size.height; y++ )
  {
    const uchar* sptr = src->data.ptr + src->step*y;
    for( x = 0; x < size.width; x++ )
      hist[sptr[x]]++;
  }
  
  float scale = 255.f/(size.width*size.height);
  int sum = 0;
  uchar lut[hist_sz+1];
  
  for( int i = 0; i < hist_sz; i++ )
  {
    sum += hist[i];
    int val = cvRound(sum*scale);
    lut[i] = CV_CAST_8U(val);
  }
  
  lut[0] = 0;
  for( y = 0; y < size.height; y++ )
  {
    const uchar* sptr = src->data.ptr + src->step*y;
    uchar* dptr = dst->data.ptr + dst->step*y;
    for( x = 0; x < size.width; x++ )
      dptr[x] = lut[sptr[x]];
  }
}

#endif
