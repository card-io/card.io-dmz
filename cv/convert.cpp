//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "dmz_macros.h"
#include "convert.h"
#include "processor_support.h"
#include "neon.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "image_util.h"

#if DMZ_HAS_NEON_COMPILETIME
#include <arm_neon.h>
#endif

typedef uint16_t uint8x2_t;

DMZ_INTERNAL void llcv_split_u8_neon(IplImage *interleaved, IplImage *channel1, IplImage *channel2) {
#if DMZ_HAS_NEON_COMPILETIME
#define kVectorSize 16
  CvSize image_size = cvGetSize(interleaved);
  
  const uint8_t *image_origin = (uint8_t *)(interleaved->imageData);
  uint16_t image_width_step = (uint16_t)interleaved->widthStep;
  if(dmz_unlikely(NULL != interleaved->roi)) {
    image_origin += interleaved->roi->yOffset * image_width_step + interleaved->roi->xOffset * sizeof(uint8x2_t);
  }

  uint8_t *channel1_origin = (uint8_t *)(channel1->imageData);
  uint16_t channel1_width_step = (uint16_t)channel1->widthStep;
  if(dmz_unlikely(NULL != channel1->roi)) {
    channel1_origin += channel1->roi->yOffset * channel1_width_step + channel1->roi->xOffset * sizeof(uint8_t);
  }

  uint8_t *channel2_origin = (uint8_t *)(channel2->imageData);
  uint16_t channel2_width_step = (uint16_t)channel2->widthStep;
  if(dmz_unlikely(NULL != channel2->roi)) {
    channel2_origin += channel2->roi->yOffset * channel2_width_step + channel2->roi->xOffset * sizeof(uint8_t);
  }

  uint16_t scalar_cols = (uint16_t)(image_size.width % kVectorSize);
  uint16_t vector_chunks = (uint16_t)(image_size.width / kVectorSize);
  uint16_t vector_cols = vector_chunks * kVectorSize;
  
  for(uint16_t row_index = 0; row_index < image_size.height; row_index++) {
    const uint8x2_t *image_row_origin = (uint8x2_t *)(image_origin + row_index * image_width_step);
    uint8_t *channel1_row_origin = channel1_origin + row_index * channel1_width_step;
    uint8_t *channel2_row_origin = channel2_origin + row_index * channel2_width_step;
    
#pragma unroll(4)
    for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
      uint16_t col_index = vector_index * kVectorSize;
      const uint8x2_t *image_data = image_row_origin + col_index;
      uint8_t *channel1_data = channel1_row_origin + col_index;
      uint8_t *channel2_data = channel2_row_origin + col_index;

      uint8x16x2_t deinterleaved = vld2q_u8((const uint8_t *)image_data); // load 32 bytes into two registers, deinterleaving along the way
      vst1q_u8(channel1_data, deinterleaved.val[0]); // write the first 16 bytes into channel 1...
      vst1q_u8(channel2_data, deinterleaved.val[1]); // and the second 16 bytes into channel 2
    }
    
    for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
      uint16_t col_index = scalar_index + vector_cols;
      uint8x2_t pixel_val = image_row_origin[col_index];
      channel1_row_origin[col_index] = (uint8_t)(pixel_val >> 8);
      channel2_row_origin[col_index] = (uint8_t)(pixel_val & 0xFF);
    }
  }
#undef kVectorSize
#endif
}

DMZ_INTERNAL void llcv_split_u8_c(IplImage *interleaved, IplImage *channel1, IplImage *channel2) {
  cvSplit(interleaved, channel1, channel2, NULL, NULL);
}

#define TEST_DEINTERLACE_NEON 0

DMZ_INTERNAL void llcv_split_u8(IplImage *interleaved, IplImage *channel1, IplImage *channel2) {
  assert(interleaved->nChannels == 2);
  assert(channel1->nChannels == 1);
  assert(channel2->nChannels == 1);

  assert(interleaved->depth == IPL_DEPTH_8U);
  assert(channel1->depth == IPL_DEPTH_8U);
  assert(channel2->depth == IPL_DEPTH_8U);

#if DMZ_DEBUG
  CvSize interleaved_size = cvGetSize(interleaved);
  CvSize channel1_size = cvGetSize(channel1);
  CvSize channel2_size = cvGetSize(channel2);
  assert(interleaved_size.width == channel1_size.width && interleaved_size.height == channel1_size.height);
  assert(interleaved_size.width == channel2_size.width && interleaved_size.height == channel2_size.height);
#endif

  if(dmz_has_neon_runtime()) {
    llcv_split_u8_neon(interleaved, channel1, channel2);
#if TEST_DEINTERLACE_NEON
    CvSize image_size = cvGetSize(interleaved);

    IplImage *channel1_c = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
    IplImage *channel2_c = cvCreateImage(image_size, IPL_DEPTH_8U, 1);

    llcv_split_u8_c(interleaved, channel1_c, channel2_c);

    IplImage *channel1_delta = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
    IplImage *channel2_delta = cvCreateImage(image_size, IPL_DEPTH_8U, 1);

    cvAbsDiff(channel1, channel1_c, channel1_delta);
    cvAbsDiff(channel2, channel2_c, channel2_delta);

    int n_channel1_errors = cvCountNonZero(channel1_delta);
    int n_channel2_errors = cvCountNonZero(channel2_delta);

    fprintf(stderr, "llcv_split_u8 errors on c1: %i; c2: %i\n", n_channel1_errors, n_channel2_errors);

    cvReleaseImage(&channel1_delta);
    cvReleaseImage(&channel2_delta);

    cvReleaseImage(&channel1_c);
    cvReleaseImage(&channel2_c);
#endif
  } else {
    llcv_split_u8_c(interleaved, channel1, channel2);
  }
}

#if DMZ_HAS_NEON_COMPILETIME
// linear2 support functions

static inline void vec_lineardown2_1d_u8_q(const uint8_t *src, uint8_t *dst) {
  asm volatile
  (
   // load src pixels
   "mov r0, %[src]" "\n\t"
   "vld2.8 {q0-q1}, [r0]" "\n\t"

   // halving rounding add
   "vrhadd.u8 q2, q0, q1" "\n\t"

   // write to dst
   "mov r0, %[dst]" "\n\t"
   "vst1.8 {q2}, [r0]" "\n\t"
   
   : // output
   
   : // input
   [src]"r" (src),
   [dst]"r" (dst)
   
   : // clobbered
   "r0", // used for src, dst pointers
   "q0", "q1", "q2",
   "memory"
   );
}

#endif

DMZ_INTERNAL void llcv_lineardown2_1d_u8_neon(IplImage *src, IplImage *dst) {
#if DMZ_HAS_NEON_COMPILETIME

  uint8_t *src_data = (uint8_t *)src->imageData;
  if(dmz_unlikely(NULL != src->roi)) {
    src_data += src->roi->yOffset * src->widthStep + src->roi->xOffset * sizeof(uint8_t);
  }
  
  uint8_t *dst_data = (uint8_t *)dst->imageData;
  if(dmz_unlikely(NULL != dst->roi)) {
    dst_data += dst->roi->yOffset * dst->widthStep + dst->roi->xOffset * sizeof(uint8_t);
  }
  
  CvSize src_size = cvGetSize(src);
  
  uint16_t src_index = 0;

  bool done = false;
  while(!done) {
    // calc using 2 q registers at a time
    vec_lineardown2_1d_u8_q(src_data + src_index, dst_data + (src_index >> 1));
    src_index += (kQRegisterElements8 << 1);
    
    if(src_index == src_size.width) {
      done = true;
    } else if(src_index > src_size.width - (kQRegisterElements8 << 1)) {
      // backtrack to handle leftovers
      src_index = (uint16_t)(src_size.width - (kQRegisterElements8 << 1));
    }
  }

#endif
}

DMZ_INTERNAL void llcv_lineardown2_1d_u8_c(IplImage *src, IplImage *dst) {
  cvResize(src, dst, CV_INTER_LINEAR);
}

#define TEST_LINEAR_DOWN2 0

DMZ_INTERNAL void llcv_lineardown2_1d_u8(IplImage *src, IplImage *dst) {
  assert(src->depth == IPL_DEPTH_8U);
  assert(dst->depth == IPL_DEPTH_8U);
  assert(src != dst);
  assert(src->nChannels == 1);
  assert(dst->nChannels == 1);
#if DMZ_DEBUG
  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
  assert(src_size.height == dst_size.height);
  assert(src_size.height == 1);  // 1d!
  assert(src_size.width >= (kQRegisterElements8 << 1));
  assert(src_size.width % 2 == 0);
  assert(src_size.width / 2 == dst_size.width);
#endif
  
  if(dmz_has_neon_runtime()) {
    llcv_lineardown2_1d_u8_neon(src, dst);
#if TEST_LINEAR_DOWN2
    IplImage *dst_c = cvCreateImage(dst_size, IPL_DEPTH_8U, 1);
    
    llcv_lineardown2_1d_u8_c(src, dst_c);
    
    IplImage *delta = cvCreateImage(dst_size, IPL_DEPTH_8U, 1);
    
    cvAbsDiff(dst, dst_c, delta);
    
    int n_errors = cvCountNonZero(delta);
    if(n_errors > 0) {
      fprintf(stderr, "llcv_lineardown2_1d_u8 errors: %i\n", n_errors);
      // for (uint16_t x = 0; x < dst_size.width; x++) {
      //   fprintf(stderr, "%d, ", int(cvGet1D(dst_c, x).val[0]));
      // }
      // fprintf(stderr, "\n");
    }
    
    cvReleaseImage(&dst_c);
    cvReleaseImage(&delta);
#endif
  } else {
    llcv_lineardown2_1d_u8_c(src, dst);
  }
}


#define TEST_NORM_CONVERT 0

DMZ_INTERNAL void llcv_minmax_1d_u8_neon(uint8_t *src_data, uint16_t src_length, uint8_t *min_val, uint8_t *max_val) {
#if DMZ_HAS_NEON_COMPILETIME
  uint8x16_t running_min = vdupq_n_u8(UINT8_MAX);
  uint8x16_t running_max = vdupq_n_u8(0);

  uint16_t index = 0;
  bool done = false;
  while(!done) {
    uint8x16_t chunk = vld1q_u8(src_data + index);
    running_min = vminq_u8(running_min, chunk);
    running_max = vmaxq_u8(running_max, chunk);
    index += kQRegisterElements8;
  
    if(index == src_length) {
      done = true;
    } else if(index > src_length - kQRegisterElements8) {
      // backtrack to handle leftovers
      index = src_length - kQRegisterElements8;
    }
  }

  // collapse q registers (16) to d registers (8)
  uint8x8_t max_8 = vmax_u8(vget_low_u8(running_max), vget_high_u8(running_max));
  uint8x8_t min_8 = vmin_u8(vget_low_u8(running_min), vget_high_u8(running_min));

  // see diagrams at http://blogs.arm.com/software-enablement/684-coding-for-neon-part-5-rearranging-vectors/
  
  // collapse adjacent pairs: 0-1, 2-3, 4-5, 6-7
  max_8 = vmax_u8(max_8, vrev16_u8(max_8));
  min_8 = vmin_u8(min_8, vrev16_u8(min_8));

  // collapse adjacent pairs-of-pairs, so now: 0-3, 4-7
  max_8 = vmax_u8(max_8, vrev32_u8(max_8));
  min_8 = vmin_u8(min_8, vrev32_u8(min_8));

  // now a simple scalar max 0, 4 is enough
  uint8_t max_first = vget_lane_u8(max_8, 0);
  uint8_t max_second = vget_lane_u8(max_8, 4);
  *max_val = MAX(max_first, max_second);

  uint8_t min_first = vget_lane_u8(min_8, 0);
  uint8_t min_second = vget_lane_u8(min_8, 4);
  *min_val = MIN(min_first, min_second);
#endif
}


DMZ_INTERNAL void llcv_norm_convert_1d_u8_to_f32_neon(IplImage *src, IplImage *dst) {
#if DMZ_HAS_NEON_COMPILETIME
  // Find min/max
  
  uint8_t *src_data = (uint8_t *)src->imageData;
  if(dmz_unlikely(NULL != src->roi)) {
    src_data += src->roi->yOffset * src->widthStep + src->roi->xOffset * sizeof(uint8_t);
  }
  
  uint8_t *dst_data_origin = (uint8_t *)dst->imageData;
  if(dmz_unlikely(NULL != dst->roi)) {
    dst_data_origin += dst->roi->yOffset * dst->widthStep + dst->roi->xOffset * sizeof(float32_t);
  }
  float32_t *dst_data = (float32_t *)dst_data_origin;

  CvSize src_size = cvGetSize(src);
  
  uint8_t src_min;
  uint8_t src_max;
  llcv_minmax_1d_u8_neon(src_data, (uint16_t)src_size.width, &src_min, &src_max);

#if TEST_NORM_CONVERT
  double cv_min, cv_max;
  cvMinMaxLoc(src, &cv_min, &cv_max, NULL, NULL);
  if(int(cv_min) != src_min || int(cv_max) != src_max) {
    fprintf(stderr, "cv_min %f src_min %d; cv_max %f, src_max %d\n", cv_min, src_min, cv_max, src_max);
  }
#endif

  uint8_t delta = src_max - src_min;
  float32_t multiplier = delta == 0 ? 0.5f : 1.0f / delta; // if delta == 0, they're *all* identical, so it doesn't really matter where we map them, it's going to be bad...
  
  uint8x16_t vec_min = vdupq_n_u8(src_min);
  float32x4_t vec_mult = vdupq_n_f32(multiplier);

  uint16_t src_index = 0;
  bool done = false;
  while(!done) {
    // calc using a q register
    uint8x16_t chunk8 = vld1q_u8(src_data + src_index);

    // subtract off min value
    if(src_min > 0) {
      // src_min == 0 is a common case, so optimize it away
      chunk8 = vsubq_u8(chunk8, vec_min);
    }

    // scale up from uint8_t to uint32_t
    uint16x8_t chunk16l = vmovl_u8(vget_low_u8(chunk8));
    uint16x8_t chunk16h = vmovl_u8(vget_high_u8(chunk8));
    uint32x4_t chunk32ll = vmovl_u16(vget_low_u16(chunk16l));
    uint32x4_t chunk32lh = vmovl_u16(vget_high_u16(chunk16l));
    uint32x4_t chunk32hl = vmovl_u16(vget_low_u16(chunk16h));
    uint32x4_t chunk32hh = vmovl_u16(vget_high_u16(chunk16h));

    // convert from uint32_t to float32_t
    float32x4_t float32ll = vcvtq_f32_u32(chunk32ll);
    float32x4_t float32lh = vcvtq_f32_u32(chunk32lh);
    float32x4_t float32hl = vcvtq_f32_u32(chunk32hl);
    float32x4_t float32hh = vcvtq_f32_u32(chunk32hh);

    // multiply out by the normalization factor
    float32ll = vmulq_f32(float32ll, vec_mult);
    float32lh = vmulq_f32(float32lh, vec_mult);
    float32hl = vmulq_f32(float32hl, vec_mult);
    float32hh = vmulq_f32(float32hh, vec_mult);

    // store to the destination
    vst1q_f32(dst_data + src_index, float32ll);
    vst1q_f32(dst_data + src_index + 4, float32lh);
    vst1q_f32(dst_data + src_index + 8, float32hl);
    vst1q_f32(dst_data + src_index + 12, float32hh);

    src_index += kQRegisterElements8;

    if(src_index == src_size.width) {
      done = true;
    } else if(src_index > src_size.width - kQRegisterElements8) {
      // backtrack to handle leftovers
      src_index = (uint16_t)(src_size.width - kQRegisterElements8);
    }
  }
#endif
}

DMZ_INTERNAL void llcv_norm_convert_1d_u8_to_f32_c(IplImage *src, IplImage *dst) {
  cvConvertScale(src, dst, 1.0f / 255.0f, 0);
  cvNormalize(dst, dst, 0.0f, 1.0f, CV_MINMAX, NULL);
}

DMZ_INTERNAL void llcv_norm_convert_1d_u8_to_f32(IplImage *src, IplImage *dst) {
  assert(src->depth == IPL_DEPTH_8U);
  assert(dst->depth == IPL_DEPTH_32F);
  assert(src->nChannels == 1);
  assert(dst->nChannels == 1);
#if DMZ_DEBUG
  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
  assert(src_size.height == dst_size.height);
  assert(src_size.height == 1);  // 1d!
  assert(src_size.width >= kQRegisterElements8);
  assert(src_size.width == dst_size.width);
#endif
  
  if(dmz_has_neon_runtime()) {
    llcv_norm_convert_1d_u8_to_f32_neon(src, dst);
#if TEST_NORM_CONVERT
    IplImage *dst_c = cvCreateImage(dst_size, IPL_DEPTH_32F, 1);
    
    llcv_norm_convert_1d_u8_to_f32_c(src, dst_c);
    
    IplImage *delta = cvCreateImage(dst_size, IPL_DEPTH_32F, 1);
    
    cvAbsDiff(dst, dst_c, delta);
    float epsilon = 0.0001f;
    cvThreshold(delta, delta, epsilon, 1.0, CV_THRESH_TOZERO);
    
    int n_errors = cvCountNonZero(delta);
    if(n_errors > 0) {
      fprintf(stderr, "llcv_norm_convert_1d_u8_to_f32 errors: %i\n", n_errors);
      for(uint16_t x = 0; x < dst_size.width; x++) {
        if(cvGet1D(delta, x).val[0] > epsilon) {
          fprintf(stderr, "<%i: %f, %f, %f>, ", x, cvGet1D(src, x).val[0], cvGet1D(dst, x).val[0], cvGet1D(dst_c, x).val[0]);
        }
      }
      fprintf(stderr, "\n");
    }
    
    cvReleaseImage(&dst_c);
    cvReleaseImage(&delta);
#endif
  } else {
    llcv_norm_convert_1d_u8_to_f32_c(src, dst);
  }
}

#define TEST_YCbCr2RGB 0
#define TIME_YCbCr2RGB 0

#if TIME_YCbCr2RGB
static clock_t fastest_c = CLOCKS_PER_SEC * 1000;
static clock_t fastest_opencv = CLOCKS_PER_SEC * 1000;
#define TIME_YCbCr2RGB_TIMING_ITERATIONS 100
#endif

#if TEST_YCbCr2RGB
DMZ_INTERNAL void llcv_YCbCr2RGB_u8_opencv(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst) {
  IplImage *merged = cvCreateImage(cvGetSize(y), y->depth, 3);
  cvMerge(y, cr, cb, NULL, merged);
  cvCvtColor(merged, dst, CV_YCrCb2RGB);
  cvReleaseImage(&merged);
}
#endif

DMZ_INTERNAL void llcv_YCbCr2RGB_u8_c(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst) {
  // Could vectorize this, but the math gets ugly, and we only do it once, and really, it's fast enough.
#define DESCALE_14(x) ((x + (1 << 13)) >> 14)
#define SATURATED_BYTE(x) (uint8_t)((x < 0) ? 0 : ((x > 255) ? 255 : x))

  bool addAlpha = (dst->nChannels == 4);

  CvSize src_size = cvGetSize(y);
  
  uint8_t *y_data_origin = (uint8_t *)llcv_get_data_origin(y);
  uint16_t y_width_step = (uint16_t)y->widthStep;

  uint8_t *cb_data_origin = (uint8_t *)llcv_get_data_origin(cb);
  uint16_t cb_width_step = (uint16_t)cb->widthStep;

  uint8_t *cr_data_origin = (uint8_t *)llcv_get_data_origin(cr);
  uint16_t cr_width_step = (uint16_t)cr->widthStep;

  uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
  uint16_t dst_width_step = (uint16_t)dst->widthStep;

  for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
    const uint8_t *y_row_origin = y_data_origin + row_index * y_width_step;
    const uint8_t *cb_row_origin = cb_data_origin + row_index * cb_width_step;
    const uint8_t *cr_row_origin = cr_data_origin + row_index * cr_width_step;
    
    uint8_t *dst_row_origin = dst_data_origin + row_index * dst_width_step;
    
    uint16_t col_index = 0;
    while(col_index < src_size.width) {
      uint8_t pix_y = y_row_origin[col_index];
      uint8_t pix_cb = cb_row_origin[col_index];
      uint8_t pix_cr = cr_row_origin[col_index];
      int8_t sCb = pix_cb - 128;
      int8_t sCr = pix_cr - 128;
      int32_t pix_b = pix_y + DESCALE_14(sCb * 29049);
      int32_t pix_g = pix_y + DESCALE_14(sCb * -5636 + sCr * -11698);
      int32_t pix_r = pix_y + DESCALE_14(sCr * 22987);

      uint16_t col_pixel_pos = (uint16_t)(col_index * dst->nChannels);

      // the SATURATED_BYTE macro is necessary to ensure that we're really only writing one
      // byte, and that we stay within it's limits. It appears that the clang (and possibly
      // gcc 4.6 vs 4.4.3) differ in how the shift/cast combo behaves.
      dst_row_origin[col_pixel_pos] = SATURATED_BYTE(pix_r);
      dst_row_origin[col_pixel_pos + 1] = SATURATED_BYTE(pix_g);
      dst_row_origin[col_pixel_pos + 2] = SATURATED_BYTE(pix_b);

      if (addAlpha) {
        dst_row_origin[col_pixel_pos + 3] = 0xff; // make an opaque image
      }

      col_index++;
    }
  }
}

DMZ_INTERNAL void llcv_YCbCr2RGB_u8(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst) {
#if DMZ_DEBUG
  CvSize y_size = cvGetSize(y);
  CvSize cb_size = cvGetSize(cb);
  CvSize cr_size = cvGetSize(cr);
  CvSize dst_size = cvGetSize(dst);

  assert(y_size.width == cb_size.width);
  assert(y_size.height == cb_size.height);
  assert(cb_size.width == cr_size.width);
  assert(cb_size.height == cr_size.height);
  assert(dst_size.width == y_size.width);
  assert(dst_size.height == y_size.height);
#endif
  
  assert(y->nChannels == 1);
  assert(cb->nChannels == 1);
  assert(cr->nChannels == 1);
  assert(dst->nChannels == 3 || dst->nChannels == 4);

  assert(y->depth == IPL_DEPTH_8U);
  assert(cb->depth == IPL_DEPTH_8U);
  assert(cr->depth == IPL_DEPTH_8U);
  assert(dst->depth == IPL_DEPTH_8U);


#if TIME_YCbCr2RGB
  clock_t start_c, end_c;
  start_c = clock();
  for(int iter = 0; iter < TIME_YCbCr2RGB_TIMING_ITERATIONS; iter++) {
#endif
    
    llcv_YCbCr2RGB_u8_c(y, cb, cr, dst);
    
#if TIME_YCbCr2RGB
  }
  end_c = clock();
  clock_t elapsed_c = end_c - start_c;
  if(elapsed_c < fastest_c) {
    fastest_c = elapsed_c;
    dmz_debug_log("(llcv_YCbCr2RGB_u8) fastest c: %f ms", (1000.0 * (double)fastest_c / (double)CLOCKS_PER_SEC) / (double)TIME_YCbCr2RGB_TIMING_ITERATIONS);
  }
#endif
  
  // Start test-only code
  
#if TEST_YCbCr2RGB
  IplImage *opencv_dst = cvCreateImage(dst_size, dst->depth, dst->nChannels);
#if TIME_YCbCr2RGB
  clock_t start_opencv, end_opencv;
  start_opencv = clock();
  for(int iter = 0; iter < TIME_YCbCr2RGB_TIMING_ITERATIONS; iter++) {
#endif
    llcv_YCbCr2RGB_u8_opencv(y, cb, cr, opencv_dst);
#if TIME_YCbCr2RGB
  }
  end_opencv = clock();
  clock_t elapsed_opencv = end_opencv - start_opencv;
  if(elapsed_opencv < fastest_opencv) {
    fastest_opencv = elapsed_opencv;
    dmz_debug_log("(llcv_YCbCr2RGB_u8) fastest opencv: %f ms", (1000.0 * (double)fastest_opencv / (double)CLOCKS_PER_SEC) / (double)TIME_YCbCr2RGB_TIMING_ITERATIONS);
  }
#endif

  IplImage *delta = cvCreateImage(dst_size, dst->depth, dst->nChannels);

  cvSub(dst, opencv_dst, delta);

  IplImage *delta_r = cvCreateImage(dst_size, dst->depth, 1);
  IplImage *delta_g = cvCreateImage(dst_size, dst->depth, 1);
  IplImage *delta_b = cvCreateImage(dst_size, dst->depth, 1);
  cvSplit(delta, delta_r, delta_g, delta_b, NULL);
  
  int n_errors_r = cvCountNonZero(delta_r);
  int n_errors_g = cvCountNonZero(delta_g);
  int n_errors_b = cvCountNonZero(delta_b);
  
  if(n_errors_r > 0 || n_errors_g > 0 || n_errors_b > 0) {
    dmz_debug_log("(llcv_YCbCr2RGB_u8) errors: %i, %i, %i", n_errors_r, n_errors_g, n_errors_b);
  } else {
    // dmz_debug_log("(llcv_YCbCr2RGB_u8) ok");
  }
  
  cvReleaseImage(&delta);
  cvReleaseImage(&delta_r);
  cvReleaseImage(&delta_g);
  cvReleaseImage(&delta_b);
  cvReleaseImage(&opencv_dst);
#endif  // TEST_YCbCr2RGB

}


#endif
