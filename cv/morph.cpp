//
//  morph.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "morph.h"
#include "image_util.h"
#include "neon.h"
#include "processor_support.h"
#include "dmz_debug.h"

#if DMZ_HAS_NEON_COMPILETIME
#include <arm_neon.h>
#endif

#if DMZ_HAS_NEON_COMPILETIME
// Morph NEON support functions

static inline void vec_morph3_1d_u8_q(const uint8_t *src, uint8_t *dst) {
  asm volatile
  (
   // load src pixel vectors (src - 1, src, src + 1)
   "mov r0, %[src]" "\n\t"
   "sub r0, r0, #1" "\n\t"
   "vld1.8 {q0}, [r0]" "\n\t"
   "add r0, r0, #1" "\n\t"
   "vld1.8 {q1}, [r0]" "\n\t"
   "add r0, r0, #1" "\n\t"
   "vld1.8 {q2}, [r0]" "\n\t"

   // pairwise max, pairwise min
   "vmax.u8 q3, q0, q1" "\n\t"
   "vmin.u8 q8, q0, q1" "\n\t"
   "vmax.u8 q9, q3, q2" "\n\t"
   "vmin.u8 q10, q8, q2" "\n\t"

   // subtract
   "vsub.u8 q11, q9, q10" "\n\t"

   // write to dst
   "mov r0, %[dst]" "\n\t"
   "vst1.8 {q11}, [r0]" "\n\t"
   
   : // output
   
   : // input
   [src]"r" (src),
   [dst]"r" (dst)
   
   : // clobbered
   "r0", // used for src, dst pointers
   // skip q4-q7, since then gcc will have to save them, see http://stackoverflow.com/questions/261419/arm-to-c-calling-convention-registers-to-save
   "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", //
   "memory"
   );
}

#endif

DMZ_INTERNAL void llcv_morph_grad3_1d_u8_neon(IplImage *src, IplImage *dst) {
#if DMZ_HAS_NEON_COMPILETIME

  uint8_t *src_data = (uint8_t *)src->imageData;
  if(dmz_likely(NULL != src->roi)) {
    src_data += src->roi->yOffset * src->widthStep + src->roi->xOffset * sizeof(uint8_t);
  }

  uint8_t *dst_data = (uint8_t *)dst->imageData;
  if(dmz_unlikely(NULL != dst->roi)) {
    dst_data += dst->roi->yOffset * dst->widthStep + dst->roi->xOffset * sizeof(uint8_t);
  }

  CvSize dst_size = cvGetSize(dst);

  // Calc first output (special, due to edge effects)
  uint16_t dst_index = 0;
  uint8_t f0 = src_data[dst_index];
  uint8_t f1 = src_data[dst_index + 1];  // safe b/c already asserted src width > 1
  dst_data[0] = MAX(f0, f1) - MIN(f0, f1);
  dst_index++;

  // + 1 in the below because size 3 kernel involves dst_index - 1, dst_index, dst_index + 1 (and we can assume dst_index - 1 is available)
  // strictly < below, because the last pixel is special
  bool done = false;
  while(!done) {
     // calc using q registers
    vec_morph3_1d_u8_q(src_data + dst_index, dst_data + dst_index);
    dst_index += kQRegisterElements8;

    if(dst_index + 1 == dst_size.width) {
      done = true;
    } else if(dst_index + 1 > dst_size.width - kQRegisterElements8) {
      // backtrack to handle leftovers
      dst_index = (uint16_t)(dst_size.width - kQRegisterElements8 - 1);
    }
  }

  // Calc last output (special, due to edge effects)
  uint8_t l0 = src_data[dst_index - 1];
  uint8_t l1 = src_data[dst_index];
  dst_data[dst_index] = MAX(l0, l1) - MIN(l0, l1);
#endif
}

DMZ_INTERNAL void llcv_morph_grad3_1d_u8_c(IplImage *src, IplImage *dst) {
  IplConvKernel *kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
  cvMorphologyEx(src, dst, NULL, kernel, CV_MOP_GRADIENT, 1);
  cvReleaseStructuringElement(&kernel);
}

#define TEST_MORPH_NEON 0

DMZ_INTERNAL void llcv_morph_grad3_1d_u8(IplImage *src, IplImage *dst) {
  assert(src->depth == IPL_DEPTH_8U);
  assert(dst->depth == IPL_DEPTH_8U);
  assert(src != dst);
  assert(src->nChannels == 1);
  assert(dst->nChannels == 1);
#if DMZ_DEBUG
  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
  assert(src_size.width == dst_size.width);
  assert(src_size.height == dst_size.height);
  assert(src_size.height == 1);  // 1d!
  assert(src_size.width > 1); // sanity
#endif
  
  if(dmz_has_neon_runtime()) {
    llcv_morph_grad3_1d_u8_neon(src, dst);
#if TEST_MORPH_NEON
    IplImage *dst_c = cvCreateImage(dst_size, IPL_DEPTH_8U, 1);
    
    llcv_morph_grad3_1d_u8_c(src, dst_c);
    
    IplImage *delta = cvCreateImage(dst_size, IPL_DEPTH_8U, 1);
    
    cvAbsDiff(dst, dst_c, delta);
    
    int n_errors = cvCountNonZero(delta);
    if(n_errors > 0) {
      fprintf(stderr, "llcv_morph_grad3_1d_u8 errors: %i\n", n_errors);
    }
    
    cvReleaseImage(&dst_c);
    cvReleaseImage(&delta);
#endif
  } else {
    llcv_morph_grad3_1d_u8_c(src, dst);
  }
}



#define TEST_MORPH2D 0
#define TIME_MORPH2D 0

#if TIME_MORPH2D
static clock_t fastest_c_neon = CLOCKS_PER_SEC * 1000;
static clock_t fastest_opencv = CLOCKS_PER_SEC * 1000;
#define TIME_MORPH2D_TIMING_ITERATIONS 1000
#endif

#if TEST_MORPH2D
DMZ_INTERNAL void llcv_morph_grad3_2d_cross_u8_opencv(IplImage *src, IplImage *dst) {
  IplConvKernel *kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
  cvMorphologyEx(src, dst, NULL, kernel, CV_MOP_GRADIENT, 1);
  cvReleaseStructuringElement(&kernel);
}
#endif

#define MAX5(a, b, c, d, e) MAX(a, MAX(b, MAX(c, MAX(d, e))))
#define MIN5(a, b, c, d, e) MIN(a, MIN(b, MIN(c, MIN(d, e))))

DMZ_INTERNAL void llcv_morph_grad3_2d_cross_u8_c_neon(IplImage *src, IplImage *dst) {
#define kMorphGrad3Cross2DVectorSize 16

  CvSize src_size = cvGetSize(src);
  assert(src_size.width > kMorphGrad3Cross2DVectorSize);

  uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
  uint16_t src_width_step = (uint16_t)src->widthStep;
  
  uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
  uint16_t dst_width_step = (uint16_t)dst->widthStep;
  bool can_use_neon = dmz_has_neon_runtime();
  
  for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
    uint16_t row1_index = row_index == 0 ? row_index : row_index - 1;
    uint16_t row2_index = row_index;
    uint16_t row3_index = row_index == src_size.height - 1 ? row_index : row_index + 1;
    
    const uint8_t *src_row1_origin = src_data_origin + row1_index * src_width_step;
    const uint8_t *src_row2_origin = src_data_origin + row2_index * src_width_step;
    const uint8_t *src_row3_origin = src_data_origin + row3_index * src_width_step;
    
    uint8_t *dst_row_origin = dst_data_origin + row_index * dst_width_step;
    
    uint16_t col_index = 0;
    while(col_index < src_size.width) {
      bool is_first_col = col_index == 0;
      uint16_t last_col_index = (uint16_t)(src_size.width - 1);
      bool is_last_col = col_index == last_col_index;
      bool can_process_next_chunk_as_vector = col_index + kMorphGrad3Cross2DVectorSize < last_col_index;
      if(is_first_col || is_last_col || !can_use_neon || !can_process_next_chunk_as_vector) {
        // scalar step
        uint16_t col1_index = is_first_col ? col_index : col_index - 1;
        uint16_t col2_index = col_index;
        uint16_t col3_index = is_last_col ? col_index : col_index + 1;
        
        uint8_t grad =
        MAX5(src_row1_origin[col2_index], src_row2_origin[col1_index], src_row2_origin[col2_index], src_row2_origin[col3_index], src_row3_origin[col2_index]) -
        MIN5(src_row1_origin[col2_index], src_row2_origin[col1_index], src_row2_origin[col2_index], src_row2_origin[col3_index], src_row3_origin[col2_index]);
        
        // write result
        dst_row_origin[col_index] = grad;
        
        col_index++;
      } else {
        // vector step
#if DMZ_HAS_NEON_COMPILETIME
        // north, east, center, west, south
        uint8x16_t n = vld1q_u8(src_row1_origin + col_index);
        uint8x16_t w = vld1q_u8(src_row2_origin + col_index - 1);
        uint8x16_t c = vld1q_u8(src_row2_origin + col_index);
        uint8x16_t e = vld1q_u8(src_row2_origin + col_index + 1);
        uint8x16_t s = vld1q_u8(src_row3_origin + col_index);
        uint8x16_t max_vec = vmaxq_u8(n, vmaxq_u8(w, vmaxq_u8(c, vmaxq_u8(e, s))));
        uint8x16_t min_vec = vminq_u8(n, vminq_u8(w, vminq_u8(c, vminq_u8(e, s))));
        uint8x16_t grad_vec = vsubq_u8(max_vec, min_vec);
        dst_row_origin[col_index +  0] = vgetq_lane_u8(grad_vec,  0);
        dst_row_origin[col_index +  1] = vgetq_lane_u8(grad_vec,  1);
        dst_row_origin[col_index +  2] = vgetq_lane_u8(grad_vec,  2);
        dst_row_origin[col_index +  3] = vgetq_lane_u8(grad_vec,  3);
        dst_row_origin[col_index +  4] = vgetq_lane_u8(grad_vec,  4);
        dst_row_origin[col_index +  5] = vgetq_lane_u8(grad_vec,  5);
        dst_row_origin[col_index +  6] = vgetq_lane_u8(grad_vec,  6);
        dst_row_origin[col_index +  7] = vgetq_lane_u8(grad_vec,  7);
        dst_row_origin[col_index +  8] = vgetq_lane_u8(grad_vec,  8);
        dst_row_origin[col_index +  9] = vgetq_lane_u8(grad_vec,  9);
        dst_row_origin[col_index + 10] = vgetq_lane_u8(grad_vec, 10);
        dst_row_origin[col_index + 11] = vgetq_lane_u8(grad_vec, 11);
        dst_row_origin[col_index + 12] = vgetq_lane_u8(grad_vec, 12);
        dst_row_origin[col_index + 13] = vgetq_lane_u8(grad_vec, 13);
        dst_row_origin[col_index + 14] = vgetq_lane_u8(grad_vec, 14);
        dst_row_origin[col_index + 15] = vgetq_lane_u8(grad_vec, 15);
        col_index += kMorphGrad3Cross2DVectorSize;
#endif
      }
    }
  }
#undef kMorphGrad3Cross2DVectorSize
}

DMZ_INTERNAL void llcv_morph_grad3_2d_cross_u8(IplImage *src, IplImage *dst) {
#if DMZ_DEBUG
  assert(src->nChannels == 1);
  assert(src->depth == IPL_DEPTH_8U);
  
  assert(dst->nChannels == 1);
  assert(dst->depth == IPL_DEPTH_8U);
  
  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
  
  assert(dst_size.width == src_size.width);
  assert(dst_size.height == src_size.height);
#endif
  
#if TIME_MORPH2D
  clock_t start_c_neon, end_c_neon;
  start_c_neon = clock();
  for(int iter = 0; iter < TIME_MORPH2D_TIMING_ITERATIONS; iter++) {
#endif
    
    llcv_morph_grad3_2d_cross_u8_c_neon(src, dst);
    
#if TIME_MORPH2D
  }
  end_c_neon = clock();
  clock_t elapsed_c_neon = end_c_neon - start_c_neon;
  if(elapsed_c_neon < fastest_c_neon) {
    fastest_c_neon = elapsed_c_neon;
    dmz_debug_log("fastest c/neon %f ms", (1000.0 * (double)fastest_c_neon / (double)CLOCKS_PER_SEC) / (double)TIME_MORPH2D_TIMING_ITERATIONS);
  }
#endif
  
  // Start test-only code
  
#if TEST_MORPH2D
  IplImage *opencv_dst = cvCreateImage(dst_size, dst->depth, dst->nChannels);
#if TIME_MORPH2D
  clock_t start_opencv, end_opencv;
  start_opencv = clock();
  for(int iter = 0; iter < TIME_MORPH2D_TIMING_ITERATIONS; iter++) {
#endif
    llcv_morph_grad3_2d_cross_u8_opencv(src, opencv_dst);
#if TIME_MORPH2D
  }
  end_opencv = clock();
  clock_t elapsed_opencv = end_opencv - start_opencv;
  if(elapsed_opencv < fastest_opencv) {
    fastest_opencv = elapsed_opencv;
    dmz_debug_log("fastest opencv %f ms", (1000.0 * (double)fastest_opencv / (double)CLOCKS_PER_SEC) / (double)TIME_MORPH2D_TIMING_ITERATIONS);
  }
#endif
  
  IplImage *delta = cvCreateImage(dst_size, dst->depth, dst->nChannels);
  
  cvSub(dst, opencv_dst, delta);
  
  int n_errors = cvCountNonZero(delta);
  if(n_errors > 0) {
    dmz_debug_log("llcv_morph_grad3_2d_cross_u8 errors: %i", n_errors);
  } else {
    // dmz_debug_log("llcv_morph_grad3_2d_cross_u8 ok");
  }
  
  cvReleaseImage(&delta);
  cvReleaseImage(&opencv_dst);
#endif  // TEST_MORPH2D
}


#endif
