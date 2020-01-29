//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include "sobel.h"
#include "processor_support.h"
#include "image_util.h"
#include "eigen.h"
#include "dmz_debug.h"

#if DMZ_HAS_NEON_COMPILETIME

#include <arm_neon.h>

static inline void convolve_eight_pixels_u8(const uint8_t *source_pixels, size_t source_stride, int16_t *dest_pixel, size_t dest_stride, const int16_t *kernel) {
  asm volatile
  (

   // load kernel
   "vld1.16 {q15}, [%[kernel]]" "\n\t"

   // load pixels
   "mov r4, %[source_pixels]" "\n\t"
   "vld1.8 {d0}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d2}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d4}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d6}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d8}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d10}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d12}, [r4], %[source_stride]" "\n\t"
   "vld1.8 {d14}, [r4], %[source_stride]" "\n\t"

   // expand pixels
   "vmovl.u8 q0,  d0" "\n\t"
   "vmovl.u8 q1,  d2" "\n\t"
   "vmovl.u8 q2,  d4" "\n\t"
   "vmovl.u8 q3,  d6" "\n\t"
   "vmovl.u8 q4,  d8" "\n\t"
   "vmovl.u8 q5, d10" "\n\t"
   "vmovl.u8 q6, d12" "\n\t"
   "vmovl.u8 q7, d14" "\n\t"

   // multiply kernel by pixels
   "vmul.s16 q0, q0, q15" "\n\t"
   "vmul.s16 q1, q1, q15" "\n\t"
   "vmul.s16 q2, q2, q15" "\n\t"
   "vmul.s16 q3, q3, q15" "\n\t"
   "vmul.s16 q4, q4, q15" "\n\t"
   "vmul.s16 q5, q5, q15" "\n\t"
   "vmul.s16 q6, q6, q15" "\n\t"
   "vmul.s16 q7, q7, q15" "\n\t"

   // Pairwise add -- first sum reduction -- after this, each q contains 4 int32_ts
   "vpaddl.s16 q0, q0" "\n\t"
   "vpaddl.s16 q1, q1" "\n\t"
   "vpaddl.s16 q2, q2" "\n\t"
   "vpaddl.s16 q3, q3" "\n\t"
   "vpaddl.s16 q4, q4" "\n\t"
   "vpaddl.s16 q5, q5" "\n\t"
   "vpaddl.s16 q6, q6" "\n\t"
   "vpaddl.s16 q7, q7" "\n\t"

   // Add d registers -- second sum reduction -- move to new d registers starting at q8 / d16, combining pairs of output values
   "vadd.s32 d16,  d0,  d1" "\n\t"
   "vadd.s32 d17,  d2,  d3" "\n\t"
   "vadd.s32 d18,  d4,  d5" "\n\t"
   "vadd.s32 d19,  d6,  d7" "\n\t"
   "vadd.s32 d20,  d8,  d9" "\n\t"
   "vadd.s32 d21, d10, d11" "\n\t"
   "vadd.s32 d22, d12, d13" "\n\t"
   "vadd.s32 d23, d14, d15" "\n\t"

   // Pairwise add -- third sum reduction -- after this, each q contains 2 int64_ts
   "vpaddl.s32  q8,  q8" "\n\t"
   "vpaddl.s32  q9,  q9" "\n\t"
   "vpaddl.s32 q10, q10" "\n\t"
   "vpaddl.s32 q11, q11" "\n\t"

   // Narrow -- after this, each q contains 4 int32_ts
   "vqmovn.s64 d24, q8" "\n\t"
   "vqmovn.s64 d25, q9" "\n\t"
   "vqmovn.s64 d26, q10" "\n\t"
   "vqmovn.s64 d27, q11" "\n\t"

   // Narrow again -- after this, each q contains 8 int16_ts!
   "vqmovn.s32 d28,  q12" "\n\t"
   "vqmovn.s32 d29,  q13" "\n\t"

   // Write to their destinations!
   "mov r4, %[dest_pixel]" "\n\t"
   "vst1.16 {d28[0]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d28[1]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d28[2]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d28[3]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d29[0]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d29[1]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d29[2]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d29[3]}, [r4], %[dest_stride]" "\n\t"

   : // output

   : // input
   [source_pixels]"r" (source_pixels),
   [source_stride]"r" (source_stride),
   [dest_pixel]"r" (dest_pixel),
   [dest_stride]"r" (dest_stride),
   [kernel]"r" (kernel)

   : // clobbered
   "r4", // mutable copies of source_pixels, dest_pixel; TODO: how to simply specify that these are read/write?
   "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", // contain data loaded from the source pixels
   "q8", "q9", "q10", "q11", // reduced data (pairs)
   "q12", "q13", // reduced data (quads)
   "q14", // reduced data (oct)
   "q15", // kernel
   "memory"

   );
}


static inline void convolve_four_pixels_s16(const int16_t *source_pixels, size_t source_stride, int16_t *dest_pixel, size_t dest_stride, const int16_t *kernel) {

  assert(source_pixels != NULL);
  assert(dest_pixel != NULL);
  assert(kernel != NULL);

  asm volatile
  (

   // load kernel
   "vld1.16 {q15}, [%[kernel]]" "\n\t"

   // load pixels
   "mov r4, %[source_pixels]" "\n\t"
   "vld1.s16 {q0}, [r4], %[source_stride]" "\n\t"
   "vld1.s16 {q1}, [r4], %[source_stride]" "\n\t"
   "vld1.s16 {q2}, [r4], %[source_stride]" "\n\t"
   "vld1.s16 {q3}, [r4], %[source_stride]" "\n\t"

   // multiply kernel by pixels
   "vmull.s16  q4, d0, d30" "\n\t"
   "vmull.s16  q5, d1, d31" "\n\t"
   "vmull.s16  q6, d2, d30" "\n\t"
   "vmull.s16  q7, d3, d31" "\n\t"
   "vmull.s16  q8, d4, d30" "\n\t"
   "vmull.s16  q9, d5, d31" "\n\t"
   "vmull.s16 q10, d6, d30" "\n\t"
   "vmull.s16 q11, d7, d31" "\n\t"

   // Pairwise add -- first sum reduction -- after this, each q contains 2 int64_ts
   "vpaddl.s32  q4,  q4" "\n\t"
   "vpaddl.s32  q5,  q5" "\n\t"
   "vpaddl.s32  q6,  q6" "\n\t"
   "vpaddl.s32  q7,  q7" "\n\t"
   "vpaddl.s32  q8,  q8" "\n\t"
   "vpaddl.s32  q9,  q9" "\n\t"
   "vpaddl.s32 q10, q10" "\n\t"
   "vpaddl.s32 q11, q11" "\n\t"

   // Add q registers -- second sum reduction
   "vadd.s64 q0,  q4,  q5" "\n\t"
   "vadd.s64 q1,  q6,  q7" "\n\t"
   "vadd.s64 q2,  q8,  q9" "\n\t"
   "vadd.s64 q3, q10, q11" "\n\t"

   // Add d registers -- third sum reduction -- move to new, adjacent registers (q4, q5) to prepare for narrowing
   "vadd.s64  d8, d0, d1" "\n\t"
   "vadd.s64  d9, d2, d3" "\n\t"
   "vadd.s64 d10, d4, d5" "\n\t"
   "vadd.s64 d11, d6, d7" "\n\t"

   // Narrow -- after this, q6 contains 4 int32_ts
   "vqmovn.s64 d12, q4" "\n\t"
   "vqmovn.s64 d13, q5" "\n\t"

   // Narrow again -- after this, each d14 contains 4 int16_ts!
   "vqmovn.s32 d14, q6" "\n\t"

   // Write to their destinations!
   "mov r4, %[dest_pixel]" "\n\t"
   "vst1.16 {d14[0]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d14[1]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d14[2]}, [r4], %[dest_stride]" "\n\t"
   "vst1.16 {d14[3]}, [r4], %[dest_stride]" "\n\t"

   : // output

   : // input
   [source_pixels]"r" (source_pixels),
   [source_stride]"r" (source_stride),
   [dest_pixel]"r" (dest_pixel),
   [dest_stride]"r" (dest_stride),
   [kernel]"r" (kernel)

   : // clobbered
   "r4", // mutable copies of source_pixels, dest_pixel; TODO: how to simply specify that these are read/write?
   "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", // contain data loaded from the source pixels
   "q8", "q9", "q10", "q11", // reduced data (pairs)
   "q12", "q13", // reduced data (quads)
   "q14", // reduced data (oct)
   "q15", // kernel
   "memory"

   );
}


static inline void edge_convolve_u8(const uint8_t *source_pixels, int16_t *dest_pixel, size_t dest_stride, const int16_t *kernels, size_t kernel_stride) {
  asm volatile
  (
   // load pixels
   "vld1.8 {d0}, [%[source_pixels]]" "\n\t"

   // load kernels (part 1)
   "mov r0, %[kernels]" "\n\t"
   "vld1.16 {q12}, [r0]" "\n\t"
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q13}, [r0]" "\n\t"

   // expand pixels
   "vmovl.u8 q0,  d0" "\n\t"

   // load kernels (part 2)
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q14}, [r0]" "\n\t"
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q15}, [r0]" "\n\t"

   // multiply kernel by pixels
   "vmul.s16 q1, q0, q12" "\n\t"
   "vmul.s16 q2, q0, q13" "\n\t"
   "vmul.s16 q3, q0, q14" "\n\t"
   "vmul.s16 q4, q0, q15" "\n\t"

   // Pairwise add -- first sum reduction -- after this, each q contains 4 int32_ts
   "vpaddl.s16 q1, q1" "\n\t"
   "vpaddl.s16 q2, q2" "\n\t"
   "vpaddl.s16 q3, q3" "\n\t"
   "vpaddl.s16 q4, q4" "\n\t"

   // Add d registers -- second sum reduction -- move to new d registers starting at q5 / d10, combining pairs of output values
   "vadd.s32 d10,  d2,  d3" "\n\t"
   "vadd.s32 d11,  d4,  d5" "\n\t"
   "vadd.s32 d12,  d6,  d7" "\n\t"
   "vadd.s32 d13,  d8,  d9" "\n\t"

   // Pairwise add -- third sum reduction -- after this, each q contains 2 int64_ts
   "vpaddl.s32  q7,  q5" "\n\t"
   "vpaddl.s32  q8,  q6" "\n\t"

   // Narrow -- after this, each q contains 4 int32_ts
   "vqmovn.s64 d18, q7" "\n\t"
   "vqmovn.s64 d19, q8" "\n\t"

   // Narrow again -- after this, each q contains 8 int16_ts!
   "vqmovn.s32 d20,  q9" "\n\t"

   // Write to their destinations!
   "mov r0, %[dest_pixel]" "\n\t"
   "vst1.16 {d20[0]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[1]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[2]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[3]}, [r0]" "\n\t"

   : // output

   : // input
   [source_pixels]"r" (source_pixels),
   [dest_pixel]"r" (dest_pixel),
   [dest_stride]"r" (dest_stride),
   [kernels]"r" (kernels),
   [kernel_stride]"r" (kernel_stride)

   : // clobbered
   "r0", // mutable copies of dest_pixel, kernels; TODO: how to simply specify that these are read/write?
   "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", //
   "q8", "q9", "q10", "q11", //
   "q12", "q13", //
   "q14", //
   "q15", //
   "memory"

   );
}

static inline void edge_convolve_s16(const int16_t *source_pixels, int16_t *dest_pixel, size_t dest_stride, const int16_t *kernels, size_t kernel_stride) {
  asm volatile
  (
   // load pixels
   "vld1.s16 {q0}, [%[source_pixels]]" "\n\t"

   // load kernels
   "mov r0, %[kernels]" "\n\t"
   "vld1.16 {q12}, [r0]" "\n\t"
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q13}, [r0]" "\n\t"
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q14}, [r0]" "\n\t"
   "add r0, r0, %[kernel_stride]" "\n\t"
   "vld1.16 {q15}, [r0]" "\n\t"

   // multiply kernel by pixels
   "vmull.s16 q1, d0, d24" "\n\t"
   "vmull.s16 q2, d1, d25" "\n\t"
   "vmull.s16 q3, d0, d26" "\n\t"
   "vmull.s16 q4, d1, d27" "\n\t"
   "vmull.s16 q5, d0, d28" "\n\t"
   "vmull.s16 q6, d1, d29" "\n\t"
   "vmull.s16 q7, d0, d30" "\n\t"
   "vmull.s16 q8, d1, d31" "\n\t"

   // Pairwise add -- first sum reduction -- after this, each q contains 2 int64_ts
   "vpaddl.s32 q1, q1" "\n\t"
   "vpaddl.s32 q2, q2" "\n\t"
   "vpaddl.s32 q3, q3" "\n\t"
   "vpaddl.s32 q4, q4" "\n\t"
   "vpaddl.s32 q5, q5" "\n\t"
   "vpaddl.s32 q6, q6" "\n\t"
   "vpaddl.s32 q7, q7" "\n\t"
   "vpaddl.s32 q8, q8" "\n\t"

   // Add q registers -- second sum reduction
   "vadd.s64 q1, q1, q2" "\n\t"
   "vadd.s64 q3, q3, q4" "\n\t"
   "vadd.s64 q5, q5, q6" "\n\t"
   "vadd.s64 q7, q7, q8" "\n\t"

   // Add d registers -- third sum reduction -- move to new, adjacent registers (q9, q10) to prepare for narrowing
   "vadd.s64 d18,  d2,  d3" "\n\t"
   "vadd.s64 d19,  d6,  d7" "\n\t"
   "vadd.s64 d20, d10, d11" "\n\t"
   "vadd.s64 d21, d14, d15" "\n\t"

   // Narrow -- after this, each q contains 4 int32_ts
   "vqmovn.s64 d18, q9" "\n\t"
   "vqmovn.s64 d19, q10" "\n\t"

   // Narrow again -- after this, each q contains 8 int16_ts!
   "vqmovn.s32 d20,  q9" "\n\t"

   // Write to their destinations!
   "mov r0, %[dest_pixel]" "\n\t"
   "vst1.16 {d20[0]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[1]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[2]}, [r0]" "\n\t"
   "add r0, r0, %[dest_stride]" "\n\t"
   "vst1.16 {d20[3]}, [r0]" "\n\t"

   : // output

   : // input
   [source_pixels]"r" (source_pixels),
   [dest_pixel]"r" (dest_pixel),
   [dest_stride]"r" (dest_stride),
   [kernels]"r" (kernels),
   [kernel_stride]"r" (kernel_stride)

   : // clobbered
   "r0", // mutable copies of dest_pixel, kernels; TODO: how to simply specify that these are read/write?
   "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", //
   "q8", "q9", "q10", "q11", //
   "q12", "q13", //
   "q14", //
   "q15", //
   "memory"

   );
}

#pragma mark vectorized_convolve_transpose7

DMZ_INTERNAL void vectorized_convolve_transpose7(IplImage *image, IplImage *dest, int16_t *kernel7) {

#define kKernelSize 7
#define kStartBorderSize 3
#define kEndBorderSize 4

  CvSize imageSize = cvGetSize(image);
  assert(image->depth == IPL_DEPTH_8U || image->depth == IPL_DEPTH_16S);
  bool is_u8 = image->depth == IPL_DEPTH_8U;
  assert(image->width > kKernelSize);

  uint8_t *data_origin = (uint8_t *)llcv_get_data_origin(image);
  uint16_t data_width_step = (uint16_t)image->widthStep;

  uint8_t *dest_data = (uint8_t *)dest->imageData;
  uint16_t dest_width_step = (uint16_t)dest->widthStep;

  uint16_t n_internal_cols = (uint16_t)(imageSize.width - (kStartBorderSize + kEndBorderSize));
  uint8_t col_vector_size = is_u8 ? 8 : 4;
  uint8_t scalar_cols = n_internal_cols % col_vector_size;

  int16_t kernels[8][8] = {
    // leading edge replication
    {(int16_t)(kernel7[0] + kernel7[1] + kernel7[2] + kernel7[3]), kernel7[4], kernel7[5], kernel7[6],          0,          0, 0, 0},
    {(int16_t)(kernel7[0] + kernel7[1] + kernel7[2]),              kernel7[3], kernel7[4], kernel7[5], kernel7[6],          0, 0, 0},
    {(int16_t)(kernel7[0] + kernel7[1]),                           kernel7[2], kernel7[3], kernel7[4], kernel7[5], kernel7[6], 0, 0},

    // main body
    {kernel7[0], kernel7[1], kernel7[2], kernel7[3], kernel7[4], kernel7[5], kernel7[6], 0},

    // trailing edge replication
    {0, kernel7[0], kernel7[1], kernel7[2], kernel7[3], kernel7[4], kernel7[5], kernel7[6]},
    {0,          0, kernel7[0], kernel7[1], kernel7[2], kernel7[3], kernel7[4], (int16_t)(kernel7[5] + kernel7[6])},
    {0,          0,          0, kernel7[0], kernel7[1], kernel7[2], kernel7[3], (int16_t)(kernel7[4] + kernel7[5] + kernel7[6])},
    {0,          0,          0,          0, kernel7[0], kernel7[1], kernel7[2], (int16_t)(kernel7[3] + kernel7[4] + kernel7[5] + kernel7[6])}
  };

  for(uint16_t row_index = 0; row_index < imageSize.height; row_index++) {
    const uint8_t *row_origin = data_origin + row_index * data_width_step;
    uint16_t dest_col_index = row_index; // transposed!
    uint16_t dest_row_index; // transposed!

    int16_t *leading_edge_dest_pixel = (int16_t *)(dest_data /* + 0 * dest_width_step */) + dest_col_index;
    if(is_u8) {
      edge_convolve_u8(row_origin, leading_edge_dest_pixel, dest_width_step, kernels[0], sizeof(int16_t) * 8);
    }  else {
      edge_convolve_s16((const int16_t *)row_origin, leading_edge_dest_pixel, dest_width_step, kernels[0], sizeof(int16_t) * 8);
    }

    for(uint16_t internal_col_index = 0; internal_col_index < n_internal_cols / col_vector_size; internal_col_index++) {
      uint16_t col_index = internal_col_index * col_vector_size;
      dest_row_index = col_index + kStartBorderSize; // transposed!
      int16_t *dest_pixel = (int16_t *)(dest_data + dest_row_index * dest_width_step) + dest_col_index;
      if(is_u8) {
        const uint8_t *source_pixels = row_origin + col_index;
        convolve_eight_pixels_u8(source_pixels, sizeof(uint8_t), dest_pixel, dest_width_step, kernels[3]);
      } else {
        const int16_t *source_pixels = (int16_t *)row_origin + col_index;
        convolve_four_pixels_s16(source_pixels, sizeof(int16_t), dest_pixel, dest_width_step, kernels[3]);
      }
    }

    if(scalar_cols > 0) {
      // do some overlapping work, to clean up the bits at the end
      uint16_t col_index = (n_internal_cols / col_vector_size) * col_vector_size - (col_vector_size - scalar_cols);
      dest_row_index = col_index + kStartBorderSize; // transposed!
      int16_t *dest_pixel = (int16_t *)(dest_data + dest_row_index * dest_width_step) + dest_col_index;
      if(is_u8) {
        const uint8_t *source_pixels = row_origin + col_index;
        convolve_eight_pixels_u8(source_pixels, sizeof(uint8_t), dest_pixel, dest_width_step, kernels[3]);
      } else {
        const int16_t *source_pixels = (int16_t *)row_origin + col_index;
        convolve_four_pixels_s16(source_pixels, sizeof(int16_t), dest_pixel, dest_width_step, kernels[3]);
      }
    }

    uint16_t trailing_edge_col = (uint16_t)(imageSize.width - 8);
    dest_row_index = trailing_edge_col + kEndBorderSize; // transposed!
    int16_t *trailing_edge_dest_pixel = (int16_t *)(dest_data + dest_row_index * dest_width_step) + dest_col_index;
    if(is_u8) {
      const uint8_t *trailing_edge_source_pixels = row_origin + trailing_edge_col;
      edge_convolve_u8(trailing_edge_source_pixels, trailing_edge_dest_pixel, dest_width_step, kernels[4], sizeof(int16_t) * 8);
    } else {
      const int16_t *trailing_edge_source_pixels = (int16_t *)row_origin + trailing_edge_col;
      edge_convolve_s16(trailing_edge_source_pixels, trailing_edge_dest_pixel, dest_width_step, kernels[4], sizeof(int16_t) * 8);
    }
  }

#undef kKernelSize
#undef kStartBorderSize
#undef kEndBorderSize
}

#endif // DMZ_HAS_NEON_COMPILETIME

#pragma mark llcv_sobel7_c

DMZ_INTERNAL void llcv_sobel7_c(IplImage *src, IplImage *dst, bool dx, bool dy) {
  cvSobel(src, dst, !!dx, !!dy, 7); // !! to ensure 0/1-ness of dx, dy
}

#pragma mark llcv_sobel7_neon

DMZ_INTERNAL void llcv_sobel7_neon(IplImage *src, IplImage *dst, IplImage *scratch, bool dx, bool dy) {
#if DMZ_HAS_NEON_COMPILETIME
  int16_t edge_kernel[7] = {-1, -4, -5, 0, 5, 4, 1};
  int16_t smooth_kernel[7] = {1, 6, 15, 20, 15, 6, 1};

  if(dx) {
    vectorized_convolve_transpose7(src, scratch, edge_kernel);
    vectorized_convolve_transpose7(scratch, dst, smooth_kernel);
  } else {
    // this must be dy, since we asserted (dx ^ dy) in llcvSobel7.
    vectorized_convolve_transpose7(src, scratch, smooth_kernel);
    vectorized_convolve_transpose7(scratch, dst, edge_kernel);
  } 
#endif
}

#pragma mark llcv_sobel7

DMZ_INTERNAL void llcv_sobel7(IplImage *src, IplImage *dst, IplImage *scratch, bool dx, bool dy) {
  assert(src != NULL);
  assert(dst != NULL);
  assert(dx ^ dy);
  assert(src->nChannels == 1);
  assert(src->depth == IPL_DEPTH_8U);
  assert(dst->nChannels == 1);
  assert(dst->depth == IPL_DEPTH_16S);

  if(dmz_has_neon_runtime()) {
    bool scratch_provided = scratch != NULL;
    CvSize src_size = cvGetSize(src);
    if(scratch_provided) {
      CvSize scratch_size = cvGetSize(scratch);
#pragma unused(scratch_size) // work around broken compiler warnings
      assert(scratch->nChannels == 1);
      assert(scratch->depth == IPL_DEPTH_16S);
      assert(scratch_size.width == src_size.height);
      assert(scratch_size.height == src_size.width);
    }
    if(!scratch_provided) {
      scratch = cvCreateImage(cvSize(src_size.height, src_size.width), IPL_DEPTH_16S, 1);
    }
    llcv_sobel7_neon(src, dst, scratch, dx, dy);
    if(!scratch_provided) {
      cvReleaseImage(&scratch);
    }
  } else {
    llcv_sobel7_c(src, dst, dx, dy);
  }
}


#define TEST_SOBEL3 0
#define TIME_SOBEL3 0

#if TIME_SOBEL3
static clock_t fastest_neon = CLOCKS_PER_SEC * 1000;
static clock_t fastest_opencv = CLOCKS_PER_SEC * 1000;
#define TIMING_ITERATIONS 100
#endif

#pragma mark llcv_sobel3_dx_dy_opencv

#if TEST_SOBEL3
DMZ_INTERNAL void llcv_sobel3_dx_dy_opencv(IplImage *src, IplImage *dst) {
  cvSobel(src, dst, 1, 1, 3);
}
#endif

#pragma mark llcv_sobel3_dx_dy_c_neon

// For reference, the sobel3_dx_dy kernel is:
//  1,  0, -1
//  0,  0,  0
// -1,  0,  1
DMZ_INTERNAL void llcv_sobel3_dx_dy_c_neon(IplImage *src, IplImage *dst) {
#define kSobel3VectorSize 8

  CvSize src_size = cvGetSize(src);
  assert(src_size.width > kSobel3VectorSize);
  
  uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
  uint16_t src_width_step = (uint16_t)src->widthStep;
  
  uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
  uint16_t dst_width_step = (uint16_t)dst->widthStep;

  uint16_t last_col_index = (uint16_t)(src_size.width - 1);
  bool can_use_neon = dmz_has_neon_runtime();
  
  for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
    uint16_t row1_index = row_index == 0 ? 0 : row_index - 1;
    uint16_t last_row = (uint16_t)(src_size.height - 1);
    uint16_t row2_index = row_index == last_row ? last_row : row_index + 1;

    const uint8_t *src_row1_origin = src_data_origin + row1_index * src_width_step;
    const uint8_t *src_row2_origin = src_data_origin + row2_index * src_width_step;

    int16_t *dst_row_origin = (int16_t *)(dst_data_origin + row_index * dst_width_step);

    uint16_t col_index = 0;
    while(col_index < src_size.width) {
      bool is_first_col = col_index == 0;
      bool is_last_col = col_index == last_col_index;
      bool can_process_next_chunk_as_vector = col_index + kSobel3VectorSize < last_col_index;
      if(is_first_col || is_last_col || !can_use_neon || !can_process_next_chunk_as_vector) {
        // scalar step
        int16_t sum;
        if(dmz_unlikely(is_first_col)) {
          sum = src_row1_origin[col_index] - src_row1_origin[col_index + 1] - src_row2_origin[col_index] + src_row2_origin[col_index + 1];
        } else if(dmz_unlikely(is_last_col)) {
          sum = src_row1_origin[col_index - 1] - src_row1_origin[col_index] - src_row2_origin[col_index - 1] + src_row2_origin[col_index];
        } else {
          sum = src_row1_origin[col_index - 1] - src_row1_origin[col_index + 1] - src_row2_origin[col_index - 1] + src_row2_origin[col_index + 1];
        }
        
        // write result
        dst_row_origin[col_index] = sum;
        
        col_index++;
      } else {
        // vector step
#if DMZ_HAS_NEON_COMPILETIME
        uint8x8_t tl = vld1_u8(src_row1_origin + col_index - 1);
        uint8x8_t tr = vld1_u8(src_row1_origin + col_index + 1);
        uint8x8_t bl = vld1_u8(src_row2_origin + col_index - 1);
        uint8x8_t br = vld1_u8(src_row2_origin + col_index + 1);
        int16x8_t tl_s16 = vreinterpretq_s16_u16(vmovl_u8(tl));
        int16x8_t tr_s16 = vreinterpretq_s16_u16(vmovl_u8(tr));
        int16x8_t bl_s16 = vreinterpretq_s16_u16(vmovl_u8(bl));
        int16x8_t br_s16 = vreinterpretq_s16_u16(vmovl_u8(br));
        int16x8_t sums = vaddq_s16(vsubq_s16(tl_s16, tr_s16), vsubq_s16(br_s16, bl_s16));
        dst_row_origin[col_index + 0] = vgetq_lane_s16(sums, 0);
        dst_row_origin[col_index + 1] = vgetq_lane_s16(sums, 1);
        dst_row_origin[col_index + 2] = vgetq_lane_s16(sums, 2);
        dst_row_origin[col_index + 3] = vgetq_lane_s16(sums, 3);
        dst_row_origin[col_index + 4] = vgetq_lane_s16(sums, 4);
        dst_row_origin[col_index + 5] = vgetq_lane_s16(sums, 5);
        dst_row_origin[col_index + 6] = vgetq_lane_s16(sums, 6);
        dst_row_origin[col_index + 7] = vgetq_lane_s16(sums, 7);
        col_index += kSobel3VectorSize;
#endif
      }
    }
  }
  
#undef kSobel3VectorSize
}

#pragma mark llcv_sobel3_dx_dy

DMZ_INTERNAL void llcv_sobel3_dx_dy(IplImage *src, IplImage *dst) {
  assert(src->nChannels == 1);
  assert(src->depth == IPL_DEPTH_8U);

  assert(dst->nChannels == 1);
  assert(dst->depth == IPL_DEPTH_16S);

  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
#pragma unused(src_size, dst_size) // work around broken compiler warnings

  assert(dst_size.width == src_size.width);
  assert(dst_size.height == src_size.height);
  
#if TIME_SOBEL3
  clock_t start_neon, end_neon;
  start_neon = clock();
  for(int iter = 0; iter < TIMING_ITERATIONS; iter++) {
#endif
    
    llcv_sobel3_dx_dy_c_neon(src, dst);
    
#if TIME_SOBEL3
  }
  end_neon = clock();
  clock_t elapsed_neon = end_neon - start_neon;
  if(elapsed_neon < fastest_neon) {
    fastest_neon = elapsed_neon;
    dmz_debug_log("fastest c/neon %f ms", (1000.0 * (double)fastest_neon / (double)CLOCKS_PER_SEC) / (double)TIMING_ITERATIONS);
  }
#endif
  
  // Start test-only code
  
#if TEST_SOBEL3
  IplImage *opencv_dst = cvCreateImage(dst_size, dst->depth, dst->nChannels);
#if TIME_SOBEL3
  clock_t start_opencv, end_opencv;
  start_opencv = clock();
  for(int iter = 0; iter < TIMING_ITERATIONS; iter++) {
#endif
    llcv_sobel3_dx_dy_opencv(src, opencv_dst);
#if TIME_SOBEL3
  }
  end_opencv = clock();
  clock_t elapsed_opencv = end_opencv - start_opencv;
  if(elapsed_opencv < fastest_opencv) {
    fastest_opencv = elapsed_opencv;
    dmz_debug_log("fastest opencv %f ms", (1000.0 * (double)fastest_opencv / (double)CLOCKS_PER_SEC) / (double)TIMING_ITERATIONS);
  }
#endif
  
  IplImage *delta = cvCreateImage(dst_size, dst->depth, dst->nChannels);
  
  cvSub(dst, opencv_dst, delta);
  
  int n_errors = cvCountNonZero(delta);
  if(n_errors > 0) {
    dmz_debug_log("llcv_sobel3_dx_dy errors: %i", n_errors);
  } else {
    // dmz_debug_log("llcv_sobel3_dx_dy ok");
  }
  
  cvReleaseImage(&delta);
  cvReleaseImage(&opencv_dst);
#endif  // TEST_SOBEL3
}

// For reference, the scharr3_dx kernel is:
//  -3,  0,  +3                                                    |  +3 |
// -10,  0, +10  =  [-1, 0, +1] applied to each pixel, followed by | +10 |
//  -3,  0,  +3                                                    |  +3 |
//
// Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
DMZ_INTERNAL void llcv_scharr3_dx_abs_c_neon(IplImage *src, IplImage *dst) {
#define kScharr3VectorSize 8
  
  CvSize src_size = cvGetSize(src);
  assert(src_size.width > kScharr3VectorSize);
  
  uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
  uint16_t src_width_step = (uint16_t)src->widthStep;
  
  uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
  uint16_t dst_width_step = (uint16_t)dst->widthStep;
#if DMZ_HAS_NEON_COMPILETIME
  uint16_t dst_width_step_in_int16s = (uint16_t)(dst->widthStep / 2);
#endif
  
  uint16_t last_col_index = (uint16_t)(src_size.width - 1);
  bool can_use_neon = dmz_has_neon_runtime();
  int16_t intermediate[src_size.width][src_size.height];  // note: intermediate[col][row]
  
  for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
    const uint8_t *src_row_origin = src_data_origin + row_index * src_width_step;
    uint16_t col_index = 0;
    while(col_index <= last_col_index) {
      uint16_t col_left_index = col_index == 0 ? 0 : col_index - 1;
      uint16_t col_right_index = col_index == last_col_index ? last_col_index : col_index + 1;
      bool can_process_next_chunk_as_vector = col_index + kScharr3VectorSize - 1 <= last_col_index;
      if (!can_use_neon || !can_process_next_chunk_as_vector) {
        // scalar step
        intermediate[col_index][row_index] = (int16_t)abs(src_row_origin[col_right_index] - src_row_origin[col_left_index]);
        col_index++;
      }
      else {
        // vector step
#if DMZ_HAS_NEON_COMPILETIME
        uint8x8_t tl = vld1_u8(src_row_origin + col_left_index);
        uint8x8_t tr = vld1_u8(src_row_origin + col_right_index);
        int16x8_t tl_s16 = vreinterpretq_s16_u16(vmovl_u8(tl));
        int16x8_t tr_s16 = vreinterpretq_s16_u16(vmovl_u8(tr));
        int16x8_t sums = vabdq_s16(tr_s16, tl_s16);
        intermediate[col_index + 0][row_index] = vgetq_lane_s16(sums, 0);
        intermediate[col_index + 1][row_index] = vgetq_lane_s16(sums, 1);
        intermediate[col_index + 2][row_index] = vgetq_lane_s16(sums, 2);
        intermediate[col_index + 3][row_index] = vgetq_lane_s16(sums, 3);
        intermediate[col_index + 4][row_index] = vgetq_lane_s16(sums, 4);
        intermediate[col_index + 5][row_index] = vgetq_lane_s16(sums, 5);
        intermediate[col_index + 6][row_index] = vgetq_lane_s16(sums, 6);
        intermediate[col_index + 7][row_index] = vgetq_lane_s16(sums, 7);
        col_index += kScharr3VectorSize;
#endif
      }
    }
  }

  uint16_t last_row_index = (uint16_t)(src_size.height - 1);

  for(uint16_t col_index = 0; col_index < src_size.width; col_index++) {
    uint16_t row_index = 0;
    while(row_index <= last_row_index) {
      int16_t *dst_row_origin = (int16_t *)(dst_data_origin + row_index * dst_width_step);
      uint16_t row_top_index = row_index == 0 ? 0 : row_index - 1;
      uint16_t row_bot_index = row_index == last_row_index ? last_row_index : row_index + 1;
      bool can_process_next_chunk_as_vector = row_index + kScharr3VectorSize - 1 <= last_row_index;
      if (!can_use_neon || !can_process_next_chunk_as_vector) {
        // scalar step
        dst_row_origin[col_index] = 3 * (intermediate[col_index][row_top_index] + intermediate[col_index][row_bot_index]) + 10 * intermediate[col_index][row_index];
        row_index++;
      }
      else {
        // vector step
#if DMZ_HAS_NEON_COMPILETIME
        int16x8_t qt = vld1q_s16(intermediate[col_index] + row_top_index);
        int16x8_t qm = vld1q_s16(intermediate[col_index] + row_index);
        int16x8_t qb = vld1q_s16(intermediate[col_index] + row_bot_index);
        int16x8_t sums = vaddq_s16(qt, qb);
        sums = vmulq_n_s16(sums, 3);
        sums = vmlaq_n_s16(sums, qm, 10);
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 0);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 1);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 2);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 3);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 4);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 5);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 6);
        dst_row_origin += dst_width_step_in_int16s;
        dst_row_origin[col_index] = vgetq_lane_s16(sums, 7);
        row_index += kScharr3VectorSize;
#endif
      }
    }
  }
  
  #undef kScharr3VectorSize
}

#pragma mark llcv_scharr3_dx_abs

// Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
#if DMZ_DEBUG
void llcv_scharr3_dx_abs(IplImage *src, IplImage *dst) {
#else
DMZ_INTERNAL_UNLESS_CYTHON void llcv_scharr3_dx_abs(IplImage *src, IplImage *dst) {
#endif
  assert(src->nChannels == 1);
  assert(src->depth == IPL_DEPTH_8U);

  assert(dst->nChannels == 1);
  assert(dst->depth == IPL_DEPTH_16S);

  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
#pragma unused(src_size, dst_size) // work around broken compiler warnings

  assert(dst_size.width == src_size.width);
  assert(dst_size.height == src_size.height);

  llcv_scharr3_dx_abs_c_neon(src, dst);
}

#pragma mark llcv_scharr3_dy_abs_c_neon

// For reference, the scharr3_dy kernel is:
// -3, -10,  -3     | -1 |
//  0,   0,   0  =  |  0 | applied to each pixel, followed by [+3, +10, +3]
// +3, +10,  +3     | +1 |
//
// Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
DMZ_INTERNAL void llcv_scharr3_dy_abs_c_neon(IplImage *src, IplImage *dst) {
#define kScharr3VectorSize 8

  CvSize src_size = cvGetSize(src);
  assert(src_size.width > kScharr3VectorSize);

  uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
  uint16_t src_width_step = (uint16_t)src->widthStep;

  uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
  uint16_t dst_width_step = (uint16_t)dst->widthStep;

//  bool can_use_neon = dmz_has_neon_runtime();
  int16_t intermediate[src_size.width][src_size.height];  // note: intermediate[col][row]

  uint16_t last_row_index = (uint16_t)src_size.height - 1;

  for(uint16_t col_index = 0; col_index < src_size.width; col_index++) {
    uint16_t row_index = 0;
    while(row_index <= last_row_index) {
      uint16_t row_top_index = row_index == 0 ? 0 : row_index - 1;
      uint16_t row_bot_index = row_index == last_row_index ? last_row_index : row_index + 1;
      const uint8_t *top_row_origin = src_data_origin + row_top_index * src_width_step;
      const uint8_t *bot_row_origin = src_data_origin + row_bot_index * src_width_step;
//      bool can_process_next_chunk_as_vector = row_index + kScharr3VectorSize - 1 <= last_row_index;
//      if (!can_use_neon || !can_process_next_chunk_as_vector)
      {
        // scalar step
        intermediate[col_index][row_index] = (int16_t)abs(bot_row_origin[col_index] - top_row_origin[col_index]);
        row_index++;
      }
//      else {
//        // vector step
//#if DMZ_HAS_NEON_COMPILETIME
//#endif
//      }
    }
  }

  uint16_t last_col_index = (uint16_t)src_size.width - 1;

  for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
    int16_t *dst_row_origin = (int16_t *)(dst_data_origin + row_index * dst_width_step);
    uint16_t col_index = 0;
    while(col_index <= last_col_index) {
      uint16_t col_left_index = col_index == 0 ? 0 : col_index - 1;
      uint16_t col_right_index = col_index == last_col_index ? last_col_index : col_index + 1;
//      bool can_process_next_chunk_as_vector = col_index + kScharr3VectorSize - 1 <= last_col_index;
//      if (!can_use_neon || !can_process_next_chunk_as_vector)
      {
        // scalar step
        dst_row_origin[col_index] = 3 * (intermediate[col_left_index][row_index] + intermediate[col_right_index][row_index]) + 10 * intermediate[col_index][row_index];
        col_index++;
      }
//      else {
//        // vector step
//#if DMZ_HAS_NEON_COMPILETIME
//#endif
//      }
    }
  }
#undef kScharr3VectorSize
}

#pragma mark llcv_scharr3_dy_abs
// Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
#if DMZ_DEBUG
void llcv_scharr3_dy_abs(IplImage *src, IplImage *dst) {
#else
DMZ_INTERNAL_UNLESS_CYTHON void llcv_scharr3_dy_abs(IplImage *src, IplImage *dst) {
#endif
  assert(src->nChannels == 1);
  assert(src->depth == IPL_DEPTH_8U);
  
  assert(dst->nChannels == 1);
  assert(dst->depth == IPL_DEPTH_16S);
  
  CvSize src_size = cvGetSize(src);
  CvSize dst_size = cvGetSize(dst);
#pragma unused(src_size, dst_size) // work around broken compiler warnings
  
  assert(dst_size.width == src_size.width);
  assert(dst_size.height == src_size.height);
  
  llcv_scharr3_dy_abs_c_neon(src, dst);
}

#endif
