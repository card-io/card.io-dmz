//
//  n_hseg.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "n_hseg.h"
#include "eigen.h"
#include "cv/image_util.h"
#include "cv/morph.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "dmz_constants.h"

static float number_grad_sum_pattern[19] = {
  0.26228655f, 0.30289554f, 0.34632607f, 0.38725636f, 0.42745813f, 0.45875135f,
  0.46498017f, 0.45258447f, 0.43045216f, 0.42430462f, 0.44796554f, 0.47726529f,
  0.48471646f, 0.46457738f, 0.42799847f, 0.38851183f, 0.33966308f, 0.28802608f,
  0.25377602f,
};

typedef struct {
  float min;
  float max;
  float step;
} SliceF32;

typedef struct {
  uint16_t min;
  uint16_t max;
  uint16_t step;
} SliceU16;

#define SliceU16_MAX UINT16_MAX

typedef Eigen::Matrix<float, 1, kCreditCardTargetWidth, Eigen::RowMajor> HorizontalStripPattern;
typedef Eigen::Matrix<float, 1, 19, Eigen::RowMajor> NumberGradSumPattern;

DMZ_INTERNAL NHorizontalSegmentation best_n_hseg_constrained(float *grad_sums, NVerticalSegmentation vseg, NHorizontalSegmentation best, SliceF32 width_slice, SliceU16 offset_slice) {
  
  HorizontalStripPattern pattern;
  Eigen::Map<HorizontalStripPattern> grad_sums_pattern(grad_sums);
  Eigen::Map<NumberGradSumPattern> number_grad_sum_pattern_array(number_grad_sum_pattern);
  uint16_t temp_offsets[16];
  
  for(float width = width_slice.min; width < width_slice.max; width += width_slice.step) {
    float pattern_width = vseg.number_pattern_length * width;
    uint16_t pattern_offset_max = offset_slice.max;
    uint16_t maximum_pattern_offset_max = (uint16_t)(kCreditCardTargetWidth - lrintf(pattern_width));
    if(pattern_offset_max == SliceU16_MAX || pattern_offset_max > maximum_pattern_offset_max) {
      pattern_offset_max = maximum_pattern_offset_max;
    }
    for(uint16_t offset = offset_slice.min; offset < pattern_offset_max; offset += offset_slice.step) {
      pattern.setZero();
      uint8_t offset_index = 0;
      bool in_bounds = true;
      for(uint8_t pattern_index = 0; pattern_index < vseg.number_pattern_length; pattern_index++) {
        if(vseg.number_pattern[pattern_index]) {
          uint16_t center_of_number = (uint16_t)(offset + lrintf(pattern_index * width));
          if(center_of_number + 19 < kCreditCardTargetWidth) { // shouldn't need this check, just being defensive
            pattern.segment<19>(center_of_number) = number_grad_sum_pattern_array;
          } else {
            in_bounds = false;
          }
          temp_offsets[offset_index] = center_of_number;
          offset_index++;
        }
      }
      
      // Not a candidate if some of the numbers fall outside the card
      if(in_bounds) {
        float score = (grad_sums_pattern - pattern).cwiseAbs().sum();
        // lower scores are better -- they're errors/L1 distances
        if(score < best.score) {
          memcpy(&best.offsets, &temp_offsets, sizeof(temp_offsets));
          best.score = score;
          best.number_width = width;
          best.pattern_offset = offset;
        }
      }
    }
  }
  
  return best;
}


DMZ_INTERNAL NHorizontalSegmentation best_n_hseg(IplImage *y_strip, NVerticalSegmentation vseg) {
  // Gradient
  IplImage *grad = cvCreateImage(cvSize(kCreditCardTargetWidth, 27), IPL_DEPTH_8U, 1);
  llcv_morph_grad3_2d_cross_u8(y_strip, grad);
  
  // Reduce (sum), normalize
  IplImage *grad_sum = cvCreateImage(cvSize(kCreditCardTargetWidth, 1), IPL_DEPTH_32F, 1); // could sum to IPL_DEPTH_16U and then convert to 32F for normalization, doing it this way for simplicity, will probably get changed during optimization
  cvReduce(grad, grad_sum, 0 /* reduce to single row */, CV_REDUCE_SUM);
  cvNormalize(grad_sum, grad_sum, 0.0f, 1.0f, CV_MINMAX, NULL);

  cvReleaseImage(&grad);
  
  NHorizontalSegmentation best;
  best.n_offsets = vseg.number_length;
  best.score = 428.0f; // lower is better, this is the max possible (i.e. the worst)
  best.number_width = 0.0f;
  memset(&best.offsets, 0, 16 * sizeof(uint16_t));
  
  float *grad_sum_data = (float *)llcv_get_data_origin(grad_sum);
  SliceF32 width_slice;
  SliceU16 offset_slice;
  
  width_slice.min = 17.1f;
  width_slice.max = 19.7f;
  width_slice.step = 0.5f;
  offset_slice.min = 0;
  offset_slice.max = SliceU16_MAX;
  offset_slice.step = 10;
  best = best_n_hseg_constrained(grad_sum_data, vseg, best, width_slice, offset_slice);

  // In the following lines, there's some bounds checking on offset_slice.min.
  // It is needed because it prevents underflow due to using uints. (The uint/int issue
  // also explains the ?: method instead of subtracting and taking max vs 0.)
  // There's no bounds checking needed on width_slice.min/max/step, because they can't
  // get outside of a reasonable range in the steps below.
  // The bounds checking on offset_slice.max is done in best_n_hseg_constrained, because
  // it can't overflow, and because we don't know enough here to do it conveniently and DRYly
  width_slice.min = best.number_width - 0.5f;
  width_slice.max = best.number_width + 0.5f;
  width_slice.step = 0.2f;
  offset_slice.min = best.pattern_offset < 10 ? 0 : best.pattern_offset - 10;
  offset_slice.max = best.pattern_offset + 10;
  offset_slice.step = 1;
  best = best_n_hseg_constrained(grad_sum_data, vseg, best, width_slice, offset_slice);
  
  width_slice.min = best.number_width - 0.2f;
  width_slice.max = best.number_width + 0.2f;
  width_slice.step = 0.1f;
  offset_slice.min = best.pattern_offset < 3 ? 0 : best.pattern_offset - 3;
  offset_slice.max = best.pattern_offset + 3;
  offset_slice.step = 1;
  best = best_n_hseg_constrained(grad_sum_data, vseg, best, width_slice, offset_slice);
  
  width_slice.min = best.number_width - 0.1f;
  width_slice.max = best.number_width + 0.1f;
  width_slice.step = 0.05f;
  offset_slice.min = best.pattern_offset < 3 ? 0 : best.pattern_offset - 3;
  offset_slice.max = best.pattern_offset + 3;
  offset_slice.step = 1;
  best = best_n_hseg_constrained(grad_sum_data, vseg, best, width_slice, offset_slice);

  cvReleaseImage(&grad_sum);

  return best;
}



#endif // COMPILE_DMZ
