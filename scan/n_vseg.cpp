//
//  n_vseg.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "n_vseg.h"
#include "eigen.h"
#include "dmz.h"

#include "cv/morph.h"
#include "cv/convert.h"

#include "models/generated/modelm_befe75da.hpp"
// TODO: gpu for matrix mult?


enum {
  NumberPatternUnknown = 0,
  NumberPatternVisalike = 1,
  NumberPatternAmexlike = 2,
};

static uint8_t const NumberLengthForNumberPatternType[3] = {0, 16, 15};
static uint8_t const NumberPatternLengthForPatternType[3] = {0, 19, 17};
static uint8_t const NumberPatternUnknownPattern[19]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint8_t const NumberPatternVisalikePattern[19] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1};
static uint8_t const NumberPatternAmexlikePattern[19] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0};
static uint8_t const * NumberPatternForPatternType[3] = {NumberPatternUnknownPattern, NumberPatternVisalikePattern, NumberPatternAmexlikePattern};


typedef Eigen::Matrix<float, 1, 204, Eigen::RowMajor> VSegModelInput;
typedef Eigen::Matrix<float, 1, 3, Eigen::RowMajor> VSegProbabilities;

#define kVertSegSumWindowSize 27

DMZ_INTERNAL inline VSegProbabilities vseg_probabilities_for_hstrip(IplImage *y, IplImage *cropped_gradient, IplImage *downsampled_normed, IplImage *as_float) {
  llcv_morph_grad3_1d_u8(y, cropped_gradient);
  llcv_lineardown2_1d_u8(cropped_gradient, downsampled_normed);
  llcv_norm_convert_1d_u8_to_f32(downsampled_normed, as_float);
  
  Eigen::Map<VSegModelInput> vseg_model_input((float *)as_float->imageData);
  VSegProbabilities probabilities = applym_befe75da(vseg_model_input);
  return probabilities;
}

DMZ_INTERNAL inline void best_segmentation_for_vseg_scores(float *visalike_scores, float *amexlike_scores, NVerticalSegmentation *best) {
  float visalike_sum = 0.0f;
  float amexlike_sum = 0.0f;
  // Why a ring buffer? As an efficient means of doing a box window convolution.
  // I tried re-summing the ring buffer each time instead of doing a running sum, to
  // see whether there were issues with accumulated errors. There were not, so I'm
  // leaving this bit of premature optimization in. :)
  float visalike_ring_buffer[kVertSegSumWindowSize];
  float amexlike_ring_buffer[kVertSegSumWindowSize];
  
  best->score = 0.0f;
  best->pattern_type = NumberPatternUnknown;
  best->y_offset = 0;
  
  for(uint16_t y_offset = 0; y_offset < kCreditCardTargetHeight; y_offset++) {
    float visalike_score = visalike_scores[y_offset];
    float amexlike_score = amexlike_scores[y_offset];

    visalike_sum += visalike_score;
    amexlike_sum += amexlike_score;
    
    uint8_t buffer_index = y_offset % kVertSegSumWindowSize;
    visalike_ring_buffer[buffer_index] = visalike_score;
    amexlike_ring_buffer[buffer_index] = amexlike_score;
    
    if(y_offset >= kVertSegSumWindowSize - 1) {
      // ring buffer is full, start using the scores
      if(visalike_sum > best->score) {
        best->score = visalike_sum;
        best->pattern_type = NumberPatternVisalike;
        best->y_offset = y_offset - kVertSegSumWindowSize + 1;
      }
      if(amexlike_sum > best->score) {
        best->score = amexlike_sum;
        best->pattern_type = NumberPatternAmexlike;
        best->y_offset = y_offset - kVertSegSumWindowSize + 1;
      }
      
      uint8_t next_buffer_index = (y_offset + 1) % kVertSegSumWindowSize;
      visalike_sum -= visalike_ring_buffer[next_buffer_index];
      amexlike_sum -= amexlike_ring_buffer[next_buffer_index];
    }
  }
}

DMZ_INTERNAL NVerticalSegmentation best_n_vseg(IplImage *y) {
  assert(y->roi == NULL);
  CvSize y_size = cvGetSize(y);
#pragma unused(y_size) // work around broken compiler warnings
  assert(y_size.width == kCreditCardTargetWidth);
  assert(y_size.height == kCreditCardTargetHeight);
  assert(y->depth == IPL_DEPTH_8U);
  assert(y->nChannels == 1);

  // Set up reusable memory for image calculations
  IplImage *cropped_gradient = cvCreateImage(cvSize(408, 1), IPL_DEPTH_8U, 1);
  IplImage *downsampled_normed = cvCreateImage(cvSize(204, 1), IPL_DEPTH_8U, 1);
  IplImage *as_float = cvCreateImage(cvSize(204, 1), IPL_DEPTH_32F, 1);

  // Score buffers, to be filled in as needed
  float visalike_scores[kCreditCardTargetHeight];
  memset(visalike_scores, 0, sizeof(visalike_scores));

  float amexlike_scores[kCreditCardTargetHeight];
  memset(amexlike_scores, 0, sizeof(amexlike_scores));

  uint16_t min_y_offset = 0;
  uint16_t max_y_offset = kCreditCardTargetHeight;
  uint8_t y_offset_step = 4;

  // Initially, calculate every fourth score, to narrow down the area in which we have to work
  for(uint16_t y_offset = min_y_offset; y_offset < max_y_offset; y_offset += y_offset_step) {
    cvSetImageROI(y, cvRect(10, y_offset, 408, 1));
    VSegProbabilities probabilities = vseg_probabilities_for_hstrip(y, cropped_gradient, downsampled_normed, as_float);
    visalike_scores[y_offset] = probabilities(0, 1);
    amexlike_scores[y_offset] = probabilities(0, 2);
  }

  NVerticalSegmentation best;
  best_segmentation_for_vseg_scores(visalike_scores, amexlike_scores, &best);

  // Now that we know roughly where we're interested in, fill in a few more scores
  // (the ones that could make a difference), and recalculate

#define kFineTuningBuffer 8
  // The scores that matter are (roughly) y_offset - kFineTuningBuffer : y_offset + kVertSegSumWindowSize + kFineTuningBuffer
  // In theory, due to windowed summing, the scores in the middle also don't matter (since they would count equally
  // for all possible y_offsets), but go ahead and calculate them anyway, since the provide useful signal about
  // whether it is actually a credit card present or not

  // All values must be bounds checked against 270 and (when needed) safely against 0 (using uints!)
  min_y_offset = MIN(kCreditCardTargetHeight, best.y_offset < kFineTuningBuffer ? 0 : best.y_offset - kFineTuningBuffer);
  max_y_offset = MIN(kCreditCardTargetHeight, best.y_offset + kVertSegSumWindowSize + kFineTuningBuffer);
  y_offset_step = 1;

  for(uint16_t y_offset = min_y_offset; y_offset < max_y_offset; y_offset += y_offset_step) {
    // Don't recalculate anything -- we already calculated 1/4th of them!
    if(visalike_scores[y_offset] == 0 && amexlike_scores[y_offset] == 0) {
      cvSetImageROI(y, cvRect(10, y_offset, 408, 1));
      VSegProbabilities probabilities = vseg_probabilities_for_hstrip(y, cropped_gradient, downsampled_normed, as_float);
      visalike_scores[y_offset] = probabilities(0, 1);
      amexlike_scores[y_offset] = probabilities(0, 2);
    }
  }

  // TODO: Hint that resumming across all the possible values isn't really necessary...
  best_segmentation_for_vseg_scores(visalike_scores, amexlike_scores, &best);

  cvReleaseImage(&cropped_gradient);
  cvReleaseImage(&downsampled_normed);
  cvReleaseImage(&as_float);

  cvResetImageROI(y);

  best.number_pattern_length = NumberPatternLengthForPatternType[best.pattern_type];
  memcpy(&best.number_pattern, NumberPatternForPatternType[best.pattern_type], sizeof(best.number_pattern));
  best.number_length = NumberLengthForNumberPatternType[best.pattern_type];
  
  return best;
}


#endif // COMPILE_DMZ
