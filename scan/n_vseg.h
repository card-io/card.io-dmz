//
//  n_vseg.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_N_VSEG_H
#define DMZ_SCAN_N_VSEG_H

#include "opencv2/core/core_c.h" // needed for IplImage
#include "dmz_macros.h"

typedef uint8_t NumberPatternType;

typedef struct {
  float score;
  uint16_t y_offset;
  NumberPatternType pattern_type; // internal use only :(
  uint8_t number_pattern[19];
  uint8_t number_pattern_length;
  uint8_t number_length;
} NVerticalSegmentation;

// Calculate the best number vertical segmentation for the card image y.
// y must be 428x270, single channel, uint8_t, with no ROI set.
DMZ_INTERNAL NVerticalSegmentation best_n_vseg(IplImage *y);


#endif
