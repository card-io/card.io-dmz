//
//  n_hseg.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_N_HSEG_H
#define DMZ_SCAN_N_HSEG_H

#include "n_vseg.h"
#include "opencv2/core/core_c.h" // needed for IplImage
#include "dmz_macros.h"

typedef struct {
  uint8_t n_offsets;
  uint16_t offsets[16];
  float score;
  float number_width;
  uint16_t pattern_offset;
} NHorizontalSegmentation;

DMZ_INTERNAL NHorizontalSegmentation best_n_hseg(IplImage *y_strip, NVerticalSegmentation vseg);


#endif
