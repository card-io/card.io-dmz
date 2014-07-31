//
//  frame.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_FRAME_H
#define DMZ_SCAN_FRAME_H

#include "n_categorize.h"
#include "opencv2/core/core_c.h" // needed for IplImage
#include "dmz_macros.h"

typedef struct {
  float                   focus_score;
  NumberScores            scores;
  NHorizontalSegmentation hseg;
  NVerticalSegmentation   vseg;
  bool                    usable;
  bool                    upside_down; // whether the frame was found to be upside-down
  bool                    flipped; // whether the frame has been pre-flipped
  float                   brightness_score;
  uint16_t                iso_speed;
  float                   shutter_speed;
  bool                    torch_is_on;
} FrameScanResult;


// Scans a single card image, returns a summary of all info gathered along the way.
// If usable is false, disregard all other info.
// y must be 428x270, uint8_t, no roi, single channel greyscale.
DMZ_INTERNAL void scan_card_image(IplImage *y, FrameScanResult *result);

#if CYTHON_DMZ
typedef struct {
  NHorizontalSegmentation hseg;
  NVerticalSegmentation   vseg;
  bool                    usable;
  
} CythonFrameScanResult;

void cython_scan_card_image(IplImage *y, CythonFrameScanResult *result);
#endif  // CYTHON_DMZ

#endif
