//
//  frame.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_FRAME_H
#define DMZ_SCAN_FRAME_H

#include "expiry_seg.h"
#include "n_categorize.h"
#include "opencv2/core/core_c.h" // needed for IplImage
#include "dmz_macros.h"

typedef int SCAN_PROGRESS;
enum {
  SCAN_PROGRESS_NONE = 0,
  SCAN_PROGRESS_FOCUS,
  SCAN_PROGRESS_EDGES,
  SCAN_PROGRESS_VSEG,
  SCAN_PROGRESS_NSEG,
  SCAN_PROGRESS_SCORE,
  SCAN_PROGRESS_STABILITY,
  SCAN_PROGRESS_EXPIRY
};

typedef struct {
  float                   focus_score;
  NumberScores            scores;
  NHorizontalSegmentation hseg;
  NVerticalSegmentation   vseg;
  GroupedRectsList        expiry_groups;
  GroupedRectsList        name_groups;
  bool                    usable;
  bool                    upside_down; // whether the frame was found to be upside-down
  bool                    flipped; // whether the frame has been pre-flipped
  float                   brightness_score;
  uint16_t                iso_speed;
  float                   shutter_speed;
  bool                    torch_is_on;
  int                     scan_progress;
} FrameScanResult;


// Scans a single card image, returns a summary of all info gathered along the way.
// If usable is false, disregard all other info.
// y must be 428x270, uint8_t, no roi, single channel greyscale.
DMZ_INTERNAL void scan_card_image(IplImage *y, bool collect_card_number, bool scan_expiry, FrameScanResult *result);

#if CYTHON_DMZ
typedef struct {
  NHorizontalSegmentation hseg;
  NVerticalSegmentation   vseg;
  bool                    usable;
  
} CythonFrameScanResult;

void cython_scan_card_image(IplImage *y, CythonFrameScanResult *result);
#endif  // CYTHON_DMZ

#endif
