//
//  frame.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "frame.h"
#include "dmz_constants.h"
#include "dmz_debug.h"

// These cutoff values derived through a very round of experimentation at my desk,
// in one set of lighting conditions, with a handful of cards.
// I've erred on the side of laxity (letting too many frames through), for now
// TODO: Experiment more, and put some rigor around these numbers!
// TODO: Try harder to find good hseg criteria, consider y_offset criteria (what range is reasonable?, consider non-sum-based score criteria
// TODO: Upside down card detection? (Use y_offset as a heuristic?)

#define kMinVSegScore 15  // non-lax value: 18?
#define kMaxNumberScoreDelta 3 // non-lax value: 1? 2?
#define kFlipVSegYOffsetCutoff ((kCreditCardTargetHeight - kNumberHeight) / 2)

DMZ_INTERNAL void scan_card_image(IplImage *y, bool collect_card_number, bool scan_expiry, FrameScanResult *result) {
  assert(NULL == y->roi);
  assert(y->width == 428);
  assert(y->height == 270);
  assert(y->depth == IPL_DEPTH_8U);
  assert(y->nChannels == 1);

  result->upside_down = false;
  result->usable = false;
  
  result->vseg = best_n_vseg(y); // TODO - report this

  // If the best vseg is in the top half of the card,
  // return early and indicate that the card is upside-down.
  if (result->vseg.y_offset < kFlipVSegYOffsetCutoff) {
    result->upside_down = true;
    return;
  }

  result->usable = result->vseg.score > kMinVSegScore;
  if(!result->usable) {
    dmz_debug_log("vseg.score %f unusable", result->vseg.score);
    return;
  } else {
    dmz_debug_log("vseg.score: %f", result->vseg.score);
    result->scan_progress = SCAN_PROGRESS_VSEG;
  }

  if (collect_card_number) {
    cvSetImageROI(y, cvRect(0, result->vseg.y_offset, kCreditCardTargetWidth, kNumberHeight));
    
    result->hseg = best_n_hseg(y, result->vseg);
    // I've not found the hseg score to be a reliable indicator of quality at all
    // Unsurprising, since this is the hardest phase of the pipeline, and we're struggling
    // just to find anything at all!
    //
    //  if(!result->usable) {
    //    cvResetImageROI(y);
    //    return result;
    //  }
    
    result->scores = number_scores(y, result->hseg);
    float number_score = result->hseg.n_offsets - result->scores.sum();
    result->usable = number_score < kMaxNumberScoreDelta;
    if (!result->usable) {
      dmz_debug_log("number_score %f unusable", number_score);
      return;
    } else {
      dmz_debug_log("number_score %f", number_score);
      result->scan_progress = SCAN_PROGRESS_HSEG;
    }
    cvResetImageROI(y);
  }

#if SCAN_EXPIRY
  if (scan_expiry && result->vseg.y_offset < kCreditCardTargetHeight - 2 * kSmallCharacterHeight) {
    best_expiry_seg(y, result->vseg.y_offset, result->expiry_groups, result->name_groups);
  #if DMZ_DEBUG
    if (result->expiry_groups.empty()) {
      dmz_debug_log("Expiry segmentation failed.");
    }
  #endif
  }
#endif
}

#if CYTHON_DMZ
void cython_scan_card_image(IplImage *y, CythonFrameScanResult *result) {
  FrameScanResult frameScanResult;
  frameScanResult.focus_score = 666;
  frameScanResult.brightness_score = 150;
  frameScanResult.iso_speed = 400;
  frameScanResult.shutter_speed = 5;
  frameScanResult.torch_is_on = 0;
  frameScanResult.flipped = 0;

  scan_card_image(y, true, true, &frameScanResult);
  
  result->usable = frameScanResult.usable;
  result->hseg = frameScanResult.hseg;
  result->vseg = frameScanResult.vseg;
}
#endif  // CYTHON_DMZ

#endif // COMPILE_DMZ
