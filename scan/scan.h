//
//  scan.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_SCAN_H
#define DMZ_SCAN_SCAN_H

#include "frame.h"
#include "dmz_macros.h"
#include "scan_analytics.h"
#include "expiry_seg.h"
#include <sys/time.h>

// TODO: Somewhere expose some data+analytics to send to a server for future model training...

typedef Eigen::Matrix<NumberScores::Index, 16, 1, Eigen::ColMajor> NumberPredictions; // one prediction per number (up to 16 of them)

typedef struct {
  bool complete; // if complete is false, the rest of the stuff in this struct must be ignored
  NumberPredictions predictions;
  uint8_t n_numbers;
  int expiry_month;
  int expiry_year;
#if DMZ_DEBUG
  GroupedRectsList expiry_groups;
  GroupedRectsList name_groups;
#endif
} ScannerResult;

typedef struct {
  uint16_t count15;
  uint16_t count16;
  NumberScores aggregated15;
  NumberScores aggregated16;
  ScanSessionAnalytics session_analytics;
  ScannerResult successfulCardNumberResult;
  NHorizontalSegmentation mostRecentUsableCardNumberHSeg;
  unsigned long timeOfCardNumberCompletionInMilliseconds;
  bool scan_expiry;
  int expiry_month;
  int expiry_year;
  GroupedRectsList expiry_groups;
  GroupedRectsList name_groups;
} ScannerState;

// Initialize a scanner.
void scanner_initialize(ScannerState *state);

// Reset a scanner. Called by initialize.
void scanner_reset(ScannerState *state);

// Provide the scanner with a single card image.
//
// Notes:
// - y must be 428x270, uint8_t, no roi, single channel greyscale.
// - the FrameScanResult struct should be pre-populated with
//   values for 'flipped' and 'focusScore'
// - if the card appears to be upside down, result->upside_down
//   will be set to true (and result->usable to false)
void scanner_add_frame(ScannerState *state, IplImage *y, FrameScanResult *result); // pre-expiry backward-compatible version
void scanner_add_frame_with_expiry(ScannerState *state, IplImage *y, bool scan_expiry, FrameScanResult *result);

// Ask the scanner for its number predictions.
// If result.complete is false, the rest of the result must be ignored.
void scanner_result(ScannerState *state, ScannerResult *result);

// Destroy a scanner and any associated resources.
void scanner_destroy(ScannerState *state);

#endif
