//
//  scan_analytics.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_SESSION_H
#define DMZ_SCAN_SESSION_H

#include "dmz.h"
#include "frame.h"
#include <sys/time.h>
#include <map>

#define kScanSessionNumFramesStored 20

// Define ranges of digits from which we can select one to save
#define kNumFirstDigitChoices 6
#define kNumLastDigitChoices 4
#define kNumTotalDigitChoices (kNumLastDigitChoices + kNumFirstDigitChoices)

// Info about a single frame
typedef struct {
  uint32_t frame_index; // Index of this frame out of all scanned frames in this session
  std::map<std::string, std::string> frame_values; // E.g., frame_values["foo_score"] == "43.1"
} ScanFrameAnalytics;

// Info about a scan session
typedef struct {
  uint32_t num_frames_scanned;
  uint8_t frames_ring_start;
  ScanFrameAnalytics frames_ring[kScanSessionNumFramesStored];
} ScanSessionAnalytics;

// Reset a ScanSessionAnalytics, whether new or re-used
void scan_analytics_init(ScanSessionAnalytics *session);

ScanFrameAnalytics *scan_analytics_record_frame(ScanSessionAnalytics *session, FrameScanResult *frame);

#endif
