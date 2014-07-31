//
//  scan_analytics.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "scan_analytics.h"
#include "mz.h"
#include "dmz_debug.h"
#include <time.h>


// --- SCAN FRAME ANALYTICS ----------------------------------------

// Record relevant fields from FrameScanResult struct into
// ScanFrameAnalytics struct.
void scan_frame_analytics_record(ScanFrameAnalytics *f, FrameScanResult *frame) {
  // Add lines of this form, to capture any relevant fields from `frame`:
  // f->frame_values["foo"] = std::to_string(frame->foo);
}


// --- SCAN SESSION --------------------------------------------------

// Reset the ScanSessionAnalytics struct
// Also starts the timer for the session
void scan_analytics_init(ScanSessionAnalytics *session) {
  session->num_frames_scanned = 0;
  session->frames_ring_start = 0;
}

// Given a FrameScanResult frame, records to the session as a ScanFrameAnalytics
// If the provided frame is NULL, returns NULL.
// Returns pointer to the recorded ScanFrameAnalytics struct.
ScanFrameAnalytics *scan_analytics_record_frame(ScanSessionAnalytics *session, FrameScanResult *frame) {

  // Get pointer to current frame analytics struct
  int index = session->num_frames_scanned % kScanSessionNumFramesStored;
  ScanFrameAnalytics *f = &(session->frames_ring[index]);

  // Once we start to overflow, move the frames_ring_start up one.
  if (session->num_frames_scanned > kScanSessionNumFramesStored) {
    session->frames_ring_start = (session->num_frames_scanned + 1) % kScanSessionNumFramesStored;
  }
  
  // Copy frame info into ScanFrameAnalytics struct and record its absolute session index
  scan_frame_analytics_record(f, frame);
  f->frame_index = session->num_frames_scanned;

  // Increment total number of frames scanned
  session->num_frames_scanned += 1;
  
  // Return pointer to this ScanFrameAnalytics struct
  return f;
}
