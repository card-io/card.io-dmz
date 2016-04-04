//
//  scan.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "scan.h"
#include "expiry_categorize.h"
#include "expiry_seg.h"

#define SCAN_FOREVER 0  // useful for performance profiling
#define EXTRA_TIME_FOR_EXPIRY_IN_MICROSECONDS 1000 // once the card number has been successfully identified, allow a bit more time to figure out the expiry

#define kDecayFactor 0.8f
#define kMinStability 0.7f

void scanner_initialize(ScannerState *state) {
  scanner_reset(state);
}

void scanner_reset(ScannerState *state) {
  state->count15 = 0;
  state->count16 = 0;
  state->aggregated15 = NumberScores::Zero();
  state->aggregated16 = NumberScores::Zero();
  scan_analytics_init(&state->session_analytics);
  state->timeOfCardNumberCompletionInMilliseconds = 0;
  state->scan_expiry = false;
  state->expiry_month = 0;
  state->expiry_year = 0;
  state->expiry_groups.clear();
  state->name_groups.clear();
}

void scanner_add_frame(ScannerState *state, IplImage *y, FrameScanResult *result) {
  scanner_add_frame_with_expiry(state, y, false, result);
}

void scanner_add_frame_with_expiry(ScannerState *state, IplImage *y, bool scan_expiry, FrameScanResult *result) {

  bool still_need_to_collect_card_number = (state->timeOfCardNumberCompletionInMilliseconds == 0);
  bool still_need_to_scan_expiry = scan_expiry && (state->expiry_month == 0 || state->expiry_year == 0);

  // Don't bother with a bunch of assertions about y here,
  // since the frame reader will make them anyway.
  scan_card_image(y, still_need_to_collect_card_number, still_need_to_scan_expiry, result);
  if (result->upside_down) {
    return;
  }
 
  scan_analytics_record_frame(&state->session_analytics, result);

  // TODO: Scene change detection?
  
  if (!result->usable) {
    return;
  }

#if SCAN_EXPIRY
  if (still_need_to_scan_expiry) {
    state->scan_expiry = true;
    expiry_extract(y, state->expiry_groups, result->expiry_groups, &state->expiry_month, &state->expiry_year);
    state->name_groups = result->name_groups;  // for now, for the debugging display
  }
#endif
  
  if (still_need_to_collect_card_number) {
    
    state->mostRecentUsableHSeg = result->hseg;
    state->mostRecentUsableVSeg = result->vseg;
    
    if(result->hseg.n_offsets == 15) {
      state->aggregated15 *= kDecayFactor;
      state->aggregated15 += result->scores * (1 - kDecayFactor);
      state->count15++;
    } else if(result->hseg.n_offsets == 16) {
      state->aggregated16 *= kDecayFactor;
      state->aggregated16 += result->scores * (1 - kDecayFactor);
      state->count16++;
    } else {
      assert(false);
    }
  }
}

void scanner_result(ScannerState *state, ScannerResult *result, FrameScanResult *frameResult) {
  result->complete = false; // until we change our minds otherwise...avoids having to set this at all the possible early exits

#if SCAN_FOREVER
  return;
#endif

  if (state->timeOfCardNumberCompletionInMilliseconds > 0) {
    *result = state->successfulCardNumberResult;
  }
  else {
    uint16_t max_count = MAX(state->count15, state->count16);
    uint16_t min_count = MIN(state->count15, state->count16);

    // We want a three frame lead at a bare minimum.
    // Also guarantees we have at least three frames, period. :)
    if(max_count - min_count < 3) {
      return;
    }

    // Want a significant opinion about whether visa or amex
    if(min_count * 2 > max_count) {
      return;
    }
    
    result->hseg = state->mostRecentUsableHSeg;
    result->vseg = state->mostRecentUsableVSeg;

    // TODO: Sanity check the scores distributions
    // TODO: Do something else sophisticated here -- look at confidences, distributions, stability, hysteresis, etc.
    NumberScores aggregated;
    if(state->count15 > state->count16) {
      result->n_numbers = 15;
      aggregated = state->aggregated15;
    } else {
      result->n_numbers = 16;
      aggregated = state->aggregated16;
    }

    // Calculate result predictions
    // At the same time, put it in a convenient format for the basic consistency checks
    uint8_t number_as_u8s[16];

    dmz_debug_print("Stability: ");
    for(uint8_t i = 0; i < result->n_numbers; i++) {
      NumberScores::Index r, c;
      float max_score = aggregated.row(i).maxCoeff(&r, &c);
      float sum = aggregated.row(i).sum();
      result->predictions(i, 0) = c;
      number_as_u8s[i] = (uint8_t)c;
      float stability = max_score / sum;
      dmz_debug_print("%d ", (int) ceilf(stability * 100));

      // Bail early if low stability
      if (stability < kMinStability) {
        dmz_debug_print("\n");
        return;
      }
    }
    dmz_debug_print("\n");
    frameResult->scan_progress = SCAN_PROGRESS_STABILITY;

    // Don't return a number that fails basic prefix sanity checks
    CardType card_type = dmz_card_info_for_prefix_and_length(number_as_u8s, result->n_numbers, false).card_type;
    if(card_type != CardTypeAmbiguous && card_type != CardTypeUnrecognized) {
      frameResult->scan_progress = SCAN_PROGRESS_CARDTYPE;
      
      if (dmz_passes_luhn_checksum(number_as_u8s, result->n_numbers)) {
        frameResult->scan_progress = SCAN_PROGRESS_LUHN;
        dmz_debug_print("CARD NUMBER SCANNED SUCCESSFULLY.\n");
        struct timeval time;
        gettimeofday(&time, NULL);
        state->timeOfCardNumberCompletionInMilliseconds = (long)((time.tv_sec * 1000) + (time.tv_usec / 1000));
        state->successfulCardNumberResult = *result;
      }
    }
  }

  // Once the card number has been successfully scanned, then wait a bit longer for successful expiry scan (if collecting expiry)
  if (state->timeOfCardNumberCompletionInMilliseconds > 0) {
#if SCAN_EXPIRY
    if (state->scan_expiry) {
#else
    if (false) {
#endif
      struct timeval time;
      gettimeofday(&time, NULL);
      long now = (long)((time.tv_sec * 1000) + (time.tv_usec / 1000));

      if ((state->expiry_month > 0 && state->expiry_year > 0) ||
          now - state->timeOfCardNumberCompletionInMilliseconds > EXTRA_TIME_FOR_EXPIRY_IN_MICROSECONDS) {

        result->expiry_month = state->expiry_month;
        result->expiry_year = state->expiry_year;
#if DMZ_DEBUG
        result->expiry_groups = state->expiry_groups;
        result->name_groups = state->name_groups;
#endif
        result->complete = true;

        dmz_debug_print("Extra time for expiry scan: %6.3f seconds\n", ((float)(now - state->timeOfCardNumberCompletionInMilliseconds)) / 1000.0f);
      }
    }
    else {
      result->expiry_month = 0;
      result->expiry_year = 0;
      result->complete = true;
    }
  }
}

void scanner_destroy(ScannerState *state) {
  // currently a no-op
}


#endif // COMPILE_DMZ
