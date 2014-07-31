//
//  scan.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#define SCAN_FOREVER 0  // useful for performance profiling

#include "scan.h"

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
}

void scanner_add_frame(ScannerState *state, IplImage *y, FrameScanResult *result) {
    
  // Don't bother with a bunch of assertions about y here,
  // since the frame reader will make them anyway.
  scan_card_image(y, result);
  if (result->upside_down) {
    return;
  }
  scan_analytics_record_frame(&state->session_analytics, result);
  if (!result->usable) {
    return;
  }
  
  // TODO: Scene change detection?
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

void scanner_result(ScannerState *state, ScannerResult *result) {
  result->complete = false; // until we change our minds otherwise...avoids having to set this at all the possible early exits

#if SCAN_FOREVER
  return;
#endif

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

  // Don't return a number that fails basic prefix sanity checks
  CardType card_type = dmz_card_info_for_prefix_and_length(number_as_u8s, result->n_numbers, false).card_type;
  if(card_type != CardTypeAmbiguous &&
     card_type != CardTypeUnrecognized &&
     dmz_passes_luhn_checksum(number_as_u8s, result->n_numbers)) {
    result->complete = true;
  }
}

void scanner_destroy(ScannerState *state) {
  // currently a no-op
}


#endif // COMPILE_DMZ
