//
//  stripes.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "stripes.h"
#include "dmz_debug.h"
#include "opencv2/imgproc/imgproc_c.h"

//#define DEBUG_STRIPES_PERFORMANCE 1

// Long integers are sufficiently large to hold all of our sums:
//
// Maximum pixel value possible with abs(Scharr) operator = 3 * 255 + 10 * 255 + 3 * 255 = 4080, so use a 16S destination.
// Maximum line-sum possible = 4080 * kCreditCardTargetWidth = 1,746,240 = 0x1AA540, so line-sum will always fit in a long.
// Maximum stripe-sum possible = (max line-sum possible) * kNumberHeight = 47,148,480 = 0x2CF6DC0, so stripe-sum will always fit in a long.

struct StripeSumCompareDescending
: public std::binary_function<StripeSum, StripeSum, bool> {
  inline bool operator()(StripeSum const &stripe_sum_1, StripeSum const &stripe_sum_2) const {
    if (stripe_sum_1.height != stripe_sum_2.height) {
      return (stripe_sum_1.height > stripe_sum_2.height);
    }
    return (stripe_sum_1.sum > stripe_sum_2.sum);
  }
};

std::vector<StripeSum> sorted_stripes(IplImage *sobel_image, uint16_t starting_y_offset, int minCharacterHeight, int maxCharacterHeight) {
#if DEBUG_STRIPES_PERFORMANCE
  dmz_debug_timer_start();
#endif
  
  CvSize card_image_size = cvGetSize(sobel_image);
  CvRect relevant_rect = cvRect(0, starting_y_offset, card_image_size.width, card_image_size.height - starting_y_offset);
  
  // Calculate relative vertical-line-segment-ness for each scan line (i.e., cvSum of the [x-axis] Sobel image for that line):
  
  int   first_stripe_base_row = relevant_rect.y + 1;  // the "+ 1" represents the tolerance above and below each stripe
  int   last_stripe_base_row = card_image_size.height - (minCharacterHeight + 1);  // the "+ 1" represents the tolerance above and below each stripe
  long  line_sum[card_image_size.height];
  
  int   left_edge = kSmallCharacterWidth * 3;  // there aren't usually any actual characters this far to the left
  int   right_edge = (card_image_size.width * 2) / 3;  // beyond here lie logos
  
  for (int row = first_stripe_base_row - 1; row < card_image_size.height; row++) {
    cvSetImageROI(sobel_image, cvRect(left_edge, row, right_edge - left_edge, 1));
    line_sum[row] = (long)cvSum(sobel_image).val[0];
  }
  
  cvResetImageROI(sobel_image);
  
#if DEBUG_STRIPES_PERFORMANCE
  dmz_debug_timer_print("line sums");
#endif
  
  int row;
  std::vector<StripeSum> stripe_sums;
  for (int base_row = first_stripe_base_row; base_row < last_stripe_base_row; base_row++) {
    long sum = 0;
    for (int row = base_row; row < base_row + minCharacterHeight; row++) {
      sum += line_sum[row];
    }
    
    // Calculate threshold = half the value of the maximum line-sum in the stripe:
    long threshold = 0;
    for (row = base_row; row < base_row + minCharacterHeight; row++) {
      if (line_sum[row] > threshold) {
        threshold = line_sum[row];
      }
    }
    threshold /= 2;
    
    // Eliminate stripes that have a a much dimmer-than-average sub-stripe at their very top or very bottom:
    if (line_sum[base_row] + line_sum[base_row + 1] < threshold) {
      continue;
    }
    if (line_sum[base_row + minCharacterHeight - 2] + line_sum[base_row + kSmallCharacterHeight - 1] < threshold) {
      continue;
    }
    
    // Eliminate stripes that contain a much dimmer-than-average sub-stripe,
    // since that usually means that we've been fooled into grabbing the bottom
    // of some card feature and the top of a different card feature.
    bool isGoodStripe = true;
    for (row = base_row; row < base_row + minCharacterHeight - 3; row++) {
      if (line_sum[row + 1] < threshold && line_sum[row + 2] < threshold) {
        isGoodStripe = false;
        break;
      }
    }
    if (!isGoodStripe) {
      continue;
    }

    StripeSum stripe_sum;
    stripe_sum.base_row = base_row;
    stripe_sum.height = minCharacterHeight;
    stripe_sum.sum = sum;

    // While successive scan line sums are also > threshold, append them to the stripe
    for (row = base_row + minCharacterHeight; row < last_stripe_base_row && stripe_sum.height <= maxCharacterHeight; row++) {
      if (line_sum[row] >= threshold) {
        stripe_sum.height++;
        stripe_sum.sum += line_sum[row];
      }
    }

    // Save the result
    stripe_sums.push_back(stripe_sum);
  }
  
#if DEBUG_STRIPES_PERFORMANCE
  dmz_debug_timer_print("sum stripes");
#endif
  
  std::sort(stripe_sums.begin(), stripe_sums.end(), StripeSumCompareDescending());
  
#if DEBUG_STRIPES_PERFORMANCE
  dmz_debug_timer_print("sort stripe sums");
#endif

  return stripe_sums;
}

#endif // COMPILE_DMZ
