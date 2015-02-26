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
// Maximum stripe-sum possible = (max line-sum possible) * kSmallCharacterHeight = 26,193,600 = 0x18FAEC0, so stripe-sum will always fit in a long.

struct StripeSumCompareDescending
: public std::binary_function<StripeSum, StripeSum, bool> {
  inline bool operator()(StripeSum const &stripe_sum_1, StripeSum const &stripe_sum_2) const {
    return (stripe_sum_1.sum > stripe_sum_2.sum);
  }
};

std::vector<StripeSum> sorted_stripes(IplImage *card_y) {
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_start();
#endif
  
  CvSize card_image_size = cvGetSize(card_y);
  
  // Look for vertical line segments -> sobel_image:
  
  IplImage *sobel_image = cvCreateImage(card_image_size, IPL_DEPTH_16S, 1);
  cvSetZero(sobel_image);
  
  CvRect below_numbers_rect = cvRect(0, starting_y_offset + kNumberHeight, card_image_size.width, card_image_size.height - (starting_y_offset + kNumberHeight));
  cvSetImageROI(card_y, below_numbers_rect);
  cvSetImageROI(sobel_image, below_numbers_rect);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("set up for Sobel");
#endif
  
  llcv_scharr3_dx_abs(card_y, sobel_image);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("do Sobel [Scharr]");
#endif
  
  cvResetImageROI(card_y);
  cvResetImageROI(sobel_image);
  
  // Calculate relative vertical-line-segment-ness for each scan line (i.e., cvSum of the [x-axis] Sobel image for that line):
  
  int   first_stripe_base_row = below_numbers_rect.y + 1;  // the "+ 1" represents the tolerance above and below each stripe
  int   last_stripe_base_row = card_image_size.height - (kSmallCharacterHeight + 1);  // the "+ 1" represents the tolerance above and below each stripe
  long  line_sum[card_image_size.height];
  
  int   left_edge = kSmallCharacterWidth * 3;  // there aren't usually any actual characters this far to the left
  int   right_edge = (card_image_size.width * 2) / 3;  // beyond here lie logos
  
  for (int row = first_stripe_base_row - 1; row < card_image_size.height; row++) {
    cvSetImageROI(sobel_image, cvRect(left_edge, row, right_edge - left_edge, 1));
    line_sum[row] = (long)cvSum(sobel_image).val[0];
  }
  
  cvResetImageROI(sobel_image);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("line sums");
#endif
  
  // Determine the 3 most probable, non-overlapping stripes. (Where "stripe" == kSmallCharacterHeight contiguous scan lines.)
  // (Two will usually get us expiry and name, but some cards have additional distractions.)
  
#define kNumberOfStripesToTry 3
  int row;
  std::vector<StripeSum> stripe_sums;
  for (int base_row = first_stripe_base_row; base_row < last_stripe_base_row; base_row++) {
    long sum = 0;
    for (int row = base_row; row < base_row + kSmallCharacterHeight; row++) {
      sum += line_sum[row];
    }
    
    // Calculate threshold = half the value of the maximum line-sum in the stripe:
    long threshold = 0;
    for (row = base_row; row < base_row + kSmallCharacterHeight; row++) {
      if (line_sum[row] > threshold) {
        threshold = line_sum[row];
      }
    }
    threshold = threshold / 2;
    
    // Eliminate stripes that have a a much dimmer-than-average sub-stripe at their very top or very bottom:
    if (line_sum[base_row] + line_sum[base_row + 1] < threshold) {
      continue;
    }
    if (line_sum[base_row + kSmallCharacterHeight - 2] + line_sum[base_row + kSmallCharacterHeight - 1] < threshold) {
      continue;
    }
    
    // Eliminate stripes that contain a much dimmer-than-average sub-stripe,
    // since that usually means that we've been fooled into grabbing the bottom
    // of some card feature and the top of a different card feature.
    bool isGoodStrip = true;
    for (row = base_row; row < base_row + kSmallCharacterHeight - 3; row++) {
      if (line_sum[row + 1] < threshold && line_sum[row + 2] < threshold) {
        isGoodStrip = false;
        break;
      }
    }
    
    if (isGoodStrip) {
      StripeSum stripe_sum;
      stripe_sum.base_row = base_row;
      stripe_sum.sum = sum;
      stripe_sums.push_back(stripe_sum);
    }
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("sum stripes");
#endif
  
  std::sort(stripe_sums.begin(), stripe_sums.end(), StripeSumCompareDescending());
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("sort stripe sums");
#endif
  
}

#endif // COMPILE_DMZ
