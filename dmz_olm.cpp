//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"

#include <iostream>
#include "dmz_olm.h"
#include "processor_support.h"
#include "opencv2/core/types_c.h"


#pragma mark points and rects

dmz_point dmz_create_point(float x, float y) { 
  dmz_point d; 
  d.x = x; 
  d.y = y; 
  return d; 
}

dmz_point dmz_scale_point(const dmz_point src_p, const dmz_rect src_f, const dmz_rect dst_f) {
  return dmz_create_point(dst_f.x + (src_p.x - src_f.x) * dst_f.w / src_f.w, 
                          dst_f.y + (src_p.y - src_f.y) * dst_f.h / src_f.h);
}

dmz_rect dmz_create_rect(float x, float y, float w, float h) {
  dmz_rect r;
  r.x = x, r.y = y, r.w = w, r.h = h;
  return r;
}

void dmz_rect_get_points(dmz_rect rect, dmz_point points[4]) {
  points[0] = dmz_create_point(rect.x, rect.y);
  points[1] = dmz_create_point(rect.x + rect.w, rect.y);
  points[2] = dmz_create_point(rect.x, rect.y + rect.h);
  points[3] = dmz_create_point(rect.x + rect.w, rect.y + rect.h);
}

#pragma mark card number validation and typing

bool dmz_passes_luhn_checksum(uint8_t *number_array, uint8_t number_length) {
  int even = 0;
  int sum = 0;
  for(int i = number_length - 1; i >= 0; i--) {
    uint8_t number_at_index = number_array[i];
    int addend = number_at_index * (1 << (even++ & 1));
    sum += addend % 10 + addend / 10;
  }
  return sum % 10 == 0;
}

dmz_card_info dmz_card_info_for_prefix_and_length(uint8_t *number_array, uint8_t number_length, bool allow_incomplete_number) {
  // Primarily based on http://en.wikipedia.org/wiki/Credit_card_numbers
  //
  // But re Maestro, even just Maestro UK, there's confusing and conflicting info out on the web.
  // See, for example, http://en.wikipedia.org/wiki/List_of_Issuer_Identification_Numbers
  // and also http://www.barclaycard.co.uk/business/documents/pdfs/bin_rules.pdf
  // So the Maestro UK rules here are *very* overly loose and inclusive.
  
  dmz_card_info card_types[] =
  {
    {CardTypeMastercard,  16, 4, 2221, 2720},      // MasterCard 2-Series
    {CardTypeDiscover,    14, 3, 300, 305},        // Diners Club (Discover)
    {CardTypeDiscover,    14, 3, 309, 309},        // Diners Club (Discover)
    {CardTypeAmex,        15, 2, 34, 34},          // AmEx
    {CardTypeJCB,         16, 4, 3528, 3589},      // JCB
    {CardTypeDiscover,    14, 2, 36, 36},          // Diners Club (Discover)
    {CardTypeDiscover,    14, 2, 38, 39},          // Diners Club (Discover)
    {CardTypeAmex,        15, 2, 37, 37},          // AmEx
    {CardTypeVisa,        16, 1, 4, 4},            // VISA
    {CardTypeMaestro,     16, 2, 50, 50},          // Maestro
    {CardTypeMastercard,  16, 2, 51, 55},          // MasterCard
    {CardTypeMaestro,     16, 2, 56, 59},          // Maestro
    {CardTypeDiscover,    16, 4, 6011, 6011},      // Discover
    {CardTypeMaestro,     16, 2, 61, 61},          // Maestro
    {CardTypeDiscover,    16, 2, 62, 62},          // China UnionPay (Discover)
    {CardTypeMaestro,     16, 2, 63, 63},          // Maestro
    {CardTypeDiscover,    16, 3, 644, 649},        // Discover
    {CardTypeDiscover,    16, 2, 65, 65},          // Discover
    {CardTypeMaestro,     16, 2, 66, 69},          // Maestro
    {CardTypeDiscover,    16, 2, 88, 88},          // China UnionPay (Discover)
  };
  
  dmz_card_info card_type_unrecognized = {CardTypeUnrecognized, -1, 1, 9, 9};
  dmz_card_info card_type_ambiguous = {CardTypeAmbiguous, -1, 1, 9, 9};
  
  if (number_length > 0) {
    dmz_card_info card_info = card_type_unrecognized;
    int number_of_compatible_card_types = 0;
    
    for (int i = 0; i < sizeof(card_types) / sizeof(dmz_card_info); i++) {
      dmz_card_info info = card_types[i];
      if (allow_incomplete_number) {
        if (number_length > info.number_length) {
          continue;
        }
      }
      else if (number_length != info.number_length) {
        continue;
      }

      int relevant_prefix_length = info.prefix_length;
      int factor = 1;
      while (relevant_prefix_length > number_length) {
        factor *= 10;
        relevant_prefix_length--;
      }
      
      long card_prefix = 0;
      for (int j = 0; j < relevant_prefix_length; j++) {
        card_prefix = (card_prefix * 10) + number_array[j];
      }
      
      if (card_prefix >= (info.min_prefix / factor) && card_prefix <= (info.max_prefix / factor)) {
        number_of_compatible_card_types++;
        card_info = info;
      }
    }
    
    if (number_of_compatible_card_types > 0) {
      if (number_of_compatible_card_types == 1) {
        return card_info;
      }
      else {
        return card_type_ambiguous;
      }
    }
  }
  
  return card_type_unrecognized;
}

#pragma mark other

dmz_rect dmz_guide_frame(FrameOrientation orientation, float preview_width, float preview_height) {
  dmz_rect guide;
  float inset_w;
  float inset_h;

  switch(orientation) {
    case FrameOrientationPortrait:
        /* no break */
    case FrameOrientationPortraitUpsideDown:
      inset_w = kPortraitHorizontalPercentInset * preview_width;
      inset_h = kPortraitVerticalPercentInset * preview_height;
      break;
    case FrameOrientationLandscapeLeft:
        /* no break */
    case FrameOrientationLandscapeRight:
      inset_w = kLandscapeVerticalPercentInset * preview_width;
      inset_h = kLandscapeHorizontalPercentInset * preview_height;
      break;
    default:
      inset_w = 0.0f;
      inset_h = 0.0f;
      break;
  }
  
  guide.x = inset_w;
  guide.y = inset_h;
  guide.w = preview_width - 2.0f * inset_w;
  guide.h = preview_height - 2.0f * inset_h;
  
  return guide;
}

FrameOrientation dmz_opposite_orientation(FrameOrientation orientation) {
  switch(orientation) {
    case FrameOrientationPortrait:
      return FrameOrientationPortraitUpsideDown;
    case FrameOrientationPortraitUpsideDown:
      return FrameOrientationPortrait;
    case FrameOrientationLandscapeRight:
      return FrameOrientationLandscapeLeft;
    case FrameOrientationLandscapeLeft:
      return FrameOrientationLandscapeRight;
    default:
      return FrameOrientationPortrait;
  }
}
