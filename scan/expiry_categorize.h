//
//  expiry_categorize.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_EXPIRY_CATEGORIZE_H
#define DMZ_SCAN_EXPIRY_CATEGORIZE_H

#include "expiry_types.h"

#include "opencv2/core/core_c.h" // needed for IplImage
#include "dmz_macros.h"

DMZ_INTERNAL void expiry_extract(IplImage *cardY,
                                 GroupedRectsList &expiry_groups,
                                 GroupedRectsList &new_groups,
                                 int *expiry_month,
                                 int *expiry_year);

#if CYTHON_DMZ
DMZ_INTERNAL void expiry_extract_group(IplImage *card_y,
                                       GroupedRects &group,
                                       ExpiryGroupScores &old_scores,
                                       int *expiry_month,
                                       int *expiry_year);
#endif
  
#endif
