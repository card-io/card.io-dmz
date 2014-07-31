//
//  n_categorize.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_N_CATEGORIZE_H
#define DMZ_SCAN_N_CATEGORIZE_H

#include "opencv2/core/core_c.h" // needed for IplImage
#include "eigen.h"
#include "n_hseg.h"
#include "dmz_macros.h"

typedef Eigen::Matrix<float, 16, 10, Eigen::RowMajor> NumberScores;  // (up to) 16 numbers, 10 possibilities each

// May alter any roi that y_strip may have prior to returning. (The inbound roi will be respected,
// it'll just be changed at the end.) If this is unwanted, pass in a copy of y_strip.
DMZ_INTERNAL NumberScores number_scores(IplImage *y_strip, NHorizontalSegmentation hseg);


#endif
