//
//  n_categorize.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "n_categorize.h"
#include "cv/image_util.h"
#include "cv/morph.h"
#include "cv/stats.h"

// conv models
#include "models/generated/modelc_5c241121.hpp"
#include "models/generated/modelc_01266c1b.hpp"
#include "models/generated/modelc_b00bf70c.hpp"

// TODO: gpu for matrix mult?


typedef Eigen::Matrix<float, 27, 19, Eigen::RowMajor> NumberImage;
typedef Eigen::Matrix<float, 1, 10, Eigen::RowMajor> SingleNumberScores;


// TODO: Refactor me
DMZ_INTERNAL inline NumberImage matrix_for_number_image(IplImage *number_image) {
  CvSize image_size = cvGetSize(number_image);
  
  NumberImage m;
  uint8_t *data_origin = (uint8_t *)llcv_get_data_origin(number_image); // use uint8_t so that the widthStep calculation is easier
  
  // TODO: Do this the not-stupid way (set up a Map with the right strides and just alias the memory)
  for(int i = 0; i < image_size.height; i++){
    float *row_origin = (float *)(data_origin + i * number_image->widthStep);
    for(int j = 0; j < image_size.width; j++) {
      m(i, j) = row_origin[j];
    }
  }
  
  return m;
}


DMZ_INTERNAL inline SingleNumberScores scores_for_number_image(IplImage *number_image) {
  NumberImage image_matrix = matrix_for_number_image(number_image);
  
  // The values in result[0|1|2] are probabilities, but once we munge them together, they just become scores
  SingleNumberScores result0 = applyc_5c241121(image_matrix);
  SingleNumberScores result1 = applyc_01266c1b(image_matrix);
  SingleNumberScores result2 = applyc_b00bf70c(image_matrix);

  // Strategy: Add the three scores together, subtract the highest for any given offset,
  // and then divide by two. The result should be that numbers with 3/3 votes (across
  // the models) get a score near 1.0, numbers with 2/3 votes get a score near 0.5,
  // and models 1/3 or 0/3 get a score near 0.0.
  //
  // The idea is to require confidence in our estimations (1/3 means nothing), and also
  // reward very high confidence (3/3 > 2/3).
  //
  // This has not been empirically tested against a test data set. There are lots of
  // other available strategies (e.g. take the median, mean, sum, product, etc.).
  // With > 3 models, there are even more options.
  //
  // TODO: Use a real test framework to decide whether this is the best approach.

  // NB: If you change the calculation, or the overall meaning or range of these scores,
  // be sure to update frame.cpp's usability decisions.
  SingleNumberScores result_max = result0.cwiseMax(result1).cwiseMax(result2);
  SingleNumberScores scores_matrix = (result0 + result1 + result2 - result_max) / 2.0f;
  return scores_matrix;
}


DMZ_INTERNAL NumberScores number_scores(IplImage *y_strip, NHorizontalSegmentation hseg) {
  // y_strip might have been made into a strip by using a vertical ROI -- must preserve and use y_offset in that case
  // though slightly complex, this is better than making an unneeded copy
  CvSize y_strip_size = cvGetSize(y_strip);
#pragma unused(y_strip_size) // work around broken compiler warnings
  assert(y_strip_size.height == 27);
  uint16_t y_offset = 0;
  if(NULL != y_strip->roi) {
    y_offset = (uint16_t)y_strip->roi->yOffset;
    assert(y_strip->roi->width == 428);
    assert(y_strip->roi->xOffset == 0);
  }

  // TODO: Set up these temporary variables once during initialization, and reuse?
  // Would be nicer yet to just have static allocation...
  IplImage *number_image = cvCreateImage(cvSize(19, 27), y_strip->depth, 1);
  IplImage *number_image_float = cvCreateImage(cvSize(19, 27), IPL_DEPTH_32F, 1);
  
  NumberScores scores = NumberScores::Zero();
  for(uint8_t offset_index = 0; offset_index < hseg.n_offsets; offset_index++) {
    uint16_t offset = hseg.offsets[offset_index];
    cvSetImageROI(y_strip, cvRect(offset, y_offset, 19, 27));
    llcv_morph_grad3_2d_cross_u8(y_strip, number_image);
    llcv_equalize_hist(number_image, number_image);
    cvConvertScale(number_image, number_image_float, 1.0f / 255.0f, 0.0f);
    SingleNumberScores single_number_scores = scores_for_number_image(number_image_float);
    scores.row(offset_index) = single_number_scores;
  }
  
  cvReleaseImage(&number_image_float);
  cvReleaseImage(&number_image);
  
  return scores;
}



#endif // COMPILE_DMZ
