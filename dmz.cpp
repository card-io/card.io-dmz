//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

#include <iostream>
#include "dmz.h"
#include "eigen.h"
#include "processor_support.h"
#include "geometry.h"
#include "cv/canny.h"
#include "cv/convert.h"
#include "cv/hough.h"
#include "cv/sobel.h"
#include "cv/stats.h"
#include "cv/warp.h"
#include "opencv2/core/core_c.h" // needed for IplImage
#include "opencv2/imgproc/imgproc.hpp"
#include "scan/scan.h"

#pragma mark life cycle

dmz_context *dmz_context_create(void) {
  dmz_context *dmz = (dmz_context *) calloc(1, sizeof(dmz_context));
  dmz->mz = mz_create();
  return dmz;
}

void dmz_context_destroy(dmz_context *dmz) {
  mz_destroy(dmz->mz);
  free(dmz);
}

void dmz_prepare_for_backgrounding(dmz_context *dmz) {
  mz_prepare_for_backgrounding(dmz->mz);
}

#pragma mark utils / conversions

#define kASCIIOffset_A ('A' - '9')

int dmz_has_opencv() {
    IplImage *test_image = cvCreateImage(cvSize(kCreditCardTargetWidth, kCreditCardTargetHeight), IPL_DEPTH_8U, 1);
    int rv = test_image != NULL;
    cvReleaseImage(&test_image);
    return rv;
}

void dmz_deinterleave_uint8_c2(IplImage *interleaved, IplImage **channel1, IplImage **channel2) {
  CvSize image_size = cvGetSize(interleaved);

  *channel1 = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
  *channel2 = cvCreateImage(image_size, IPL_DEPTH_8U, 1);

  llcv_split_u8(interleaved, *channel1, *channel2);
}


void dmz_YCbCr_to_RGB(IplImage *y, IplImage *cb, IplImage *cr, IplImage **rgb) {
  if (*rgb == NULL) {
	*rgb = cvCreateImage(cvGetSize(y), y->depth, 3);
  }
  llcv_YCbCr2RGB_u8(y, cb, cr, *rgb);
}

void dmz_deinterleave_RGBA_to_R(uint8_t *source, uint8_t *dest, int size) {
#if DMZ_HAS_NEON_COMPILETIME
  if (dmz_has_neon_runtime()) {
    assert(size >= 16); // required for the vectorized handling of leftover_bytes; also, a reasonable expectation!

    for (int offset = 0; offset + 15 < size; offset += 16) {
      uint8x16x4_t r1 = vld4q_u8(&source[offset * 4]);
      vst1q_u8(&dest[offset], r1.val[0]);
    }

    // use "overlapping" to process the remaining bytes
    // See http://community.arm.com/groups/processors/blog/2010/05/10/coding-for-neon--part-2-dealing-with-leftovers
    if (size % 16 > 0) {
      int offset = size - 16;
      uint8x16x4_t r1 = vld4q_u8(&source[offset * 4]);
      vst1q_u8(&dest[offset], r1.val[0]);
    }
  }
  else
#endif
  {
    for (int offset = 0; offset + 7 < size; offset += 8) {
      int bufferOffset = offset * 4;
      dest[offset] = source[bufferOffset];
      dest[offset + 1] = source[bufferOffset + (1 * 4)];
      dest[offset + 2] = source[bufferOffset + (2 * 4)];
      dest[offset + 3] = source[bufferOffset + (3 * 4)];
      dest[offset + 4] = source[bufferOffset + (4 * 4)];
      dest[offset + 5] = source[bufferOffset + (5 * 4)];
      dest[offset + 6] = source[bufferOffset + (6 * 4)];
      dest[offset + 7] = source[bufferOffset + (7 * 4)];
    }
    
    int leftover_bytes = size % 8; // each RGBA pixel is 4 bytes, so can assume size % 4 == 0
    if (leftover_bytes > 0) {
      for (int offset = size - leftover_bytes; offset < size; offset += 4) {
        int bufferOffset = offset * 4;
        dest[offset] = source[bufferOffset];
        dest[offset + 1] = source[bufferOffset + (1 * 4)];
        dest[offset + 2] = source[bufferOffset + (2 * 4)];
        dest[offset + 3] = source[bufferOffset + (3 * 4)];
      }
    }
  }
}

#pragma mark focus score

float dmz_focus_score_for_image(IplImage *image) {
  assert(image->nChannels == 1);
  assert(image->depth == IPL_DEPTH_8U);

  CvSize image_size = cvGetSize(image);
  IplImage *sobel_image = cvCreateImage(image_size, IPL_DEPTH_16S, 1);

  llcv_sobel3_dx_dy(image, sobel_image);

  float stddev = llcv_stddev_of_abs(sobel_image);
  cvReleaseImage(&sobel_image);
  return stddev;
}

float dmz_brightness_score_for_image(IplImage *image) {
  assert(image->nChannels == 1);
  assert(image->depth == IPL_DEPTH_8U);
  
  // could Neon and/or GPU this; however, this call to cvAvg apparently has NO effect on FPS (iPhone 4S)
  CvScalar mean = cvAvg(image, NULL);
  return (float)mean.val[0];
}

CvRect dmz_card_rect_for_screen(CvSize standardCardSize, CvSize standardScreenSize, CvSize actualScreenSize) {
  if (standardCardSize.width == 0 || standardCardSize.height == 0 ||
      standardScreenSize.width == 0 || standardScreenSize.height == 0 ||
      actualScreenSize.width == 0 || actualScreenSize.height == 0) {
    return cvRect(0, 0, 0, 0);
  }
  
  CvRect actualCardRect;
  
  if (actualScreenSize.width == standardScreenSize.width && actualScreenSize.height == standardScreenSize.height) {
    actualCardRect.width = standardCardSize.width;
    actualCardRect.height = standardCardSize.height;
  }
  else {
    float screenWidthRatio = ((float)actualScreenSize.width) / ((float)standardScreenSize.width);
    float screenHeightRatio = ((float)actualScreenSize.height) / ((float)standardScreenSize.height);
    float screenRatio = MIN(screenWidthRatio, screenHeightRatio);
    
    actualCardRect.width = (int)(standardCardSize.width * screenRatio);
    actualCardRect.height = (int)(standardCardSize.height * screenRatio);
  }

  actualCardRect.x = (actualScreenSize.width - actualCardRect.width) / 2;
  actualCardRect.y = (actualScreenSize.height - actualCardRect.height) / 2;
  
  return actualCardRect;
}

void dmz_set_roi_for_scoring(IplImage *image, bool use_full_image) {
  // Usually we calculate the focus score only on the center 1/9th of the credit card
  // in the image (assume it is centered), for performance reasons
  CvSize focus_size;
  if (use_full_image) {
    focus_size = cvSize(kCreditCardTargetWidth, kCreditCardTargetHeight);
  }
  else {
    focus_size = cvSize(kCreditCardTargetWidth / 3, kCreditCardTargetHeight / 3);
  }
  
  CvRect focus_rect = dmz_card_rect_for_screen(focus_size,
                                               cvSize(kLandscapeSampleWidth, kLandscapeSampleHeight),
                                               cvGetSize(image));
  
  cvSetImageROI(image, focus_rect);
}

float dmz_focus_score(IplImage *image, bool use_full_image) {
  dmz_set_roi_for_scoring(image, use_full_image);
  float focus_score = dmz_focus_score_for_image(image);
  cvResetImageROI(image);
  return focus_score;
}

float dmz_brightness_score(IplImage *image, bool use_full_image) {
  dmz_set_roi_for_scoring(image, use_full_image);
  float focus_score = dmz_brightness_score_for_image(image);
  cvResetImageROI(image);
  return focus_score;
}

#pragma mark detection

#define kHoughGradientAngleThreshold 10

#define kHoughThresholdLengthDivisor 6 // larger value --> accept more lines as lines

#define kHorizontalAngle ((float)(CV_PI / 2.0f))
#define kVerticalAngle ((float)CV_PI)
#define kMaxAngleDeviationAllowed ((float)(5.0f * (CV_PI / 180.0f)))

#define kVerticalPercentSlop 0.03f
#define kHorizontalPercentSlop 0.03f

typedef struct {
  CvRect top;
  CvRect bottom;
  CvRect left;
  CvRect right;
} DetectionBoxes;

enum {
  LineOrientationVertical = 0,
  LineOrientationHorizontal = 1,
};
typedef uint8_t LineOrientation;

#pragma mark: best_line_for_sample
ParametricLine best_line_for_sample(IplImage *image, LineOrientation expectedOrientation) {
  bool expected_vertical = expectedOrientation == LineOrientationVertical;

  // Calculate dx and dy derivatives; they'll be reused a lot throughout
  CvSize image_size = cvGetSize(image);
  assert(image_size.width > 0 && image_size.height > 0);
  dmz_trace_log("looking for best line in %ix%i patch with orientation:%i", image_size.width, image_size.height, expectedOrientation);
  IplImage *sobel_scratch = cvCreateImage(cvSize(image_size.height, image_size.width), IPL_DEPTH_16S, 1);
  assert(sobel_scratch != NULL);
  IplImage *dx = cvCreateImage(image_size, IPL_DEPTH_16S, 1);
  assert(dx != NULL);
  IplImage *dy = cvCreateImage(image_size, IPL_DEPTH_16S, 1);
  assert(dy != NULL);
  llcv_sobel7(image, dx, sobel_scratch, 1, 0);
  llcv_sobel7(image, dy, sobel_scratch, 0, 1);
  cvReleaseImage(&sobel_scratch);

  // Calculate the canny image
  IplImage *canny_image = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
  llcv_adaptive_canny7_precomputed_sobel(image, canny_image, dx, dy);

  // Calculate the hough transform, throwing away edge components with the wrong gradient angles
  int hough_accumulator_threshold = MAX(image_size.width, image_size.height) / kHoughThresholdLengthDivisor;
  float base_angle = expected_vertical ? kVerticalAngle : kHorizontalAngle;
  float theta_min = base_angle - kMaxAngleDeviationAllowed;
  float theta_max = base_angle + kMaxAngleDeviationAllowed;

  CvLinePolar best_line = llcv_hough(canny_image,
                                     dx, dy,
                                     1, // rho resolution
                                     (float)CV_PI / 180.0f, // theta resolution
                                     hough_accumulator_threshold,
                                     theta_min,
                                     theta_max,
                                     expected_vertical,
                                     kHoughGradientAngleThreshold);
  
  ParametricLine ret = ParametricLineNone();
  if(!best_line.is_null) {
    ret.rho = best_line.rho;
    ret.theta = best_line.angle;
  }

  cvReleaseImage(&dx);
  cvReleaseImage(&dy);
  cvReleaseImage(&canny_image);
  return ret;
}

#pragma mark: dmz_found_all_edges
bool dmz_found_all_edges(dmz_edges found_edges) {
  return (found_edges.top.found && found_edges.bottom.found && found_edges.left.found && found_edges.right.found);
}

#pragma mark: detection_boxes_for_sample
DetectionBoxes detection_boxes_for_sample(IplImage *sample, FrameOrientation orientation) {
  CvSize size = cvGetSize(sample);
  dmz_trace_log("detection_boxes_for_sample sized %ix%i with orientation:%i", size.width, size.height, orientation);
  int absolute_inset_vert, absolute_slop_vert, absolute_inset_horiz, absolute_slop_horiz;

  // Regardless of the dimensions of the incoming image (640x480, 1280x720, etc),
  // we do everything based on the central 4:3 rectangle (which for 640x480 is the entire image).
  int width = (size.height * 4) / 3;
  int leftMargin = (size.width - width) / 2;
  size.width = width;
  
  switch(orientation) {
    case FrameOrientationPortrait:
    /* no break */
    case FrameOrientationPortraitUpsideDown:
      absolute_inset_vert = (int)roundf(kPortraitHorizontalPercentInset * size.height);
      absolute_slop_vert = (int)roundf(kHorizontalPercentSlop * size.height);
      absolute_inset_horiz = (int)roundf(kPortraitVerticalPercentInset * size.width);
      absolute_slop_horiz = (int)roundf(kVerticalPercentSlop * size.width);
      break;
    case FrameOrientationLandscapeLeft:
    /* no break */
    case FrameOrientationLandscapeRight:
      absolute_inset_vert = (int)roundf(kLandscapeVerticalPercentInset * size.height);
      absolute_slop_vert = (int)roundf(kHorizontalPercentSlop * size.height);
      absolute_inset_horiz = (int)roundf(kLandscapeHorizontalPercentInset * size.width);
      absolute_slop_horiz = (int)roundf(kVerticalPercentSlop * size.width);
      break;
    default:
      absolute_inset_vert = 0;
      absolute_slop_vert = 0;
      absolute_inset_horiz = 0;
      absolute_slop_horiz = 0;
      break;
  }

  CvRect image_rect = cvRect(leftMargin, 0, size.width - 1, size.height - 1);
  CvRect outerSlopRect = cvInsetRect(image_rect,
                                     absolute_inset_horiz - absolute_slop_horiz,
                                     absolute_inset_vert - absolute_slop_vert);
  CvRect innerSlopRect = cvInsetRect(image_rect,
                                     absolute_inset_horiz + absolute_slop_horiz,
                                     absolute_inset_vert + absolute_slop_vert);

  DetectionBoxes boxes;

  boxes.top = cvRect(innerSlopRect.x, outerSlopRect.y,
                     innerSlopRect.width, 2 * absolute_slop_vert);
  dmz_trace_log("boxes.top: {x:%i y:%i w:%i h:%i}", boxes.top.x, boxes.top.y, boxes.top.width, boxes.top.height);
  boxes.bottom = cvRect(innerSlopRect.x, innerSlopRect.y + innerSlopRect.height,
                        innerSlopRect.width, 2 * absolute_slop_vert);
  dmz_trace_log("boxes.bottom: {x:%i y:%i w:%i h:%i}", boxes.bottom.x, boxes.bottom.y, boxes.bottom.width, boxes.bottom.height);

  boxes.left = cvRect(outerSlopRect.x, innerSlopRect.y,
                      2 * absolute_slop_horiz, innerSlopRect.height);
  dmz_trace_log("boxes.left: {x:%i y:%i w:%i h:%i}", boxes.left.x, boxes.left.y, boxes.left.width, boxes.left.height);
  
  boxes.right = cvRect(innerSlopRect.x + innerSlopRect.width, innerSlopRect.y,
                       2 * absolute_slop_horiz, innerSlopRect.height);
  dmz_trace_log("boxes.right: {x:%i y:%i w:%i h:%i}", boxes.right.x, boxes.right.y, boxes.right.width, boxes.right.height);

  return boxes;
}

#define kNumColorPlanes 3

#pragma mark: find_line_in_detection_rects
void find_line_in_detection_rects(IplImage **samples, float *rho_multiplier, CvRect *detection_rects, dmz_found_edge *found_edge, LineOrientation line_orientation) {
  assert(detection_rects != NULL);
  assert(found_edge != NULL);
  assert(samples != NULL);
  dmz_trace_log("inputs to find_line_in_detection_rects are valid");
  for(int i = 0; i < kNumColorPlanes && !found_edge->found; i++) {
    IplImage *image = samples[i];
    assert(image != NULL);
    #if DMZ_TRACE
    CvSize imageSize = cvGetSize(image);
    dmz_trace_log("sample %i has size %ix%i", i, imageSize.width, imageSize.height);
    CvRect r = detection_rects[i];
    dmz_trace_log("detection_rect {x:%i y:%i w:%i h:%i}", r.x, r.y, r.width, r.height);
    #endif
    cvSetImageROI(image, detection_rects[i]);
    ParametricLine local_edge = best_line_for_sample(image, line_orientation);
    dmz_trace_log("local_edge - {rho:%f theta:%f}", local_edge.rho, local_edge.theta);
    cvResetImageROI(image);
    found_edge->location = lineByShiftingOrigin(local_edge, detection_rects[i].x, detection_rects[i].y);
    found_edge->location.rho *= rho_multiplier[i];
    found_edge->found = !is_parametric_line_none(found_edge->location);
  }
  dmz_trace_log("resulting edge - {found:%i ...}", found_edge->found);
}

bool dmz_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,
                      FrameOrientation orientation, dmz_edges *found_edges, dmz_corner_points *corner_points) {
  assert(y_sample != NULL);
  assert(cb_sample != NULL);
  assert(cr_sample != NULL);
  assert(found_edges != NULL);
  assert(corner_points != NULL);

  dmz_trace_log("dmz_detect_edges");

  IplImage *samples[kNumColorPlanes] = {y_sample, cb_sample, cr_sample};
  DetectionBoxes boxes[kNumColorPlanes];
  float rho_multiplier[kNumColorPlanes] = {1.0f, 2.0f, 2.0f}; // cb and cr are half the size of Y

  for(int i = 0; i < kNumColorPlanes; i++) {
    boxes[i] = detection_boxes_for_sample(samples[i], orientation);
  }

  dmz_trace_log("got boxes, looking for lines...");

  found_edges->top.found = 0;
  found_edges->bottom.found = 0;
  found_edges->left.found = 0;
  found_edges->right.found = 0;

  CvRect detection_rects[kNumColorPlanes];

  for(uint8_t i = 0; i < kNumColorPlanes; i++) {
    detection_rects[i] = boxes[i].top;
  }
  find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->top, LineOrientationHorizontal);
  dmz_trace_log("dmz top edge? %i", found_edges->top.found);

  for(uint8_t i = 0; i < kNumColorPlanes; i++) {
    detection_rects[i] = boxes[i].bottom;
  }
  find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->bottom, LineOrientationHorizontal);
  dmz_trace_log("dmz bottom edge? %i", found_edges->bottom.found);

  for(uint8_t i = 0; i < kNumColorPlanes; i++) {
    detection_rects[i] = boxes[i].left;
  }
  find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->left, LineOrientationVertical);
  dmz_trace_log("dmz left edge? %i", found_edges->left.found);

  for(uint8_t i = 0; i < kNumColorPlanes; i++) {
    detection_rects[i] = boxes[i].right;
  }
  find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->right, LineOrientationVertical);
  dmz_trace_log("dmz right edge? %i", found_edges->right.found);

  // Find corner intersections
  bool found_all_corners = true;
  if(dmz_found_all_edges(*found_edges)) {
    bool tl_intersects = parametricIntersect(found_edges->top.location, found_edges->left.location, &corner_points->top_left.x, &corner_points->top_left.y);
    bool bl_intersects = parametricIntersect(found_edges->bottom.location, found_edges->left.location, &corner_points->bottom_left.x, &corner_points->bottom_left.y);
    bool tr_intersects = parametricIntersect(found_edges->top.location, found_edges->right.location, &corner_points->top_right.x, &corner_points->top_right.y);
    bool br_intersects = parametricIntersect(found_edges->bottom.location, found_edges->right.location, &corner_points->bottom_right.x, &corner_points->bottom_right.y);
    int all_intersect = tl_intersects && bl_intersects && tr_intersects && br_intersects;
    if(!all_intersect) {
      // never seen this happen, but best to be safe
      found_all_corners = false;
    }
  } else {
    found_all_corners = false;
  }

  return found_all_corners;
}

#pragma mark transform

void dmz_transform_card(dmz_context *dmz, IplImage *sample, dmz_corner_points corner_points, FrameOrientation orientation, bool upsample, IplImage **transformed) {
  
  dmz_point src_points[4];
  switch(orientation) {
    case FrameOrientationPortrait:
      src_points[0] = corner_points.bottom_left;
      src_points[1] = corner_points.top_left;
      src_points[2] = corner_points.bottom_right;
      src_points[3] = corner_points.top_right;
      break;
    case FrameOrientationLandscapeLeft:
      src_points[0] = corner_points.bottom_right;
      src_points[1] = corner_points.bottom_left;
      src_points[2] = corner_points.top_right;
      src_points[3] = corner_points.top_left;
      break;
    case FrameOrientationLandscapeRight: // this is the canonical one
      src_points[0] = corner_points.top_left;
      src_points[1] = corner_points.top_right;
      src_points[2] = corner_points.bottom_left;
      src_points[3] = corner_points.bottom_right;
      break;
    case FrameOrientationPortraitUpsideDown:
      src_points[0] = corner_points.top_right;
      src_points[1] = corner_points.bottom_right;
      src_points[2] = corner_points.top_left;
      src_points[3] = corner_points.bottom_left;
      break;
  }
  
  if(upsample) {
    if(!llcv_warp_auto_upsamples()) {
      // upsample source_points, since CbCr are half size.
      for(unsigned int i = 0; i < sizeof(dmz_corner_points) / sizeof(dmz_point); i++) {
        src_points[i].x /= 2.0f;
        src_points[i].y /= 2.0f;
      }    
    }
  }
  
  // Destination rectangle is the same as the size of the image
  dmz_rect dst_rect = dmz_create_rect(0, 0, kCreditCardTargetWidth - 1, kCreditCardTargetHeight - 1);
  
  int nChannels = sample->nChannels;
#if ANDROID_USE_GLES_WARP
  // override because OpenGLES 1.1 only supports RGBA in glReadPixels!!
  if (dmz_use_gles_warp()) nChannels = 4;
#endif

  // Some environments (Android) may prefer to dictate where the result image is stored.
  if (*transformed == NULL) {
	  *transformed = cvCreateImage(cvSize(kCreditCardTargetWidth, kCreditCardTargetHeight), sample->depth, nChannels);
  }
  llcv_unwarp(dmz, sample, src_points, dst_rect, *transformed);
}

void dmz_blur_card(IplImage* cardImageRGB, ScannerState* state, int unblurDigits)
{
    if (unblurDigits < 0) return;
    int blurCount = state->mostRecentUsableHSeg.n_offsets - unblurDigits;
    for (int i = 0; i < state->mostRecentUsableHSeg.n_offsets && i < blurCount ; i++) {
        int num_x = state->mostRecentUsableHSeg.offsets[i] - 1;
        int num_y = state->mostRecentUsableVSeg.y_offset - 1;
        int num_w = state->mostRecentUsableHSeg.number_width + 2;
        int num_h = kNumberHeight + 2;
        if (i < 4) num_h *= 2; // blur smaller four digits below first bucket
        cvSetImageROI(cardImageRGB, cvRect(num_x, num_y, num_w, num_h));
        cv::Mat blurMat = cv::Mat(cardImageRGB, false);
        cv::medianBlur(blurMat, blurMat, 25);
        blurMat.release();
    }
    cvResetImageROI(cardImageRGB);
}

// FOR CYTHON USE ONLY
#if CYTHON_DMZ
void dmz_scharr3_dx_abs(IplImage *src, IplImage *dst) {
  llcv_scharr3_dx_abs(src, dst);
}

// FOR CYTHON USE ONLY
void dmz_scharr3_dy_abs(IplImage *src, IplImage *dst) {
  llcv_scharr3_dy_abs(src, dst);
}

// FOR CYTHON USE ONLY
void dmz_sobel3_dx_dy(IplImage *src, IplImage *dst) {
  llcv_sobel3_dx_dy(src, dst);
}

// FOR CYTHON USE ONLY
ExpiryGroupScores cythonScores_to_ExpiryGroupScores(CythonGroupScores cython_scores) {
  ExpiryGroupScores scores;
  
  for (int character_index = 0; character_index < kExpiryMaxValidLength; character_index++) {
    for (int digit_value = 0; digit_value < 10; digit_value++) {
      scores(character_index, digit_value) = cython_scores[character_index][digit_value];
    }
  }
  
  return scores;
}

// FOR CYTHON USE ONLY
CythonGroupedRects groupedRects_to_CythonGroupedRects(GroupedRectsListIterator group) {
  CythonGroupedRects cython_expiry_group;
  int character_index;
  int digit_value;
  
  cython_expiry_group.top = group->top;
  cython_expiry_group.left = group->left;
  cython_expiry_group.width = group->width;
  cython_expiry_group.height = group->height;
  cython_expiry_group.character_width = group->character_width;
  cython_expiry_group.pattern = group->pattern;

  for (int character_index = 0; character_index < kExpiryMaxValidLength; character_index++) {
    for (int digit_value = 0; digit_value < 10; digit_value++) {
      cython_expiry_group.scores[character_index][digit_value] = group->scores(character_index, digit_value);
    }
  }
  
  cython_expiry_group.recently_seen_count = group->recently_seen_count;
  cython_expiry_group.total_seen_count = group->total_seen_count;
  
  cython_expiry_group.number_of_character_rects = group->character_rects.size();
  size_t character_rects_size = (group->character_rects.size() * sizeof(CythonCharacterRect));
  cython_expiry_group.character_rects = (CythonCharacterRect *) malloc(character_rects_size);
  
  for (character_index = 0; character_index < group->character_rects.size(); ++character_index) {
    cython_expiry_group.character_rects[character_index].top = group->character_rects[character_index].top;
    cython_expiry_group.character_rects[character_index].left = group->character_rects[character_index].left;
  }

  return cython_expiry_group;
}

// FOR CYTHON USE ONLY
GroupedRects cythonGroupedRects_to_GroupedRects(CythonGroupedRects *cython_group) {
  GroupedRects group;
  
  group.top = cython_group->top;
  group.left = cython_group->left;
  group.width = cython_group->width;
  group.height = cython_group->height;
  group.character_width = cython_group->character_width;
  group.pattern = (ExpiryPattern) cython_group->pattern;
  group.scores = cythonScores_to_ExpiryGroupScores(cython_group->scores);
  group.recently_seen_count = cython_group->recently_seen_count;
  group.total_seen_count = cython_group->total_seen_count;
  
  for (int character_index = 0; character_index < cython_group->number_of_character_rects; character_index++) {
    CythonCharacterRect cython_rect = cython_group->character_rects[character_index];
    CharacterRect rect = CharacterRect(cython_rect.top, cython_rect.left, 0);
    group.character_rects.push_back(rect);
  }
  
  return group;
}

// FOR CYTHON USE ONLY
#include "scan/expiry_seg.h"
void dmz_best_expiry_seg(IplImage *card_y, uint16_t starting_y_offset, CythonGroupedRects **cython_expiry_groups, uint16_t *number_of_groups) {
  GroupedRectsList expiry_groups;
  GroupedRectsList name_groups;
  
  best_expiry_seg(card_y, starting_y_offset, expiry_groups, name_groups);

  *cython_expiry_groups = (CythonGroupedRects *) malloc(expiry_groups.size() * sizeof(CythonGroupedRects));
  
  GroupedRectsListIterator group;
  int index;
  for (group = expiry_groups.begin(), index = 0; group != expiry_groups.end(); ++group, ++index) {
    (*cython_expiry_groups)[index] = groupedRects_to_CythonGroupedRects(group);
  }
  
  *number_of_groups = expiry_groups.size();
}

#include "scan/expiry_categorize.h"

// FOR CYTHON USE ONLY
void dmz_expiry_extract(IplImage *card_y,
                        uint16_t *number_of_expiry_groups, CythonGroupedRects **cython_expiry_groups,
                        uint16_t *number_of_new_groups, CythonGroupedRects **cython_new_groups,
                        int *expiry_month, int *expiry_year) {
  GroupedRectsList expiry_groups;
  GroupedRectsList new_groups;
  uint16_t index;
  
  for (index = 0; index < *number_of_expiry_groups; index++) {
    expiry_groups.push_back(cythonGroupedRects_to_GroupedRects(*cython_expiry_groups + index));
  }
  
  for (index = 0; index < *number_of_new_groups; index++) {
    new_groups.push_back(cythonGroupedRects_to_GroupedRects(*cython_new_groups + index));
  }
  
  expiry_extract(card_y, expiry_groups, new_groups, expiry_month, expiry_year);
  
  *number_of_expiry_groups = expiry_groups.size();
  
//  for (index = 0; index < *number_of_expiry_groups; index++) {
//    free((*cython_expiry_groups)[index].character_rects);
//  }
  
  *cython_expiry_groups = (CythonGroupedRects *) realloc(*cython_expiry_groups, expiry_groups.size() * sizeof(CythonGroupedRects));
  
  GroupedRectsListIterator group;
  for (group = expiry_groups.begin(), index = 0; group != expiry_groups.end(); ++group, ++index) {
    (*cython_expiry_groups)[index] = groupedRects_to_CythonGroupedRects(group);
  }
}

void dmz_expiry_extract_group(IplImage *card_y,
                              CythonGroupedRects &cython_group,
                              CythonGroupScores cython_scores,
                              int *expiry_month,
                              int *expiry_year) {
  GroupedRects group = cythonGroupedRects_to_GroupedRects(&cython_group);

  ExpiryGroupScores old_scores = cythonScores_to_ExpiryGroupScores(cython_group.scores);
  
  expiry_extract_group(card_y, group, old_scores, expiry_month, expiry_year);

  for (int character_index = 0; character_index < kExpiryMaxValidLength; character_index++) {
    for (int digit_value = 0; digit_value < 10; digit_value++) {
      cython_scores[character_index][digit_value] = group.scores(character_index, digit_value);
    }
  }
}
#endif

#endif // COMPILE_DMZ
