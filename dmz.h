//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_H
#define DMZ_H

//
// The one true header for the dmz. All publicly accessible APIs live here.
//

#include "dmz_olm.h"

#include "opencv2/core/core_c.h" // needed for IplImage


/******* Types *******/

typedef struct {
  // TODO - add fields that persist over life of a dmz
  void *mz; // Pointer to whatever is needed for your platform's mz implementation
} dmz_context;

typedef struct {
  float rho;
  float theta;
} ParametricLine;

typedef struct {
  int found; // bool indicating whether this edge was detected; if 0, the other values in this struct may contain garbage
  ParametricLine location;
} dmz_found_edge;

typedef struct {
  dmz_found_edge top;
  dmz_found_edge left;
  dmz_found_edge bottom;
  dmz_found_edge right;
} dmz_edges;

/******* Functions *******/


// LIFE CYCLE

// Initialize dmz. Should be called before dmz activity. Returns pointer to dmz-allocated memory.
dmz_context *dmz_context_create(void);

// Clean up and release dmz pointer created by dmz_init. Should be called once, after dmz use is complete.
void dmz_context_destroy(dmz_context *dmz);

// Perform any necessary operations prior to app backgrounding (e.g., calling glFinish() on any OpenGL contexts)
void dmz_prepare_for_backgrounding(dmz_context *dmz);


// CHECKS, UTILITIES & CONVERSIONS

// Check that OpenCV has been successfully compiled and linked in -- just creates an image and releases it.
int dmz_has_opencv(void);

// Deinterleave an interleaved two channel uint8 image into its two component image channels.
// It is the caller's responsibility to free channel1 and channel2.
void dmz_deinterleave_uint8_c2(IplImage *interleaved, IplImage **channel1, IplImage **channel2);

// Deinterleave an interleaved 4-channel RGBA vector into just its R vector (intended for 4-channel RGBA grayscale -> 1-channel grayscale)
void dmz_deinterleave_RGBA_to_R(uint8_t *source, uint8_t *dest, int size);

// Convert a YCbCr image to an RGB image. This will only work if the y, cb, and cr planes
// are the same size (which the output planes of dmz_detect_card are). *rgb MUST be initialized to NULL or
// a valid IplImage. It is the caller's responsibility to free rgb.
void dmz_YCbCr_to_RGB(IplImage *y, IplImage *cb, IplImage *cr, IplImage **rgb);


// DETECTION

float dmz_focus_score(IplImage *image, bool use_full_image);

float dmz_brightness_score(IplImage *image, bool use_full_image);

// Convenience method that returns whether a set of found_edges contains all edges as being found.
bool dmz_found_all_edges(dmz_edges found_edges);

// Detect card edges, and calculate the corner points if all four edges have been detected.
// The boolean return value indicates whether a card was successfully detected.
bool dmz_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,
                                       FrameOrientation orientation, dmz_edges *found_edges, dmz_corner_points *corner_points);


// TRANSFORMATION

// Convert a sample from the camera to a transformed, rectified card image.
// Use corner_points from dmz_detect_edges.
// *transformed MUST be initialized to NULL or a valid IplImage. It is the caller's responsibility
// to free transformed.
void dmz_transform_card(dmz_context *dmz, IplImage *sample, dmz_corner_points corner_points, FrameOrientation orientation, bool upsample, IplImage **transformed);


// FOR CYTHON USE ONLY
#if CYTHON_DMZ
void dmz_scharr3_dx_abs(IplImage *src, IplImage *dst);
void dmz_scharr3_dy_abs(IplImage *src, IplImage *dst);
void dmz_sobel3_dx_dy(IplImage *src, IplImage *dst);

#include "scan/expiry_types.h"
void dmz_best_expiry_seg(IplImage *card_y, uint16_t starting_y_offset, CythonGroupedRects **expiry_groups, uint16_t *number_of_groups);
void dmz_expiry_extract(IplImage *card_y,
                        uint16_t *number_of_expiry_groups, CythonGroupedRects **cython_expiry_groups,
                        uint16_t *number_of_new_groups, CythonGroupedRects **cython_new_groups,
                        int *expiry_month, int *expiry_year);
#endif


#endif // DMZ_H