//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_OLM_H
#define DMZ_OLM_H

//
// The sightless, cave-dwelling portion of the dmz (no OpenCV, no Eigen).
//

#include <stdint.h>
#include "dmz_macros.h"
#include "dmz_constants.h"


/******* Types *******/

typedef uint8_t FrameOrientation;
enum {
    FrameOrientationPortrait = 1, // == UIInterfaceOrientationPortrait
    FrameOrientationPortraitUpsideDown = 2, // == UIInterfaceOrientationPortraitUpsideDown
    FrameOrientationLandscapeRight = 3, // == UIInterfaceOrientationLandscapeRight
    FrameOrientationLandscapeLeft = 4 // == UIInterfaceOrientationLandscapeLeft
};

typedef struct {
  float x;
  float y;
} dmz_point;

typedef struct {
  float x;
  float y;
  float w;
  float h;
} dmz_rect;

typedef struct {
  dmz_point top_left;
  dmz_point bottom_left;
  dmz_point top_right;
  dmz_point bottom_right;
} dmz_corner_points;

typedef uint8_t CardType;
enum {
  CardTypeUnrecognized = 0,
  CardTypeAmbiguous,
  CardTypeAmex,
  CardTypeJCB,
  CardTypeVisa,
  CardTypeMastercard,
  CardTypeDiscover,
  CardTypeMaestro
};

typedef struct {
  CardType card_type;
  int number_length;
  int prefix_length;
  long min_prefix;
  long max_prefix;
} dmz_card_info;

/******* Functions *******/


// POINTS AND RECTS

// Create a dmz_point struct
dmz_point dmz_create_point(float x, float y);

// Create a dmz_rect struct
// TODO - make private?
dmz_rect dmz_create_rect(float x, float y, float w, float h);

// Given a rect, returns top-left, top-right, bottom-left, bottom-right points
void dmz_rect_get_points(dmz_rect rect, dmz_point points[4]);

// Given a source point, a source rect it lies relative to, and a destination rect, 
// return a point scaled to match that destination rect.
// TODO - make private?
dmz_point dmz_scale_point(const dmz_point src_p, const dmz_rect src_f, const dmz_rect dst_f);


// CARD NUMBER VALIDATION AND TYPING

// number_array must contain number_length unsigned ints, each between between 0 and 9
bool dmz_passes_luhn_checksum(uint8_t *number_array, uint8_t number_length);

// number_array must contain number_length unsigned ints, each between between 0 and 9
dmz_card_info dmz_card_info_for_prefix_and_length(uint8_t *number_array,
                                                              uint8_t number_length,
                                                              bool allow_incomplete_number);


// OTHER

// Calculates where in a displayed video preview to draw the guide frame (or anyway, where the card should be placed).
// preview_width and preview_height are the width and height of preview in the ui, not of the actual video frame;
// similarly, the returned rect is in the ui coordinate system.
dmz_rect dmz_guide_frame(FrameOrientation orientation, float preview_width, float preview_height);

// Return the opposite orientation, as in FrameOrientationPortraitUpsideDown -> FrameOrientationPortrait
FrameOrientation dmz_opposite_orientation(FrameOrientation orientation);

#endif // DMZ_OLM_H
