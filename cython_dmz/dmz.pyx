from __future__ import division

import collections
import math
import sys

import cv

import util.opencv_helpers
from util.opencv_helpers import copy_image_from_image, YCrCbImage, ROI
from util.opencv_helpers import Color, LINE_FULLY_CONNECTED, CvRect, draw_rectangle

# Pull in some standard C definitions
from libc.stdint cimport uint16_t, uint8_t, int64_t
from libc.stdlib cimport malloc, free

# Set up the Python/C OpenCV image bridge
cdef extern from "opencv2/core/types_c.h":
    ctypedef struct IplImage

# Pull in some dmz constants...
cdef extern from "dmz.h":
    uint16_t kCreditCardTargetWidth
    uint16_t kCreditCardTargetHeight

    uint16_t kPortraitSampleWidth
    uint16_t kPortraitSampleHeight
    uint16_t kLandscapeSampleWidth
    uint16_t kLandscapeSampleHeight

    uint8_t FrameOrientationPortrait
    uint8_t FrameOrientationPortraitUpsideDown
    uint8_t FrameOrientationLandscapeRight
    uint8_t FrameOrientationLandscapeLeft

cdef extern from "scan/expiry_types.h":
    uint8_t kExpiryMaxValidLength

# ...and give them Python names
CREDIT_CARD_TARGET_WIDTH = kCreditCardTargetWidth
CREDIT_CARD_TARGET_HEIGHT = kCreditCardTargetHeight

PORTRAIT_SAMPLE_WIDTH = kPortraitSampleWidth
PORTRAIT_SAMPLE_HEIGHT = kPortraitSampleHeight

LANDSCAPE_SAMPLE_WIDTH = kLandscapeSampleWidth
LANDSCAPE_SAMPLE_HEIGHT = kLandscapeSampleHeight

FRAME_ORIENTATION_PORTRAIT = FrameOrientationPortrait
FRAME_ORIENTATION_PORTRAIT_UPSIDE_DOWN = FrameOrientationPortraitUpsideDown
FRAME_ORIENTATION_LANDSCAPE_LEFT = FrameOrientationLandscapeRight
FRAME_ORIENTATION_LANDSCAPE_RIGHT = FrameOrientationLandscapeLeft

EXPIRY_MAX_VALID_LENGTH = kExpiryMaxValidLength

NUMBER_PATTERN_UNKNOWN = 0
NUMBER_PATTERN_VISALIKE = 1
NUMBER_PATTERN_AMEXLIKE = 2

# A few more constants

NUMBER_WIDTH = 19
NUMBER_HEIGHT = 27

EXPIRY_DIGIT_WIDTH = 9
EXPIRY_DIGIT_HEIGHT = 15

TRIMMED_CHARACTER_IMAGE_WIDTH = 11
TRIMMED_CHARACTER_IMAGE_HEIGHT = 16

WHITE_16 = (65535, 65535, 65535)
WHITE_32F = (1.0, 1.0, 1.0)

# Globals

# Pull in some dmz structs & functions
cdef extern from "mz.h":
    IplImage *py_mz_create_from_cv_image_data(char *image_data, int image_size,
                                              int width, int height,
                                              int64_t depth, int n_channels,
                                              int roi_x_offset, int roi_y_offset, int roi_width, int roi_height)
    void py_mz_release_ipl_image(IplImage *image)
    void py_mz_get_cv_image_data(IplImage *source,
                                 char **image_data, int *image_size,
                                 int *width, int *height,
                                 int64_t *depth, int *n_channels,
                                 int *roi_x_offset, int *roi_y_offset, int *roi_width, int *roi_height
                                 )
    void py_mz_cvSetImageROI(IplImage* image, int left, int top, int width, int height)
    void py_mz_cvResetImageROI(IplImage* image)


cdef extern from "scan/n_hseg.h":
    ctypedef struct NHorizontalSegmentation:
        uint8_t  n_offsets
        uint16_t offsets[16]
        float    score
        float    number_width
        uint16_t pattern_offset

cdef extern from "scan/n_vseg.h":
    ctypedef uint8_t NumberPatternType

    ctypedef struct NVerticalSegmentation:
        float score
        uint16_t y_offset
        NumberPatternType pattern_type
        uint8_t number_pattern[19]
        uint8_t number_pattern_length
        uint8_t number_length

cdef extern from "scan/frame.h":
    ctypedef struct CythonFrameScanResult:
        NHorizontalSegmentation hseg
        NVerticalSegmentation   vseg
        bint                    usable

    void cython_scan_card_image(IplImage *y, CythonFrameScanResult *result)
    
cdef extern from "scan/expiry_types.h":
    ctypedef struct CythonCharacterRect:
        int   top
        int   left

    ctypedef struct CythonGroupedRects:
        int                 top
        int                 left
        int                 width
        int                 height
        int                 character_width
        uint8_t             pattern
        float               scores[11][10]  # 11 == EXPIRY_MAX_VALID_LENGTH
        int                 recently_seen_count
        int                 total_seen_count
        int                 number_of_character_rects
        CythonCharacterRect *character_rects

cdef extern from "dmz.h":
    ctypedef struct dmz_context:
        void *mz

    ctypedef uint8_t FrameOrientation

    ctypedef struct dmz_point:
        float x
        float y

    ctypedef struct dmz_rect:
        float x
        float y
        float w
        float h

    ctypedef struct dmz_corner_points:
        dmz_point top_left
        dmz_point bottom_left
        dmz_point top_right
        dmz_point bottom_right

    ctypedef struct dmz_ParametricLine:
        float rho
        float theta

    ctypedef struct dmz_found_edge:
        int found  # bool indicating whether this edge was detected; if 0, the other values in this struct may contain garbage
        dmz_ParametricLine location

    ctypedef struct dmz_edges:
        dmz_found_edge top
        dmz_found_edge left
        dmz_found_edge bottom
        dmz_found_edge right

    float dmz_focus_score(IplImage *image, bint use_full_image)
    float dmz_brightness_score(IplImage *image, bint use_full_image)

    void dmz_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,
                          FrameOrientation orientation,
                          dmz_edges *found_edges, dmz_corner_points *corner_points)

    void dmz_transform_card(dmz_context *dmz, IplImage *sample, dmz_corner_points corner_points,
                            FrameOrientation orientation, bint upsample, IplImage **transformed)

    void dmz_scharr3_dx_abs(IplImage *src, IplImage *dst)
    void dmz_scharr3_dy_abs(IplImage *src, IplImage *dst)

    void dmz_sobel3_dx_dy(IplImage *src, IplImage *dst)
    
    void dmz_best_expiry_seg(IplImage *card_y, uint16_t starting_y_offset, CythonGroupedRects **expiry_groups, uint16_t *number_of_groups)

    void dmz_expiry_extract(IplImage *card_y,
                            uint16_t *number_of_expiry_groups, CythonGroupedRects **cython_expiry_groups,
                            uint16_t *number_of_new_groups, CythonGroupedRects **cython_new_groups,
                            int *expiry_month, int *expiry_year)

    void dmz_expiry_extract_group(IplImage *card_y,
                                  CythonGroupedRects &cython_group,
                                  float cython_scores[11][10],
                                  int *expiry_month,
                                  int *expiry_year)
    

# OpenCV IplImage <-> PIL cv_image conversions

cdef IplImage *to_ipl_image(object cv_image):
    """The return value from this call must be freed with release_ipl_image() when it is no longer directly needed."""
    cv_image_data = cv_image.tostring()
    cdef char *image_data = cv_image_data  # convert python string to c string
    cdef int width = cv_image.width
    cdef int height = cv_image.height
    cdef int64_t depth = cv_image.depth
    cdef int n_channels = cv_image.nChannels
    cdef int roi_x_offset, roi_y_offset, roi_width, roi_height
    roi_x_offset, roi_y_offset, roi_width, roi_height = cv.GetImageROI(cv_image)
    cdef IplImage *ipl_image = py_mz_create_from_cv_image_data(image_data, len(image_data),
                                                               width, height,
                                                               depth, n_channels,
                                                               roi_x_offset, roi_y_offset, roi_width, roi_height)
    return ipl_image

cdef release_ipl_image(IplImage *ipl_image):
    """Releases IplImage created by to_ipl_image()"""
    py_mz_release_ipl_image(ipl_image)

cdef object to_cv_image(IplImage *ipl_image):
    """Helper cdef function to convert from OpenCV's C image to Python image."""
    cdef char *image_data
    cdef int image_size
    cdef int width
    cdef int height
    cdef int64_t depth
    cdef int n_channels
    cdef int roi_x_offset
    cdef int roi_y_offset
    cdef int roi_width
    cdef int roi_height
    py_mz_get_cv_image_data(ipl_image,
                            &image_data, &image_size,
                            &width, &height,
                            &depth, &n_channels,
                            &roi_x_offset, &roi_y_offset, &roi_width, &roi_height)
    cdef image_bytes
    cdef Py_ssize_t images_bytes_length = image_size
    image_bytes = image_data[:images_bytes_length]

    cv_image = cv.CreateImageHeader((width, height), depth, n_channels)
    cv.SetData(cv_image, image_bytes)
    cv.SetImageROI(cv_image, (roi_x_offset, roi_y_offset, roi_width, roi_height))

    return cv_image


def echo_image(cv_image):
    """Simple example and test hook for Python/C/Python OpenCV image conversions."""
    # convert to C image
    cdef IplImage *ipl_image = to_ipl_image(cv_image)  # convert Python image to C image
    # convert back to Python image
    python_image = to_cv_image(ipl_image)
    # don't leak memory
    release_ipl_image(ipl_image)
    return python_image


# dmz C structs --> Python classes


class Point(collections.namedtuple("Point", "x, y")):

    def __new__(cls, x=0, y=0):
        return super(Point, cls).__new__(cls, int(x), int(y))


cdef dmz_point point_to_dmz_point(py_point):
    cdef dmz_point point
    point.x = py_point.x
    point.y = py_point.y
    return point


class Rect(collections.namedtuple("Rect", "x, y, w, h")):

    def __new__(cls, x=0, y=0, w=0, h=0):
        return super(Rect, cls).__new__(cls, int(x), int(y), int(w), int(h))


class CornerPoints(object):

    def __init__(self, top_left=None, bottom_left=None, top_right=None, bottom_right=None):
        self.top_left = Point() if top_left is None else top_left
        self.top_right = Point() if top_right is None else top_right
        self.bottom_left = Point() if bottom_left is None else bottom_left
        self.bottom_right = Point() if bottom_right is None else bottom_right

    def __repr__(self):
        return '<top_left={},\n top_right={},\n bottom_left={},\n bottom_right={}>'.format(self.top_left, self.top_right, self.bottom_left, self.bottom_right)


cdef dmz_corner_points corner_points_to_dmz_corner_points(py_corner_points):
    cdef dmz_corner_points points
    points.top_left = point_to_dmz_point(py_corner_points.top_left)
    points.top_right = point_to_dmz_point(py_corner_points.top_right)
    points.bottom_left = point_to_dmz_point(py_corner_points.bottom_left)
    points.bottom_right = point_to_dmz_point(py_corner_points.bottom_right)
    return points


class ParametricLine(util.opencv_helpers.ParametricLine):

    def __new__(cls, rho=0, theta=0):
        return super(ParametricLine, cls).__new__(cls, rho, theta)


class FoundEdge(object):

    def __init__(self, found=0, parametric_line=None):
        self.found = found
        self.parametric_line = ParametricLine() if parametric_line is None else parametric_line

    def __repr__(self):
        if self.found:
            return '{}'.format(repr(self.parametric_line))
        else:
            return 'not found'


class Edges(object):

    def __init__(self, top=None, left=None, bottom=None, right=None):
        self.top = FoundEdge() if top is None else top
        self.left = FoundEdge() if left is None else left
        self.bottom = FoundEdge() if bottom is None else bottom
        self.right = FoundEdge() if right is None else right

    def __repr__(self):
        return '<top: {},\n left: {},\n bottom: {},\n right: {}>'.format(repr(self.top), repr(self.left), repr(self.bottom), repr(self.right))

    def found_all(self):
        return all((self.top.found, self.left.found, self.bottom.found, self.right.found))


class VerticalSegmentation(collections.namedtuple("VerticalSegmentation",
   "score, y_offset, pattern_type, number_pattern, number_pattern_length, number_length")):

    def __init__(self, score=0, y_offset=0, pattern_type=NUMBER_PATTERN_UNKNOWN,
                 number_pattern=[], number_pattern_length=0, number_length=0):
        pass

    def __repr__(self):
        return "VSeg: score={}\n      y_offset={}\n      pattern_type={}\n      number_pattern={}\n      number_pattern_length={}\n      number_length={}".format(self.score, self.y_offset, self.pattern_type, self.number_pattern, self.number_pattern_length, self.number_length)


class HorizontalSegmentation(collections.namedtuple("HorizontalSegmentation",
   "n_offsets, offsets, score, number_width, pattern_offset")):

    def __init__(self, n_offsets=0, offsets=[], score=0, number_width=0, pattern_offset=0):
        pass

    def __repr__(self):
        return "HSeg: n_offsets={}\n      offsets={}\n      score={}\n      number_width={}\n      pattern_offset={}".format(self.n_offsets, self.offsets, self.score, self.number_width, self.pattern_offset)


# dmz C functions --> Python functions


def focus_score(cv_image, use_full_image):
    cdef IplImage *ipl_image = to_ipl_image(cv_image)  # convert Python image to C image
    cdef float score = dmz_focus_score(ipl_image, use_full_image)
    release_ipl_image(ipl_image)
    return score


def brightness_score(cv_image, use_full_image):
    cdef IplImage *ipl_image = to_ipl_image(cv_image)  # convert Python image to C image
    cdef float score = dmz_brightness_score(ipl_image, use_full_image)
    release_ipl_image(ipl_image)
    return score


def detect_edges(yCrCb_image, orientation):
    cdef dmz_edges dmz_found_edges
    cdef dmz_corner_points dmz_corner_points
    cdef IplImage *y_ipl_image = to_ipl_image(yCrCb_image.y)
    cdef IplImage *cr_ipl_image = to_ipl_image(yCrCb_image.cr)
    cdef IplImage *cb_ipl_image = to_ipl_image(yCrCb_image.cb)

    dmz_detect_edges(y_ipl_image, cb_ipl_image, cr_ipl_image, orientation, &dmz_found_edges, &dmz_corner_points)

    release_ipl_image(y_ipl_image)
    release_ipl_image(cr_ipl_image)
    release_ipl_image(cb_ipl_image)

    found_edges = Edges(top=FoundEdge(dmz_found_edges.top.found,
                                      ParametricLine(dmz_found_edges.top.location.rho, dmz_found_edges.top.location.theta)),
                        left=FoundEdge(dmz_found_edges.left.found,
                                       ParametricLine(dmz_found_edges.left.location.rho, dmz_found_edges.left.location.theta)),
                        bottom=FoundEdge(dmz_found_edges.bottom.found,
                                         ParametricLine(dmz_found_edges.bottom.location.rho, dmz_found_edges.bottom.location.theta)),
                        right=FoundEdge(dmz_found_edges.right.found,
                                        ParametricLine(dmz_found_edges.right.location.rho, dmz_found_edges.right.location.theta))
                        )

    corner_points = CornerPoints(top_left=Point(dmz_corner_points.top_left.x, dmz_corner_points.top_left.y),
                                 top_right=Point(dmz_corner_points.top_right.x, dmz_corner_points.top_right.y),
                                 bottom_left=Point(dmz_corner_points.bottom_left.x, dmz_corner_points.bottom_left.y),
                                 bottom_right=Point(dmz_corner_points.bottom_right.x, dmz_corner_points.bottom_right.y)
                                 )

    return found_edges, corner_points


def transform_card(yCrCb_image, corner_points, orientation):
    cdef IplImage *y_ipl_image = to_ipl_image(yCrCb_image.y)
    cdef IplImage *card_y_ipl_image = NULL
    dmz_transform_card(<dmz_context *>NULL, y_ipl_image, corner_points_to_dmz_corner_points(corner_points), orientation, 0, &card_y_ipl_image)

    cdef IplImage *cr_ipl_image = to_ipl_image(yCrCb_image.cr)
    cdef IplImage *card_cr_ipl_image = NULL
    dmz_transform_card(<dmz_context *>NULL, cr_ipl_image, corner_points_to_dmz_corner_points(corner_points), orientation, 1, &card_cr_ipl_image)

    cdef IplImage *cb_ipl_image = to_ipl_image(yCrCb_image.cb)
    cdef IplImage *card_cb_ipl_image = NULL
    dmz_transform_card(<dmz_context *>NULL, cb_ipl_image, corner_points_to_dmz_corner_points(corner_points), orientation, 1, &card_cb_ipl_image)

    transformed_image = YCrCbImage(y=to_cv_image(card_y_ipl_image), cr=to_cv_image(card_cr_ipl_image), cb=to_cv_image(card_cb_ipl_image))

    release_ipl_image(y_ipl_image)
    release_ipl_image(cr_ipl_image)
    release_ipl_image(cb_ipl_image)
    release_ipl_image(card_y_ipl_image)
    release_ipl_image(card_cr_ipl_image)
    release_ipl_image(card_cb_ipl_image)

    return transformed_image.as_rgb(), transformed_image.y


class CharacterRect(collections.namedtuple("CharacterRect", "top, left, sum")):

    def __new__(cls, top=0, left=0, sum=0):
        return super(CharacterRect, cls).__new__(cls, int(top), int(left), int(sum))


class GroupedRects(object):

    def __init__(self, left=0, top=0, width=0, height=0, grouped_yet=False, sum=0, character_rects=[], character_width=EXPIRY_DIGIT_WIDTH, pattern=0, scores=None, recently_seen_count=0, total_seen_count=0):
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self.grouped_yet = bool(grouped_yet)
        self.sum = int(sum)
        self.character_rects = [] if character_rects is None else character_rects
        self.character_width = int(character_width)
        self.pattern = int(pattern)
        self.scores = [[0.0,] * 10] * EXPIRY_MAX_VALID_LENGTH if scores is None else scores
        self.recently_seen_count = int(recently_seen_count)
        self.total_seen_count = int(total_seen_count)

    def __repr__(self):
        return '<left: {},\n top: {},\n width: {},\n height: {},\n grouped_yet: {},\n sum: {},\n character_rects: {},\n character_width: {},\n pattern: {}\n scores: {}>'.format(repr(self.top), repr(self.left), repr(self.width), repr(self.height), repr(self.grouped_yet), repr(self.sum), repr(self.character_rects), repr(self.character_width), repr(self.pattern), repr(self.scores))


def scan_card(y_image):
    cdef CythonFrameScanResult result
    cdef IplImage *y_ipl_image = to_ipl_image(y_image)
    cython_scan_card_image(y_ipl_image, &result)
    release_ipl_image(y_ipl_image)

    vertical_segmentation = VerticalSegmentation(score=result.vseg.score,
                                                 y_offset=result.vseg.y_offset,
                                                 pattern_type=result.vseg.pattern_type,
                                                 number_pattern=[result.vseg.number_pattern[i] for i in range(19)],
                                                 number_pattern_length=result.vseg.number_pattern_length,
                                                 number_length=result.vseg.number_length
                                                 )

    horizontal_segmentation = HorizontalSegmentation(n_offsets=result.hseg.n_offsets,
                                                     offsets=[result.hseg.offsets[i] for i in range(16)],
                                                     score=result.hseg.score,
                                                     number_width=result.hseg.number_width,
                                                     pattern_offset=result.hseg.pattern_offset
                                                     )

    return result.usable, vertical_segmentation, horizontal_segmentation


def expiration_segment(card_image_YCrCb, y_offset, doing_segmentation):
    if card_image_YCrCb is None:
        if doing_segmentation:
            return []

        sobel_image = card_image_annotated = probable_lines_for_display = create_card_mask(cv.IPL_DEPTH_8U)
    else:
        image_width, image_height = cv.GetSize(card_image_YCrCb)
        
        if doing_segmentation:
            card_image_y = card_image_YCrCb
        else:
            card_image_y = create_card_mask(card_image_YCrCb.depth)
            cv.Split(card_image_YCrCb, card_image_y, None, None, None)

        # Look for vertical line segments -> sobel_image
        
        sobel_image = scharr_dx_and_abs(card_image_y, y_offset)
        
        # Calculate relative probability for each line
        
        below_numbers_rect = CvRect(0, y_offset + NUMBER_HEIGHT, CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT - (y_offset + NUMBER_HEIGHT))
        first_stripe_base_row = below_numbers_rect.y + 1  # the "+ 1" represents the tolerance above and below each stripe
        last_stripe_base_row = image_height - 2 * EXPIRY_DIGIT_HEIGHT  # exclude a bottom margin of EXPIRY_DIGIT_HEIGHT
        left_edge = EXPIRY_DIGIT_WIDTH * 3  # there aren't usually any actual characters this far to the left
        right_edge = (image_width * 2) // 3  # beyond here lie logos
        line_sum = []
  
        for row in range(first_stripe_base_row - 1):
            line_sum.append(0)
            
        for row in range(first_stripe_base_row - 1, image_height):
            cv.SetImageROI(sobel_image, (left_edge, row, right_edge - left_edge, 1))
            line_sum.append(int(cv.Sum(sobel_image)[0]))
  
        cv.ResetImageROI(sobel_image)
        
        probable_lines_for_display = create_card_mask(cv.IPL_DEPTH_32F)
        
        for row in range(y_offset + NUMBER_HEIGHT + EXPIRY_DIGIT_HEIGHT, CREDIT_CARD_TARGET_HEIGHT - EXPIRY_DIGIT_HEIGHT):
            cv.SetImageROI(probable_lines_for_display, (0, row, CREDIT_CARD_TARGET_WIDTH, 1))
            cv.Set(probable_lines_for_display, cv.ScalarAll(line_sum[row]))

        cv.ResetImageROI(probable_lines_for_display)
        
        # Determine the 2-3 most probable, non-overlapping stripes
        # (Two will usually get us expiry and name, but one gift-card in the collection has a third line indicating total $$.)
        number_of_stripes = 3  #5 if doing_segmentation else 3
        
        stripe_sums = []  # array of (stripe_sum, base_row)
        for base_row in range(first_stripe_base_row, last_stripe_base_row):
            stripe_sum = 0.0
            for row in range(base_row, base_row + EXPIRY_DIGIT_HEIGHT):
                stripe_sum += line_sum[row]

            threshold = max([line_sum[row] for row in range(base_row, base_row + EXPIRY_DIGIT_HEIGHT)]) / 2

            # Eliminate stripes that have a a much dimmer-than-average sub-stripe
            # at their very top or very bottom.
            if line_sum[base_row] + line_sum[base_row + 1] < threshold:
                continue
            if line_sum[base_row + EXPIRY_DIGIT_HEIGHT - 2] + line_sum[base_row + EXPIRY_DIGIT_HEIGHT - 1] < threshold:
                continue

            # Eliminate stripes that contain a much dimmer-than-average sub-stripe,
            # since that usually means that we've been fooled into grabbing the bottom
            # of some card feature and the top of a different card feature.
            for row in range(base_row, base_row + EXPIRY_DIGIT_HEIGHT - 3):
                if line_sum[row + 1] < threshold and line_sum[row + 2] < threshold:
                    break
            else:
                # Okay, it's a reasonable candidate stripe. Save it.
                stripe_sums.append((stripe_sum, base_row))
    
        stripe_sums = reversed(sorted(stripe_sums, key=lambda stripe: stripe[0]))
        
        probable_stripes = []  # array of (stripe_sum, base_row)
        for stripe_sum in stripe_sums:
            for probable_stripe in probable_stripes:
                if probable_stripe[1] - EXPIRY_DIGIT_HEIGHT < stripe_sum[1] < probable_stripe[1] + EXPIRY_DIGIT_HEIGHT:
                    break
            else:
                probable_stripes.append(stripe_sum)
                if len(probable_stripes) >= number_of_stripes:
                    break
        
        for (index, stripe_base_row) in enumerate(probable_stripes):
            draw_rectangle(probable_lines_for_display, (100 * (index + 1), stripe_base_row[1], 50, EXPIRY_DIGIT_HEIGHT), cv.CV_FILLED, Color.BLACK)
        
        # Find the character groups
        
        expiry_groups = []
        name_groups = []
        for probable_stripe in probable_stripes:
            expiry_groups_for_stripe, name_groups_for_stripe = find_character_groups(sobel_image, probable_stripe[1], probable_stripe[0], y_offset, doing_segmentation)
            expiry_groups.extend(expiry_groups_for_stripe)
            name_groups.extend(name_groups_for_stripe)
        
        if doing_segmentation:
            return expiry_groups
        
        # Display the results
    
        card_groups_mask = create_card_mask(cv.IPL_DEPTH_8U)
        cv.SetZero(card_groups_mask)
    
        card_image_annotated_1 = cv.CreateImage((CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT), cv.IPL_DEPTH_32F, 1)
        cv.ConvertScale(sobel_image, card_image_annotated_1, 1, 0)
        cv.Normalize(card_image_annotated_1, card_image_annotated_1, 1.0, 0, cv.CV_C)
    
        normalized_rects = []  # (sobel_image_rect, card_image_y_rect)
        for group in expiry_groups:
            for rect in group.character_rects:
                left = rect.left
                top = rect.top
                width = TRIMMED_CHARACTER_IMAGE_WIDTH
                height = TRIMMED_CHARACTER_IMAGE_HEIGHT

                cv.SetImageROI(card_groups_mask, (left, top, width, height))
                cv.Set(card_groups_mask, cv.ScalarAll(255))
            
                cv.SetImageROI(sobel_image, (left, top, width, height))
                normalized_rect = cv.CreateImage((width, height), cv.IPL_DEPTH_32F, 1)
                cv.Normalize(sobel_image, normalized_rect, 1.0, 0, cv.CV_C)

                y_rect = cv.CreateImage((width, height), cv.IPL_DEPTH_32F, 1)
                cv.SetImageROI(card_image_y, (left, top, width, height))
                cv.ConvertScale(card_image_y, y_rect, 1.0/255.0, 0)
            
                normalized_rects.append((normalized_rect, y_rect))
    
        cv.ResetImageROI(card_groups_mask)
        cv.ResetImageROI(sobel_image)
        cv.ResetImageROI(card_image_y)
    
        card_image_annotated = cv.CreateImage((CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT), cv.IPL_DEPTH_32F, 1)
        cv.SetZero(card_image_annotated)
        cv.Copy(card_image_annotated_1, card_image_annotated, card_groups_mask)
    
        RECT_OUTSET = 2
    
        for group in expiry_groups:
            draw_rectangle(card_image_annotated, (group.left - RECT_OUTSET, group.top - RECT_OUTSET, group.width + 2 * RECT_OUTSET, group.height + 2 * RECT_OUTSET), 1, 1.0)

        cv.SetImageROI(card_image_annotated, (0, 0, CREDIT_CARD_TARGET_WIDTH, y_offset + NUMBER_HEIGHT))
        cv.Set(card_image_annotated, cv.ScalarAll(0.8))
        cv.ResetImageROI(card_image_annotated)

        rect_x = 10
        rect_y = 10
        rect_row = 0
        ROW_VERTICAL_OFFSET = EXPIRY_DIGIT_HEIGHT + 2 * RECT_OUTSET + 8
        NUMBER_OF_ROWS = 3
        for rect in normalized_rects:
            width, height = cv.GetSize(rect[0])
            cv.SetImageROI(card_image_annotated, (rect_x, rect_y, width, height))
            cv.Copy(rect[0], card_image_annotated)
        
            cv.SetImageROI(card_image_annotated, (rect_x, rect_y + NUMBER_OF_ROWS * ROW_VERTICAL_OFFSET, width, height))
            cv.Copy(rect[1], card_image_annotated)
        
            rect_x += EXPIRY_DIGIT_WIDTH + 2 * RECT_OUTSET + 8
            if rect_row + 1 == NUMBER_OF_ROWS:
                break
            if rect_x + EXPIRY_DIGIT_WIDTH + 2 * RECT_OUTSET > CREDIT_CARD_TARGET_WIDTH - 10:
                rect_x = 10
                rect_y += ROW_VERTICAL_OFFSET
                rect_row += 1

        cv.ResetImageROI(card_image_annotated)
    
    # Normalize images for display
    
    cv.Normalize(probable_lines_for_display, probable_lines_for_display, 1.0, 0, cv.CV_C)
    
    sobel_image_32 = cv.CreateImage((CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT), cv.IPL_DEPTH_32F, 1)
    cv.ConvertScale(sobel_image, sobel_image_32, 1, 0)
    cv.Normalize(sobel_image_32, sobel_image_32, 1.0, 0, cv.CV_C)
    
    return collections.OrderedDict([('Probable', probable_lines_for_display),
                                    ('Sobel 3-Scharr', sobel_image_32),
                                    ('Annotated', card_image_annotated),
                                    ])


def dmz_expiration_segment(card_image_YCrCb, y_offset, doing_segmentation):
    if not doing_segmentation:
        return expiration_segment(card_image_YCrCb, y_offset, doing_segmentation)
    if card_image_YCrCb is None:
        return []
    
    cdef IplImage *ipl_image = to_ipl_image(card_image_YCrCb)  # convert Python image to C image
    cdef CythonGroupedRects *cython_expiry_groups = NULL
    cdef CythonGroupedRects cython_group
    cdef CythonCharacterRect cython_rect
    cdef uint16_t number_of_groups

    dmz_best_expiry_seg(ipl_image, y_offset, &cython_expiry_groups, &number_of_groups)
    
    release_ipl_image(ipl_image)

    expiry_groups = []
    for index in range(0, number_of_groups):
        cython_group = cython_expiry_groups[index]
        character_rects = []
        for character_rect_index in range(0, cython_group.number_of_character_rects):
            cython_rect = cython_group.character_rects[character_rect_index]
            character_rect = CharacterRect(cython_rect.top, cython_rect.left, 0)
            character_rects.append(character_rect)
        
        group = GroupedRects(cython_group.left, cython_group.top, cython_group.width, cython_group.height, True, 0, character_rects)
        expiry_groups.append(group)
        
        free(cython_group.character_rects)
    free(cython_expiry_groups)

    return expiry_groups


cdef to_group_scores(float cython_scores[11][10]):
    scores = []
    for character_index in range(0, EXPIRY_MAX_VALID_LENGTH):
        values = []
        for digit_value in range(0, 10):
            values.append(cython_scores[character_index][digit_value])
        scores.append(values)
    return scores


cdef CythonGroupedRects to_cython_grouped_rects(group):
    cdef CythonGroupedRects cython_group
    
    cython_group.top = group.top
    cython_group.left = group.left
    cython_group.width = group.width
    cython_group.height = group.height
    cython_group.character_width = group.character_width
    cython_group.pattern = group.pattern
    
    for character_index in range(0, EXPIRY_MAX_VALID_LENGTH):
        for digit_value in range(0, 10):
            cython_group.scores[character_index][digit_value] = group.scores[character_index][digit_value]
    
    cython_group.recently_seen_count = group.recently_seen_count
    cython_group.total_seen_count = group.total_seen_count

    cython_group.number_of_character_rects = len(group.character_rects)
    character_rects_size = (len(group.character_rects) * sizeof(CythonCharacterRect))
    cython_group.character_rects = <CythonCharacterRect *> malloc(character_rects_size)

    for character_index in range(0, len(group.character_rects)):
        cython_group.character_rects[character_index].top = group.character_rects[character_index].top
        cython_group.character_rects[character_index].left = group.character_rects[character_index].left
    
    return cython_group


cdef to_grouped_rects(CythonGroupedRects cython_group):
    cdef CythonCharacterRect cython_rect
    
    group = GroupedRects(cython_group.left,
                         cython_group.top,
                         cython_group.width,
                         cython_group.height,
                         True,
                         0,
                         None,
                         cython_group.character_width,
                         cython_group.pattern,
                         to_group_scores(cython_group.scores),
                         cython_group.recently_seen_count,
                         cython_group.total_seen_count)

    for character_index in range(0, cython_group.number_of_character_rects):
        cython_rect = cython_group.character_rects[character_index]
        group.character_rects.append(CharacterRect(cython_rect.top, cython_rect.left, 0))

    return group


def dmz_expiry_categorize(card_image_Y, categorized_expiry_groups, candidate_expiry_groups):
    cdef IplImage *ipl_image = to_ipl_image(card_image_Y)  # convert Python image to C image
    cdef uint16_t number_of_categorized_expiry_groups = len(categorized_expiry_groups)
    cdef uint16_t number_of_candidate_expiry_groups = len(candidate_expiry_groups)
    cdef CythonGroupedRects *cython_categorized_expiry_groups = <CythonGroupedRects *> malloc(number_of_categorized_expiry_groups * sizeof(CythonGroupedRects))
    cdef CythonGroupedRects *cython_candidate_expiry_groups = <CythonGroupedRects *> malloc(number_of_candidate_expiry_groups * sizeof(CythonGroupedRects))
    cdef int expiry_month = 0
    cdef int expiry_year = 0

    for index, group in enumerate(categorized_expiry_groups):
        cython_categorized_expiry_groups[index] =  to_cython_grouped_rects(group)

    for index, group in enumerate(candidate_expiry_groups):
        cython_candidate_expiry_groups[index] =  to_cython_grouped_rects(group)
    
    dmz_expiry_extract(ipl_image,
                       &number_of_categorized_expiry_groups, &cython_categorized_expiry_groups,
                       &number_of_candidate_expiry_groups, &cython_candidate_expiry_groups,
                       &expiry_month, &expiry_year)
    
    release_ipl_image(ipl_image)
    
    categorized_expiry_groups = []
    for index in range(0, number_of_categorized_expiry_groups):
        group = to_grouped_rects(cython_categorized_expiry_groups[index])
        categorized_expiry_groups.append(group)
    
    free(cython_categorized_expiry_groups)
    free(cython_candidate_expiry_groups)
    
    return (expiry_month, expiry_year, categorized_expiry_groups)


def dmz_expiry_categorize_group(card_image_Y, group, old_scores):
    cdef IplImage *ipl_image = to_ipl_image(card_image_Y)  # convert Python image to C image
    cdef CythonGroupedRects cython_group = to_cython_grouped_rects(group)
    
    cdef float[11][10] cython_scores
    for character_index in range(0, EXPIRY_MAX_VALID_LENGTH):
        for digit_value in range(0, 10):
            cython_scores[character_index][digit_value] = old_scores[character_index][digit_value] if old_scores else 0.0
    
    cdef int expiry_month = 0
    cdef int expiry_year = 0
    
    dmz_expiry_extract_group(ipl_image,
                             cython_group,
                             cython_scores,
                             &expiry_month, &expiry_year)
    
    release_ipl_image(ipl_image)
    
    scores = to_group_scores(cython_scores)

    return (scores, expiry_month, expiry_year)


def create_card_mask(depth):
    mask = cv.CreateImage((CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT), depth, 1)
    cv.SetZero(mask)
    return mask


def scharr_dx_and_abs(image_y, y_offset = None):
    sobel_image = cv.CreateImage(cv.GetSize(image_y), cv.IPL_DEPTH_16S, 1)
    cv.SetZero(sobel_image)

    cdef IplImage *ipl_image = to_ipl_image(image_y)  # convert Python image to C image
    cdef IplImage *ipl_image_sobel = to_ipl_image(sobel_image)  # convert Python image to C image

    if y_offset is not None:
        below_numbers_rect = CvRect(0, y_offset + NUMBER_HEIGHT, CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT - (y_offset + NUMBER_HEIGHT))
        py_mz_cvSetImageROI(ipl_image, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
        py_mz_cvSetImageROI(ipl_image_sobel, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
    
    dmz_scharr3_dx_abs(ipl_image, ipl_image_sobel)

    if y_offset is not None:
        py_mz_cvResetImageROI(ipl_image)
        py_mz_cvResetImageROI(ipl_image_sobel)

    sobel_image = to_cv_image(ipl_image_sobel)

    release_ipl_image(ipl_image)
    release_ipl_image(ipl_image_sobel)

    return sobel_image


def scharr_dy_and_abs(image_y, y_offset = None):
    sobel_image = cv.CreateImage(cv.GetSize(image_y), cv.IPL_DEPTH_16S, 1)
    cv.SetZero(sobel_image)

    cdef IplImage *ipl_image = to_ipl_image(image_y)  # convert Python image to C image
    cdef IplImage *ipl_image_sobel = to_ipl_image(sobel_image)  # convert Python image to C image

    if y_offset is not None:
        below_numbers_rect = CvRect(0, y_offset + NUMBER_HEIGHT, CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT - (y_offset + NUMBER_HEIGHT))
        py_mz_cvSetImageROI(ipl_image, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
        py_mz_cvSetImageROI(ipl_image_sobel, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
    
    dmz_scharr3_dy_abs(ipl_image, ipl_image_sobel)

    if y_offset is not None:
        py_mz_cvResetImageROI(ipl_image)
        py_mz_cvResetImageROI(ipl_image_sobel)

    sobel_image = to_cv_image(ipl_image_sobel)

    release_ipl_image(ipl_image)
    release_ipl_image(ipl_image_sobel)

    return sobel_image


def sobel_dx_dy_and_abs(image_y, y_offset):
    sobel_image = cv.CreateImage(cv.GetSize(image_y), cv.IPL_DEPTH_16S, 1)
    cv.SetZero(sobel_image)

    cdef IplImage *ipl_image = to_ipl_image(image_y)  # convert Python image to C image
    cdef IplImage *ipl_image_sobel = to_ipl_image(sobel_image)  # convert Python image to C image

    if y_offset > 0:
        below_numbers_rect = CvRect(0, y_offset + NUMBER_HEIGHT, CREDIT_CARD_TARGET_WIDTH, CREDIT_CARD_TARGET_HEIGHT - (y_offset + NUMBER_HEIGHT))
        py_mz_cvSetImageROI(ipl_image, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
        py_mz_cvSetImageROI(ipl_image_sobel, below_numbers_rect.x, below_numbers_rect.y, below_numbers_rect.w, below_numbers_rect.h)
    
    dmz_sobel3_dx_dy(ipl_image, ipl_image_sobel)

    if y_offset > 0:
        py_mz_cvResetImageROI(ipl_image)
        py_mz_cvResetImageROI(ipl_image_sobel)

    sobel_image = to_cv_image(ipl_image_sobel)

    release_ipl_image(ipl_image)
    release_ipl_image(ipl_image_sobel)

    cv.Abs(sobel_image, sobel_image)

    return sobel_image


def find_character_groups(sobel_image, stripe_base_row, stripe_sum, y_offset, doing_segmentation):
    image_width, image_height = cv.GetSize(sobel_image)
    expanded_stripe_top = stripe_base_row - 1
    expanded_stripe_rect = CvRect(0, expanded_stripe_top, image_width, min(EXPIRY_DIGIT_HEIGHT + 2, image_height - expanded_stripe_top))

    cv.SetImageROI(sobel_image, expanded_stripe_rect)

    # Calculate sum for each possible digit rectangle

    rect_sums = cv.CreateImage((expanded_stripe_rect.w, expanded_stripe_rect.h), cv.IPL_DEPTH_16S, 1)
    rect_kernel = cv.CreateMat(expanded_stripe_rect.h, EXPIRY_DIGIT_WIDTH, cv.CV_8UC1)
    cv.Set(rect_kernel, cv.ScalarAll(1))
    cv.Filter2D(sobel_image, rect_sums, rect_kernel, (0,0))

    cv.ResetImageROI(sobel_image)

    # Collect rectangles into a list
    
    RECTANGLE_AVERAGE_THRESHOLD_FACTOR = 5
    rect_average_based_on_stripe_sum = ((stripe_sum * EXPIRY_DIGIT_WIDTH) / image_width)
    rectangle_summation_threshold = rect_average_based_on_stripe_sum / RECTANGLE_AVERAGE_THRESHOLD_FACTOR
   
    rect_list = []  # CharacterRect
    rect_sum_total = 0
    
    for col in range(expanded_stripe_rect.x, expanded_stripe_rect.x + expanded_stripe_rect.w - EXPIRY_DIGIT_WIDTH):
        sum = cv.Get2D(rect_sums, 0, col)[0]
        if sum > rectangle_summation_threshold:
            rect_list.append(CharacterRect(expanded_stripe_rect.y, col, sum))
            rect_sum_total += sum

    if len(rect_list) == 0:
        return [], []

    rect_sum_average = rect_sum_total / len(rect_list)
    RECT_SUM_THRESHOLD_FACTOR  = 0.8
    rect_sum_threshold = RECT_SUM_THRESHOLD_FACTOR * rect_sum_average
    
    # Sort rectangles by sum (in descending order)

    rect_list = reversed(sorted(rect_list, key=lambda rect: rect.sum))
    
    # Find the non-overlapping rectangles, ignoring rectangles whose sum is excessively small (compared to the average rect sum)
    
    non_overlapping_rect_list = []  # GroupedRects
    non_overlapping_rect_mask = []
    
    for rect in rect_list:
        if rect.sum <= rect_sum_threshold:
            break
        if not rect.left in non_overlapping_rect_mask and not (rect.left + EXPIRY_DIGIT_WIDTH - 1) in non_overlapping_rect_mask:
            group = GroupedRects(rect.left, rect.top, EXPIRY_DIGIT_WIDTH, expanded_stripe_rect.h, False, rect.sum, [], EXPIRY_DIGIT_WIDTH)
            non_overlapping_rect_list.append(group)
            for col in range(EXPIRY_DIGIT_WIDTH):
                non_overlapping_rect_mask.append(rect.left + col)
  
    # "local group" = a set of character rects with inter-rect horizontal gaps of less than EXPIRY_DIGIT_WIDTH
    # "super-group" = a set of local groups with inter-group horizontal gaps of less than 2 * EXPIRY_DIGIT_WIDTH
  
    # Expiry must be a local group (for now, anyhow).
    # Name is a super-group (since we'll get firstname and lastname as separate local groups).
  
    # Collect character rects into local groups
    local_groups = gather_into_groups(non_overlapping_rect_list, EXPIRY_DIGIT_WIDTH)
  
    # Collect local groups into super-groups
    super_groups = gather_into_groups(local_groups, 2 * EXPIRY_DIGIT_WIDTH)
  
    if doing_segmentation:
        MINIMUM_EXPIRY_STRIP_CHARACTERS = 2
        MINIMUM_NAME_STRIP_CHARACTERS = 2
    else:
        MINIMUM_EXPIRY_STRIP_CHARACTERS = 5
        MINIMUM_NAME_STRIP_CHARACTERS = 5

    local_groups = [group for group in local_groups if len(group.character_rects) >= MINIMUM_EXPIRY_STRIP_CHARACTERS - 1]
    super_groups = [group for group in super_groups if len(group.character_rects) >= MINIMUM_NAME_STRIP_CHARACTERS - 1]
    
    for group in local_groups:
        regrid_group(sobel_image, group)
    
    for group in super_groups:
        regrid_group(sobel_image, group)

    groups = []
    for group in local_groups:
        optimize_character_rects(sobel_image, group, y_offset)
        if len(group.character_rects) > 0:
            groups.append(group)
    local_groups = groups

    groups = []
    for group in super_groups:
        optimize_character_rects(sobel_image, group, y_offset)
        if len(group.character_rects) > 0:
            groups.append(group)
    super_groups = groups

    local_groups = [group for group in local_groups if len(group.character_rects) >= MINIMUM_EXPIRY_STRIP_CHARACTERS]
    super_groups = [group for group in super_groups if len(group.character_rects) >= MINIMUM_NAME_STRIP_CHARACTERS]

    return local_groups, super_groups
    
    
def gather_into_groups(items, horizontal_tolerance):
    groups = []

    # Sort items from left to right

    items = sorted(items, key=lambda group: group.left)

    for base_index in range(len(items)):
        base_item = items[base_index]
        if base_item.grouped_yet:
            continue
        group = GroupedRects(base_item.left, base_item.top, base_item.width, base_item.height, False, 0, [], EXPIRY_DIGIT_WIDTH)
        gather_character_rects(group, base_item)
        base_item.grouped_yet = True
        for index in range(base_index + 1, len(items)):
            item = items[index]
            if item.left - (group.left + group.width) >= horizontal_tolerance:
                break
            if not item.grouped_yet:
                item.grouped_yet = True
                former_bottom = group.top + group.height
                group.top = min(group.top, item.top)
                group.width = item.left + item.width - base_item.left
                group.height = max(former_bottom, item.top + item.height) - group.top
                gather_character_rects(group, item)
        groups.append(group)

    for group in groups:
        strip_group_white_space(group)
        
    return groups


def gather_character_rects(group, sub_group):
    group.sum += sub_group.sum

    if len(sub_group.character_rects) == 0:
        group.character_rects.append(CharacterRect(sub_group.top, sub_group.left, sub_group.sum))
    else:
        group.character_rects.extend(sub_group.character_rects)


def strip_group_white_space(group):
    # Strip leading or trailing "white-space" from super-groups, based on the average sum of the central 4 character rects
    WHITESPACE_THRESHOLD = 0.8
    if len(group.character_rects) > 5:
        white_space_found = False
        index = (len(group.character_rects) - 4) // 2
        threshold_sum = ((group.character_rects[index + 0].sum +
                          group.character_rects[index + 1].sum +
                          group.character_rects[index + 2].sum +
                          group.character_rects[index + 3].sum) / 4) * WHITESPACE_THRESHOLD
    
        if group.character_rects[0].sum < threshold_sum:
            # print "Stripping leading space."
            group.character_rects = group.character_rects[1:]
            group.left = group.character_rects[0].left
            white_space_found = True
        elif group.character_rects[-1].sum < threshold_sum:
            # print "Stripping trailing space."
            group.character_rects = group.character_rects[:-2]
            white_space_found = True
        
        if white_space_found:
            group.width = group.character_rects[-1].left + group.character_width - group.left
            strip_group_white_space(group)


def regrid_group(sobel_image, group):
    MIN_GRID_SPACING = 11
    MAX_GRID_SPACING = 15
    best_grid_spacing = 0
    best_starting_col_offset = 0
    best_ratio = sys.maxint
    
    bounds_left = max(group.left - 2 * EXPIRY_DIGIT_WIDTH, 0)
    bounds_right = min(group.left + group.width + 2 * EXPIRY_DIGIT_WIDTH, CREDIT_CARD_TARGET_WIDTH)
    bounds_width = bounds_right - bounds_left
    minimum_allowable_number_of_grid_lines = math.floor(float(bounds_width) / float(MIN_GRID_SPACING))

    group_sum = 0
    col_sums = []
    for col in range(bounds_left, bounds_right):
        cv.SetImageROI(sobel_image, (col, group.top, 1, group.height))
        col_sum = cv.Sum(sobel_image)[0]
        col_sums.append(col_sum)
        group_sum += col_sum

    cv.ResetImageROI(sobel_image)

    for grid_spacing in range(MIN_GRID_SPACING, MAX_GRID_SPACING + 1):
        for starting_col_offset in range(grid_spacing):
            grid_line_sum = 0.0
            number_of_grid_lines = 0
            grid_line_offset = starting_col_offset
            while grid_line_offset < bounds_width:
                number_of_grid_lines += 1
                grid_line_sum += col_sums[grid_line_offset]
                grid_line_offset += grid_spacing
            average_grid_line_sum = grid_line_sum / float(number_of_grid_lines)
            grid_line_sum = average_grid_line_sum * minimum_allowable_number_of_grid_lines
            ratio = grid_line_sum / (group_sum - grid_line_sum)
            if ratio < best_ratio:
                best_ratio = ratio
                best_grid_spacing = grid_spacing
                best_starting_col_offset = starting_col_offset

    # print "BEST Spacing: {}, Start: {}, Ratio: {}".format(best_grid_spacing, best_starting_col_offset, best_ratio)
    
    regridded_rects = []
    grid_line_offset = best_starting_col_offset
    while grid_line_offset + 1 < bounds_width:
        regridded_rects.append(CharacterRect(group.top,
                                             bounds_left + grid_line_offset + 1,
                                             sum(col_sums[grid_line_offset + 1 : min(grid_line_offset + best_grid_spacing, bounds_width)])))
        grid_line_offset += best_grid_spacing
    
    group.character_rects = regridded_rects
    group.character_width = best_grid_spacing - 1
    group.left = group.character_rects[0].left
    group.width = group.character_rects[-1].left + group.character_width - group.left
    strip_group_white_space(group)


def optimize_character_rects(image, group, y_offset):
    EXPANDED_CHARACTER_IMAGE_WIDTH = 18
    CHARACTER_RECT_OUTSET = 2
    card_image_size = cv.GetSize(image)

    group_left = max(group.left - CHARACTER_RECT_OUTSET, 0)
    group_top = max(group.top - CHARACTER_RECT_OUTSET, y_offset + NUMBER_HEIGHT)
    group_width = min(group.left + group.width + CHARACTER_RECT_OUTSET, card_image_size[0]) - group_left
    group_height = min(group.top + group.height + CHARACTER_RECT_OUTSET, card_image_size[1]) - group_top
    cv.SetImageROI(image, (group_left, group_top, group_width, group_height))
    group_image = cv.CreateImage((group_width, group_height), cv.IPL_DEPTH_16S, 1)
    cv.Copy(image, group_image)

    # normalize & threshold is time-consuming (though probably somewhat optimizable),
    # but does help to more consistently position the image
    cv.Normalize(group_image, group_image, 255, 0, cv.CV_C)
    cv.Threshold(group_image, group_image, 100, 255, cv.CV_THRESH_TOZERO)

    row_sums = []
    top_row = 0
    bottom_row = group_height - 1

    for row in range(top_row, group_height):
        cv.SetImageROI(group_image, (0, row, group_width, 1))
        row_sums.append(int(cv.Sum(group_image)[0]))

    while group_height > TRIMMED_CHARACTER_IMAGE_HEIGHT:
        if row_sums[top_row] <= row_sums[bottom_row]:
            top_row += 1
        else:
            bottom_row -= 1
        group_height -= 1
    
    group.top += top_row - CHARACTER_RECT_OUTSET
    group.height = TRIMMED_CHARACTER_IMAGE_HEIGHT
    
    character_image = cv.CreateImage((EXPANDED_CHARACTER_IMAGE_WIDTH * 2, TRIMMED_CHARACTER_IMAGE_HEIGHT), cv.IPL_DEPTH_16S, 1)
    character_image_width = group.character_width + 2 * CHARACTER_RECT_OUTSET

    for rect_index in reversed(range(0, len(group.character_rects))):
        rect_left = group.character_rects[rect_index].left - CHARACTER_RECT_OUTSET
        rect_top = group.top
        if (rect_left < 0 or
            rect_left + character_image_width > card_image_size[0]):
          group.character_rects.pop(rect_index)
          continue

        cv.SetImageROI(image, (rect_left, rect_top, character_image_width, TRIMMED_CHARACTER_IMAGE_HEIGHT))
        cv.SetImageROI(character_image, (0, 0, character_image_width, TRIMMED_CHARACTER_IMAGE_HEIGHT))
        cv.Copy(image, character_image)

        # normalize & threshold is time-consuming (though probably somewhat optimizable),
        # but does help to more consistently position the image
        cv.Normalize(character_image, character_image, 255, 0, cv.CV_C)
        cv.Threshold(character_image, character_image, 100, 255, cv.CV_THRESH_TOZERO)

        character_width = character_image_width
        col_sums = []
        left_col = 0
        right_col = character_width - 1

        for col in range(left_col, right_col + 1):
            cv.SetImageROI(character_image, (col, 0, 1, TRIMMED_CHARACTER_IMAGE_HEIGHT))
            col_sums.append(int(cv.Sum(character_image)[0]))

        while character_width > TRIMMED_CHARACTER_IMAGE_WIDTH:
            if col_sums[left_col] <= col_sums[right_col]:
                left_col += 1
            else:
                right_col -= 1
            character_width -= 1

        group.character_rects[rect_index] = CharacterRect(top = rect_top,
                                                          left = rect_left + left_col,
                                                          sum = sum(col_sums[left_col : right_col + 1]))

    if len(group.character_rects) > 0:
        group.character_width = TRIMMED_CHARACTER_IMAGE_WIDTH
        group.left = group.character_rects[0].left
        group.width = group.character_rects[-1].left + TRIMMED_CHARACTER_IMAGE_WIDTH - group.left

    cv.ResetImageROI(image)
