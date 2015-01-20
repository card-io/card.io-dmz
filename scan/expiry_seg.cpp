//
//  expiry_seg.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "expiry_seg.h"
#include "dmz_debug.h"
#include "opencv2/imgproc/imgproc_c.h"

//#define DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE 1

// slash categorizer
#include "models/expiry/modelm_730c4cbd.hpp"

#pragma mark - image preparation

DMZ_INTERNAL void prepare_image_for_seg(IplImage *image, IplImage *as_float, CharacterRect *rect) {
  // Input image: IPL_DEPTH_8U [0 - 255]
  // Data for models: IPL_DEPTH_32F [0.0 - 1.0]
  
  cvSetImageROI(image, cvRect(rect->left, rect->top, kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight));
  cvConvertScale(image, as_float, 1.0f / 255.0f, 0); // TODO: vectorize this as a llcv_* function
  cvResetImageROI(image);
}

#pragma mark - slash detection via machine learning

typedef Eigen::Matrix<float, 1, 176, Eigen::RowMajor> SlashModelInput;
typedef Eigen::Matrix<float, 1, 2, Eigen::RowMajor> SlashProbabilities;

DMZ_INTERNAL inline SlashProbabilities slash_probabilities(IplImage *as_float) {
  assert(as_float->width * sizeof(float) == as_float->widthStep);
  Eigen::Map<SlashModelInput> slash_model_input((float *)as_float->imageData);
  SlashProbabilities probabilities = applym_730c4cbd(slash_model_input);
  return probabilities;
}

DMZ_INTERNAL bool is_slash(IplImage *sobel_image, IplImage *as_float, CharacterRect *rect) {
  prepare_image_for_seg(sobel_image, as_float, rect);
  SlashProbabilities probabilities = slash_probabilities(as_float);
  return probabilities(0, 0) > 0.7f;
}

#pragma mark - locate candidate character rectangles

// Long integers are sufficiently large to hold all of our sums:
//
// Maximum pixel value possible with abs(Scharr) operator = 3 * 255 + 10 * 255 + 3 * 255 = 4080, so use a 16S destination.
// Maximum line-sum possible = 4080 * kCreditCardTargetWidth = 1,746,240 = 0x1AA540, so line-sum will always fit in a long.
// Maximum stripe-sum possible = (max line-sum possible) * kSmallCharacterHeight = 26,193,600 = 0x18FAEC0, so stripe-sum will always fit in a long.
// Maximum character-rect sum possible = 4080 * kSmallCharacterWidth * kSmallCharacterHeight = 550,800 = 0x86790, so character-rect sum will always fit in a long.
// Maximum grouped-rect sum possible = (max character-rect sum) * (max possible #characters per group)
//                                   = (max character-rect sum) * (kCreditCardTargetWidth / kSmallCharacterWidth)
//                                   = 26,193,600 = 0x18FAEC0, so group-rect sum will always fit in a long.
// [Note: It's no coincidence that (Maximum grouped-rect sum possible) == (Maximum stripe-sum possible).]

struct StripeSum
 {
  int   base_row;
  long  sum;
};

struct StripeSumCompareDescending
 : public std::binary_function<StripeSum, StripeSum, bool> {
  inline bool operator()(StripeSum const &stripe_sum_1, StripeSum const &stripe_sum_2) const {
    return (stripe_sum_1.sum > stripe_sum_2.sum);
  }
};

struct CharacterRectCompareSumDescending
 : public std::binary_function<CharacterRect, CharacterRect, bool> {
  inline bool operator()(CharacterRect const &character_rect_1, CharacterRect const &character_rect_2) const {
    return (character_rect_1.sum > character_rect_2.sum);
  }
};

struct GroupedRectsCompareLeftAscending
 : public std::binary_function<GroupedRects, GroupedRects, bool> {
  inline bool operator()(GroupedRects const &grouped_rect_1, GroupedRects const &grouped_rect_2) const {
    return (grouped_rect_1.left < grouped_rect_2.left);
  }
};

DMZ_INTERNAL void gather_character_rects(GroupedRects &group, const GroupedRects &sub_group) {
  group.sum += sub_group.sum;
  
  if (sub_group.character_rects.size() == 0) {
    group.character_rects.push_back(CharacterRect(sub_group.top, sub_group.left, sub_group.sum));
  }
  else {
    group.character_rects.insert(group.character_rects.end(), sub_group.character_rects.begin(), sub_group.character_rects.end());
  }
}

DMZ_INTERNAL void strip_group_white_space(GroupedRects &group) {
  // Strip leading or trailing "white-space" from super-groups, based on the average sum of the central 4 character rects
  if (group.character_rects.size() > 5) {
#define WHITESPACE_THRESHOLD 0.8
    bool white_space_found = false;
    size_t index = (group.character_rects.size() - 4) / 2;
    long threshold_sum = (long)(((group.character_rects[index + 0].sum +
                                  group.character_rects[index + 1].sum +
                                  group.character_rects[index + 2].sum +
                                  group.character_rects[index + 3].sum) / 4) * WHITESPACE_THRESHOLD);
    
    if (group.character_rects[0].sum < threshold_sum) {
      group.character_rects.erase(group.character_rects.begin());
      group.left = group.character_rects.begin()->left;
      white_space_found = true;
    }
    else if ((group.character_rects.end() - 1)->sum < threshold_sum) {
      group.character_rects.erase(group.character_rects.end() - 1);
      white_space_found = true;
    }

    if (white_space_found) {
      group.width = (group.character_rects.end() - 1)->left + group.character_width - group.left;
      strip_group_white_space(group);
    }
  }
}

DMZ_INTERNAL void gather_into_groups(GroupedRectsList &groups, GroupedRectsList &items, int horizontal_tolerance) {

  std::sort(items.begin(), items.end(), GroupedRectsCompareLeftAscending());
  
  for (size_t base_index = 0; base_index < items.size(); base_index++) {
    GroupedRects *base_item = &items[base_index];
    if (!base_item->grouped_yet) {
      GroupedRects group(*base_item);
      group.sum = 0;
      group.character_rects.clear();
      gather_character_rects(group, *base_item);
      
      base_item->grouped_yet = true;
      
      for (size_t index = base_index + 1; index < items.size(); index++) {
        GroupedRects *item = &items[index];
        if (item->left - (group.left + group.width) >= horizontal_tolerance) {
          break;
        }
        if (!item->grouped_yet) {
          item->grouped_yet = true;
          
          int formerBottom = group.top + group.height;
          group.top = MIN(group.top, item->top);
          group.width = item->left + item->width - base_item->left;
          group.height = MAX(formerBottom, item->top + item->height) - group.top;
          
          gather_character_rects(group, *item);
        }
      }
      groups.push_back(group);
    }
  }
  
  for (GroupedRectsListIterator group = groups.begin(); group != groups.end(); ++group) {
    strip_group_white_space(*group);
  }
}

DMZ_INTERNAL void regrid_group(IplImage *sobel_image, GroupedRects &group) {
  // Choose grid-spacing (and starting column) to minimize the sum of pixel-values covered by the grid lines,
  // while maximizing the sum of pixel-values within the grid squares.
  // I.e., minimize the ratio of the former to the latter.
#define MIN_GRID_SPACING 11
#define MAX_GRID_SPACING 15
  int best_grid_spacing = 0;
  int best_starting_col_offset = 0;
  float best_ratio = MAXFLOAT;
  
  int bounds_left = MAX(group.left - 2 * kSmallCharacterWidth, 0);
  int bounds_right = MIN(group.left + group.width + 2 * kSmallCharacterWidth, kCreditCardTargetWidth);
  int bounds_width = bounds_right - bounds_left;
  int minimum_allowable_number_of_grid_lines = (int)(floorf(float(bounds_width) / float(MIN_GRID_SPACING)));
  
  long group_sum = 0;
  long col_sums[bounds_width];
  for (int col = bounds_left; col < bounds_right; col++) {
    long col_sum = 0;
    for (int row = group.top; row < group.top + group.height; row++) {
      col_sum += CV_IMAGE_ELEM(sobel_image, short, row, col);
    }
    col_sums[col - bounds_left] = col_sum;
    group_sum += col_sum;
  }
  
  for (int grid_spacing = MIN_GRID_SPACING; grid_spacing <= MAX_GRID_SPACING; grid_spacing++) {
    for (int starting_col_offset = 0; starting_col_offset < grid_spacing; starting_col_offset++) {
      float grid_line_sum = 0.0;
      int number_of_grid_lines = 0;
      int grid_line_offset = starting_col_offset;
      
      while (grid_line_offset < bounds_width) {
        number_of_grid_lines += 1;
        grid_line_sum += col_sums[grid_line_offset];
        grid_line_offset += grid_spacing;
      }
      
      float average_grid_line_sum = grid_line_sum / float(number_of_grid_lines);
      grid_line_sum = average_grid_line_sum * minimum_allowable_number_of_grid_lines;
      float ratio = grid_line_sum / (group_sum - grid_line_sum);
      
      if (ratio < best_ratio) {
        best_ratio = ratio;
        best_grid_spacing = grid_spacing;
        best_starting_col_offset = starting_col_offset;
      }
    }
  }
  
  CharacterRectList regridded_rects;
  int grid_line_offset = best_starting_col_offset;
  while (grid_line_offset + 1 < bounds_width) {
    long sum = 0;
    for (int col = grid_line_offset + 1; col < MIN(grid_line_offset + best_grid_spacing, bounds_width); col++) {
      sum += col_sums[col];
    }

    regridded_rects.push_back(CharacterRect(group.top, bounds_left + grid_line_offset + 1, sum));
    grid_line_offset += best_grid_spacing;
  }
  
  group.character_rects = regridded_rects;
  group.character_width = best_grid_spacing - 1;
  group.left = group.character_rects[0].left;
  group.width = (group.character_rects.end() - 1)->left + group.character_width - group.left;
  strip_group_white_space(group);
}

DMZ_INTERNAL void optimize_character_rects(IplImage *sobel_image, GroupedRects &group) {
#define kExpandedCharacterImageWidth 18
#define kExpandedCharacterImageHeight 21
#define kCharacterRectOutset 2
  
  static IplImage *character_image = NULL;
  if (character_image == NULL) {
    character_image = cvCreateImage(cvSize(kExpandedCharacterImageWidth * 2, kExpandedCharacterImageHeight * 2), IPL_DEPTH_16S, 1);
  }
  
  CvSize  card_image_size = cvGetSize(sobel_image);
  int character_image_width = group.character_width + 2 * kCharacterRectOutset;
  int character_image_height = group.height + 2 * kCharacterRectOutset;
  
  for (int rect_index = (int)group.character_rects.size() - 1; rect_index >= 0; rect_index--) {
    int rect_left = group.character_rects[rect_index].left - kCharacterRectOutset;
    int rect_top = group.top - kCharacterRectOutset;
    
    if (rect_left < 0 ||
        rect_left + character_image_width > card_image_size.width ||
        rect_top + character_image_height > card_image_size.height) {
      group.character_rects.erase(group.character_rects.begin() + rect_index);
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
      // dmz_debug_print("Erasing character_rect %d [%d, %d]\n", rect_index, rect_left, rect_top);
#endif
      continue;
    }
    
    cvSetImageROI(sobel_image, cvRect(rect_left, rect_top, character_image_width, character_image_height));
    cvSetImageROI(character_image, cvRect(0, 0, character_image_width, character_image_height));
    cvCopy(sobel_image, character_image);

    // normalize & threshold is time-consuming (though probably somewhat optimizable),
    // but does help to more consistently position the image
    cvNormalize(character_image, character_image, 255, 0, CV_C);
    cvThreshold(character_image, character_image, 100, 255, CV_THRESH_TOZERO);
    
    int character_width = character_image_width;
    int character_height = character_image_height;
    int col_sums[character_width];
    int row_sums[character_height];
    int left_col = 0;
    int right_col = character_width - 1;
    int top_row = 0;
    int bottom_row = character_height - 1;
    
    for (int col = left_col; col <= right_col; col++) {
      col_sums[col] = 0;
      for (int row = top_row; row <= bottom_row; row++) {
        col_sums[col] += CV_IMAGE_ELEM(character_image, short, row, col);
      }
    }
    
    while (character_width > kTrimmedCharacterImageWidth) {
      if (col_sums[left_col] <= col_sums[right_col]) {
        left_col++;
      }
      else {
        right_col--;
      }
      character_width--;
    }
    
    for (int row = top_row; row <= bottom_row; row++) {
      row_sums[row] = 0;
      for (int col = left_col; col <= right_col; col++) {
        row_sums[row] += CV_IMAGE_ELEM(character_image, short, row, col);
      }
    }
    
    while (character_height > kTrimmedCharacterImageHeight) {
      if (row_sums[top_row] <= row_sums[bottom_row]) {
        top_row++;
      }
      else {
        bottom_row--;
      }
      character_height--;
    }
    
    group.character_rects[rect_index].left = rect_left + left_col;
    group.character_rects[rect_index].top = rect_top + top_row;
  }
  
  if (!group.character_rects.empty()) {
    int highest_top = kCreditCardTargetHeight;
    int lowest_top = 0;
    for (CharacterRectListIterator rect = group.character_rects.begin(); rect != group.character_rects.end(); ++rect) {
      highest_top = MIN(highest_top, rect->top);
      lowest_top = MAX(lowest_top, rect->top);
    }

    group.character_width = kTrimmedCharacterImageWidth;
    group.left = group.character_rects[0].left;
    group.width = (group.character_rects.end() - 1)->left + kTrimmedCharacterImageWidth - group.left;
    group.top = highest_top;
    group.height = lowest_top + kTrimmedCharacterImageHeight - group.top;
  }
  
  cvResetImageROI(sobel_image);
}

DMZ_INTERNAL void find_character_groups_for_stripe(IplImage *card_y, IplImage *sobel_image, int stripe_base_row, long stripe_sum, GroupedRectsList &expiry_groups, GroupedRectsList &name_groups) {
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_start(1);
#endif
  
  // "Expanded" stripe is kSmallCharacterHeight + 2 scan lines in height ("expanded" refers to the "+ 2" -- one extra scan line above and below the input stripe)
  CvSize  card_image_size = cvGetSize(sobel_image);
  int     expanded_stripe_top = stripe_base_row - 1;
  CvRect  expanded_stripe_rect = cvRect(0, expanded_stripe_top, card_image_size.width, MIN(kSmallCharacterHeight + 2, card_image_size.height - expanded_stripe_top));

  // Any rect whose pixel-sum is less than rectangle_summation_threshold is too dim to care about
#define RECT_AVERAGE_THRESHOLD_FACTOR 5
  long rect_average_based_on_stripe_sum = ((stripe_sum * kSmallCharacterWidth) / card_image_size.width);
  float rectangle_summation_threshold = rect_average_based_on_stripe_sum / RECT_AVERAGE_THRESHOLD_FACTOR;
  
  // [1] Calculate the pixel-sum for each possible character rectangle within the stripe...
  CharacterRectList rect_list;
  float rect_sum_total = 0;
  float rect_sum_average = 0;
  long rect_sum = 0;
  
  // [1a] Calculate pixel-sum for the leftmost character rect
  
  for (int col = 0; col < kSmallCharacterWidth; col++) {
    for (int row = 0; row < expanded_stripe_rect.height; row++) {
      rect_sum += CV_IMAGE_ELEM(sobel_image, short, stripe_base_row + row, col);
    }
  }
  
  // [1b] For each possible character rect...
  
  for (int col = 0; col < card_image_size.width - kSmallCharacterWidth + 1; col++) {
    
    // Record pixel-sum of current character rect (ignoring excessively dim rects)
    
    if (rect_sum > rectangle_summation_threshold) {
      CharacterRect rect;
      rect.top = expanded_stripe_rect.y;
      rect.left = col;
      rect.sum = rect_sum;
      rect_list.push_back(rect);
      
      rect_sum_total += (float)rect_sum;
    }
    
    if (col < card_image_size.width - kSmallCharacterWidth) {
      
      // Update pixels-sum by subtracting the leftmost pixel values and adding the next pixel values to the right
      
      for (int row = 0; row < expanded_stripe_rect.height; row++) {
        rect_sum -= CV_IMAGE_ELEM(sobel_image, short, stripe_base_row + row, col);
        rect_sum += CV_IMAGE_ELEM(sobel_image, short, stripe_base_row + row, col + kSmallCharacterWidth);
      }
    }
  }
  
  if (rect_list.empty()) {
    return;
  }
  
  rect_sum_average = (rect_sum_total / rect_list.size());
#define RECT_SUM_THRESHOLD_FACTOR 0.8
  float rect_sum_threshold = (float) (RECT_SUM_THRESHOLD_FACTOR * rect_sum_average);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("calc all rects", 1);
#endif
  
  // [2] Sort rectangles descending by sum
  
  std::sort(rect_list.begin(), rect_list.end(), CharacterRectCompareSumDescending());
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg[256];
  sprintf(msg, "sort %ld non-zero rects", rect_list.size());
  dmz_debug_timer_print(msg, 1);
#endif
  
  // [3] Find the non-overlapping rectangles, ignoring rectangles whose sum is excessively small (compared to the average rect sum)
  
  GroupedRectsList non_overlapping_rect_list;
  
  bool non_overlapping_rect_mask[expanded_stripe_rect.width];
  memset(non_overlapping_rect_mask, 0, sizeof(non_overlapping_rect_mask));
  
  for (CharacterRectListIterator rect = rect_list.begin(); rect != rect_list.end(); ++ rect) {
    if ((float)rect->sum <= rect_sum_threshold) {
      break;
    }
    
    if (!non_overlapping_rect_mask[rect->left] && !non_overlapping_rect_mask[rect->left + kSmallCharacterWidth - 1]) {
      GroupedRects grouped_rect;
      grouped_rect.top = rect->top;
      grouped_rect.left = rect->left;
      grouped_rect.width = kSmallCharacterWidth;
      grouped_rect.height = expanded_stripe_rect.height;
      grouped_rect.grouped_yet = false;
      grouped_rect.sum = rect->sum;
      grouped_rect.character_width = kSmallCharacterWidth;
      non_overlapping_rect_list.push_back(grouped_rect);
      
      non_overlapping_rect_mask[rect->left + 0] = true;
      non_overlapping_rect_mask[rect->left + 1] = true;
      non_overlapping_rect_mask[rect->left + 2] = true;
      non_overlapping_rect_mask[rect->left + 3] = true;
      non_overlapping_rect_mask[rect->left + 4] = true;
      non_overlapping_rect_mask[rect->left + 5] = true;
      non_overlapping_rect_mask[rect->left + 6] = true;
      non_overlapping_rect_mask[rect->left + 7] = true;
      non_overlapping_rect_mask[rect->left + 8] = true;
      assert(8 == kSmallCharacterWidth - 1);
    }
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg2[256];
  sprintf(msg2, "find %ld non-overlapping rects", non_overlapping_rect_list.size());
  dmz_debug_timer_print(msg2, 1);
#endif
  
  // "local group" = a set of character rects with inter-rect horizontal gaps of less than kSmallCharacterWidth
  // "super-group" = a set of local groups with inter-group horizontal gaps of less than 2 * kSmallCharacterWidth
  //
  // Expiry must be a local group (for now, anyhow).
  // Name is a super-group (since we'll get firstname and lastname as separate local groups).
  
  // [4] Collect character rects into local groups
  
  GroupedRectsList local_groups;
  gather_into_groups(local_groups, non_overlapping_rect_list, kSmallCharacterWidth);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg3[256];
  sprintf(msg3, "%ld local groups", local_groups.size());
  dmz_debug_timer_print(msg3, 1);
#endif
  
  // [5] Collect local groups into super-groups
  GroupedRectsList super_groups;
  // Let's skip these for the moment, while we're focusing on expiry:
  // gather_into_groups(super_groups, local_groups, 2 * kSmallCharacterWidth);
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg4[256];
  sprintf(msg4, "%ld super-groups", super_groups.size());
  dmz_debug_timer_print(msg4, 1);
#endif
  
  GroupedRectsList new_groups;
  for (GroupedRectsListIterator group = local_groups.begin(); group != local_groups.end(); ++group) {
    if (group->character_rects.size() >= kMinimumExpiryStripCharacters - 1) {
      new_groups.push_back(*group);
    }
  }
  local_groups = new_groups;
  
  new_groups.clear();
  for (GroupedRectsListIterator group = super_groups.begin(); group != super_groups.end(); ++group) {
    if (group->character_rects.size() >= kMinimumNameStripCharacters - 1) {
      new_groups.push_back(*group);
    }
  }
  super_groups = new_groups;
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg5[256];
  sprintf(msg5, "width-filtering -> %ld local groups, %ld super-groups", local_groups.size(), super_groups.size());
  dmz_debug_timer_print(msg5, 1);
#endif
  
  for (GroupedRectsListIterator group = local_groups.begin(); group != local_groups.end(); ++group) {
    regrid_group(sobel_image, *group);
  }
  
  for (GroupedRectsListIterator group = super_groups.begin(); group != super_groups.end(); ++group) {
    regrid_group(sobel_image, *group);
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("regrid the groups", 1);
#endif
  
  for (int index = (int)local_groups.size() - 1; index >= 0; index--) {
    optimize_character_rects(sobel_image, local_groups[index]);
    if (local_groups[index].character_rects.size() == 0) {
      local_groups.erase(local_groups.begin() + index);
      // dmz_debug_print("Erasing local_group %d, which is now empty.\n", index);
    }
  }
  
  for (int index = (int)super_groups.size() - 1; index >= 0; index--) {
    optimize_character_rects(sobel_image, super_groups[index]);
    if (super_groups[index].character_rects.size() == 0) {
      super_groups.erase(super_groups.begin() + index);
    }
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("shrink the character rects", 1);
#endif
  
  new_groups.clear();
  for (GroupedRectsListIterator group = local_groups.begin(); group != local_groups.end(); ++group) {
    if (group->character_rects.size() >= kMinimumExpiryStripCharacters) {
      new_groups.push_back(*group);
    }
  }
  local_groups = new_groups;
  
  new_groups.clear();
  for (GroupedRectsListIterator group = super_groups.begin(); group != super_groups.end(); ++group) {
    if (group->character_rects.size() >= kMinimumNameStripCharacters) {
      new_groups.push_back(*group);
    }
  }
  super_groups = new_groups;
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  char msg6[256];
  sprintf(msg6, "width-filtering -> %ld local groups, %ld super-groups", local_groups.size(), super_groups.size());
  dmz_debug_timer_print(msg6, 1);
#endif
  
  // Add local groups to the passed-in expiry_groups GroupedRectsList, iff they contain a slash in a reasonable position
  
  static IplImage *as_float = NULL;
  if (as_float == NULL) {
    as_float = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), IPL_DEPTH_32F, 1);
  }
  
  for (GroupedRectsListIterator group = local_groups.begin(); group != local_groups.end(); ++group) {
    if (group->character_rects.size() < 5) {
      continue;
    }
    for (size_t firstCharacterIndex = 0; firstCharacterIndex + 4 < group->character_rects.size(); firstCharacterIndex++) {
      if (is_slash(sobel_image, as_float, &group->character_rects[firstCharacterIndex + 2])) {
        GroupedRects grouped_5_characters;
        grouped_5_characters.top = group->character_rects[firstCharacterIndex].top;
        grouped_5_characters.left = group->character_rects[firstCharacterIndex].left;
        grouped_5_characters.width = kSmallCharacterWidth;
        grouped_5_characters.height = kSmallCharacterHeight;
        grouped_5_characters.grouped_yet = false;
        grouped_5_characters.sum = 0;
        grouped_5_characters.character_width = kTrimmedCharacterImageWidth;
        grouped_5_characters.pattern = ExpiryPatternMMsYY;
        
        for (size_t index = 0; index < 5; index++) {
          CharacterRect char_rect = group->character_rects[firstCharacterIndex + index];
          int formerBottom = grouped_5_characters.top + grouped_5_characters.height;
          grouped_5_characters.top = MIN(char_rect.top, grouped_5_characters.top);
          grouped_5_characters.width = (char_rect.left + kSmallCharacterWidth) - grouped_5_characters.left;
          grouped_5_characters.height = MAX(char_rect.top + kSmallCharacterHeight, formerBottom) - grouped_5_characters.top;
          grouped_5_characters.character_rects.push_back(char_rect);
        }
        
        expiry_groups.push_back(grouped_5_characters);
      }
    }
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("insert local groups into expiry_groups param", 1);
#endif
  
  // Add supergroups to the passed-in name_groups GroupedRectsList
  name_groups.insert(name_groups.end(), super_groups.begin(), super_groups.end());

#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("insert supergroups into name_groups param", 1);
#endif
}

DMZ_INTERNAL void best_expiry_seg(IplImage *card_y, uint16_t starting_y_offset, GroupedRectsList &expiry_groups, GroupedRectsList &name_groups) {
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
  
  std::vector<StripeSum> probable_stripes;

  for (std::vector<StripeSum>::iterator stripe_sum = stripe_sums.begin(); stripe_sum != stripe_sums.end(); ++stripe_sum) {
    bool overlap = false;
    for (std::vector<StripeSum>::iterator probable_stripe = probable_stripes.begin(); probable_stripe != probable_stripes.end(); ++probable_stripe) {
      if (probable_stripe->base_row - kSmallCharacterHeight < stripe_sum->base_row &&
          stripe_sum->base_row < probable_stripe->base_row + kSmallCharacterHeight) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      probable_stripes.push_back(*stripe_sum);
      if (probable_stripes.size() >= kNumberOfStripesToTry) {
        break;
      }
    }
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("pick probable stripes");
#endif
  
  // For each stripe, find the potential expiry groups and name groups:
  
  for (std::vector<StripeSum>::iterator probable_stripe = probable_stripes.begin(); probable_stripe != probable_stripes.end(); ++probable_stripe) {
    find_character_groups_for_stripe(card_y, sobel_image, probable_stripe->base_row, probable_stripe->sum, expiry_groups, name_groups);
  }
  
#if DEBUG_EXPIRY_SEGMENTATION_PERFORMANCE
  dmz_debug_timer_print("find character groups");
  dmz_debug_print("Grand Total for Expiry segmentation: %.3f\n", ((float)dmz_debug_timer_stop()) / 1000.0);
#endif
  
  cvReleaseImage(&sobel_image);
}

#endif // COMPILE_DMZ
