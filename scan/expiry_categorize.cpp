//
//  expiry_categorize.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "expiry_categorize.h"
#include "cv/image_util.h"
#include "cv/morph.h"
#include "cv/stats.h"
#include <time.h>

#define DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE 0
#define DEBUG_EXPIRY_CATEGORIZATION_RESULTS 1

// digit categorizers
#include "models/expiry/modelc_ab0b6054.hpp"
#include "models/expiry/modelc_d3fc216e.hpp"

#define GROUPED_RECTS_VERTICAL_ALLOWANCE (kTrimmedCharacterImageHeight / 2)
#define GROUPED_RECTS_HORIZONTAL_ALLOWANCE (kTrimmedCharacterImageWidth / 2)

#define kExpiryDecayFactor 0.7f
#define kExpiryMinStability 0.7f

#define digit_to_int(c) ((uint8_t)c - (uint8_t)'0')

typedef Eigen::Matrix<float, 16, 11, Eigen::RowMajor> DigitModelInput;
typedef Eigen::Matrix<float, 1, 10, Eigen::RowMajor> DigitProbabilities;

#pragma mark - image preparation

DMZ_INTERNAL void prepare_image_for_cat(IplImage *image, IplImage *as_float, CharacterRectListIterator rect) {
  // Input image: IPL_DEPTH_8U [0 - 255]
  // Data for models: IPL_DEPTH_32F [0.0 - 1.0]
  
  cvSetImageROI(image, cvRect(rect->left, rect->top, kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight));
  
  // TODO: optimize this a lot!
  
  // Gradient
  IplImage *filtered_image = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), IPL_DEPTH_8U, 1);
  //llcv_morph_grad3_2d_cross_u8(image, filtered_image);
  IplConvKernel *kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
  cvMorphologyEx(image, filtered_image, NULL, kernel, CV_MOP_GRADIENT, 1);
  cvReleaseStructuringElement(&kernel);
  
  // Equalize
  llcv_equalize_hist(filtered_image, filtered_image);
  
  // Bilateral filter
  int aperture = 3;
  double space_sigma = (aperture / 2.0 - 1) * 0.3 + 0.8;
  double color_sigma = (aperture - 1) / 3.0;
  IplImage *smoothed_image = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), IPL_DEPTH_8U, 1);
  cvSmooth(filtered_image, smoothed_image, CV_BILATERAL, aperture, aperture, space_sigma, color_sigma);
  
  // Convert to float
  cvConvertScale(smoothed_image, as_float, 1.0f / 255.0f, 0);
  
  cvReleaseImage(&smoothed_image);
  cvReleaseImage(&filtered_image);
  
  cvResetImageROI(image);

#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_timer_print("prepare image", 2);
#endif
}

#pragma mark - categorize expiry digits via machine learning

DMZ_INTERNAL inline std::vector<DigitProbabilities> digit_probabilities(IplImage *as_float) {
  std::vector<DigitProbabilities> probabilities;
  assert(as_float->width * sizeof(float) == as_float->widthStep);
  Eigen::Map<DigitModelInput> digit_model_input((float *)as_float->imageData);

//  probabilities.push_back(applyc_e1533ae9(digit_model_input));
//#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
//  suseconds_t interval1 = dmz_debug_timer_print("apply model 0", 2);
//#endif
  
  probabilities.push_back(applyc_ab0b6054(digit_model_input));
#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  /*suseconds_t interval2 = */ dmz_debug_timer_print("apply model 1", 2);
  //  dmz_debug_print("Faster: %d, %.2f\n", interval1 - interval2, ((float)interval2) / ((float)interval1));
#endif

  return probabilities;
}


DMZ_INTERNAL inline ExpiryGroupScores categorize_expiry_digits(IplImage *card_y, IplImage *as_float, GroupedRects group, char *expiries_string) {
  std::vector<std::string> expiry_strings;
  std::vector<ExpiryGroupScores> probability_vector;
  
  char positions[256];
  sprintf(positions, " top: %3d, left: %3d character-lefts:", group.top, group.left);
  
  for (int character_index = 0; character_index < 5; character_index++) {
    if (character_index == 2) { // the slash character
      continue;
    }
    
    CharacterRectListIterator rect = group.character_rects.begin() + character_index;
    
    prepare_image_for_cat(card_y, as_float, rect);
    std::vector<DigitProbabilities> probabilities = digit_probabilities(as_float);
    
    char position[32];
    sprintf(position, " %3d", rect->left);
    strcat(positions, position);
    
    if (character_index == 0) {
      ExpiryGroupScores scores;
      for (int model_index = 0; model_index < probabilities.size(); model_index++) {
        char header[64];
        sprintf(header, "**/** Model %d ", model_index);
        expiry_strings.push_back(std::string(header));
        probability_vector.push_back(scores);
      }
      
#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
      dmz_debug_timer_print("initialize expiry_strings", 2);
#endif
    }
    
    for (int model_index = 0; model_index < expiry_strings.size(); model_index++) {
      float max_probability = 0.0f;
      int most_probable_digit = -1;
      for (int digit_index = 0; digit_index < 10; digit_index++) {
        float digit_probability = probabilities[model_index](0, digit_index);
        probability_vector[model_index](character_index, digit_index) = digit_probability;
        if (digit_probability > max_probability) {
          max_probability = digit_probability;
          most_probable_digit = digit_index;
        }
      }
      if (most_probable_digit >= 0 && max_probability > 0.7) {
        expiry_strings[model_index][character_index] = (char)(int('0') + most_probable_digit);
      }
    }
  }
  
  expiries_string[0] = '\0';
  for (int model_index = 0; model_index < expiry_strings.size(); model_index++) {
    if (strlen(expiries_string)) {
      strcat(expiries_string, "\n");
    }
    strcat(expiries_string, expiry_strings[model_index].c_str());
    strcat(expiries_string, positions);
    strcat(expiries_string, "\n");
    
    strcat(expiries_string, "        ");
    char label_string[32];
    for (int char_label = 0; char_label < 10; char_label++) {
      sprintf(label_string, "   %d  ", char_label);
      strcat(expiries_string, label_string);
    }
    strcat(expiries_string, "\n");
    
    for (int character_index = 0; character_index < 5; character_index++) {
      if (character_index == 2) { // the slash character
        continue;
      }
      
      char char_pos_string[32];
      sprintf(char_pos_string, "char %d:", character_index);
      strcat(expiries_string, char_pos_string);
      
      for (int digit_index = 0; digit_index < 10; digit_index++) {
        char prob_string[32];
        sprintf(prob_string, " %5.3f", probability_vector[model_index](character_index, digit_index));
        strcat(expiries_string, prob_string);
      }
      strcat(expiries_string, "\n");
    }
  }
  
  // TODO: if we end up using more than one model, return their coalesced results
  return probability_vector[0];
}


DMZ_INTERNAL void expiry_aggregate_grouped_rects(GroupedRectsList &aggregated_groups, GroupedRectsList &new_groups) {
  // Coalesce equivalent groups within new_groups (*** TODO *** IS THIS STEP EVER ACTUALLY NECESSARY? ***)
  for (size_t new_index_1 = 0; new_index_1 < new_groups.size(); new_index_1++) {
    GroupedRects &group1 = new_groups[new_index_1];
    int top1 = group1.top;
    int left1 = group1.left;
    size_t nChars1 = group1.character_rects.size();
    float groups_coalesced_so_far = 1;
    
    for (size_t new_index_2 = new_groups.size() - 1; new_index_2 > new_index_1; new_index_2--) {
      GroupedRects &group2 = new_groups[new_index_2];
      if (abs(group2.top - top1) > GROUPED_RECTS_VERTICAL_ALLOWANCE ||
          abs(group2.left - left1) > GROUPED_RECTS_HORIZONTAL_ALLOWANCE ||
          group2.character_rects.size() != nChars1) {
        continue;
      }

      group1.scores = ((group1.scores * groups_coalesced_so_far) + group2.scores) / (groups_coalesced_so_far + 1);
      groups_coalesced_so_far++;
      new_groups.erase(new_groups.begin() + new_index_2);
      dmz_debug_print("** Yup, coalesced a new group with another! WTF? ***\n");
    }
  }
  
#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_timer_print("coalesce new groups", 2);
#endif
  
  // Coalesce new_groups with equivalent groups inside aggregated_groups
  for (GroupedRectsListIterator old_group = aggregated_groups.begin(); old_group != aggregated_groups.end(); ++old_group) {
    int old_top = old_group->top;
    int old_left = old_group->left;
    size_t old_n_chars = old_group->character_rects.size();
    
    for (int new_index = (int)new_groups.size() - 1; new_index >= 0; new_index--) {
      GroupedRects &new_group = new_groups[new_index];
      if (abs(new_group.top - old_top) > GROUPED_RECTS_VERTICAL_ALLOWANCE ||
          abs(new_group.left - old_left) > GROUPED_RECTS_HORIZONTAL_ALLOWANCE ||
          new_group.character_rects.size() != old_n_chars) {
        continue;
      }

      old_group->recently_seen_count++;
      old_group->total_seen_count++;
      old_group->scores = (old_group->scores * kExpiryDecayFactor) + (new_group.scores * (1 - kExpiryDecayFactor));
      old_group->top = new_group.top;
      old_group->left = new_group.left;
      new_groups.erase(new_groups.begin() + new_index);
    }
  }
  
#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_timer_print("coalesce new and old groups", 2);
#endif
  
  // Decrement recently_seen_count for each group inside aggregated_groups,
  // and forget any that haven't been seen for a while
  for (int old_index = (int)aggregated_groups.size() - 1; old_index >= 0; old_index--) {
    aggregated_groups[old_index].recently_seen_count--;
    if (aggregated_groups[old_index].recently_seen_count <= 0) {
      aggregated_groups.erase(aggregated_groups.begin() + old_index);
    }
  }
  
  // Add new, non-equivalent, groups to aggregated_groups
  for (GroupedRectsListIterator new_group = new_groups.begin(); new_group != new_groups.end(); ++new_group) {
    GroupedRects fresh_group(*new_group);
    fresh_group.recently_seen_count = 3; // stick around for at least the next couple of frames
    fresh_group.total_seen_count = 1;
    aggregated_groups.push_back(fresh_group);
  }
  
#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_timer_print("aggregate groups", 2);
#endif
}


DMZ_INTERNAL void expiry_string_to_expiry_month_and_year(char *expiry_string, GroupedRects &group, int *expiry_month, int *expiry_year) {
  int month = -1;
  int year = -1;
  
  switch (group.pattern) {
    case ExpiryPatternMMsYY:
      if (expiry_string[0] != ' ' && expiry_string[1] != ' ' && expiry_string[3] != ' ' && expiry_string[4] != ' ') {
        month = digit_to_int(expiry_string[0]) * 10 + digit_to_int(expiry_string[1]);
        year = digit_to_int(expiry_string[3]) * 10 + digit_to_int(expiry_string[4]);
      }
      break;
    case ExpiryPatternMMs20YY:
    case ExpiryPatternXXsXXsYY:
    case ExpiryPatternXXsXXs20YY:
    case ExpiryPatternMMdMMsYY:
    case ExpiryPatternMMdMMs20YY:
    case ExpiryPatternMMsYYdMMsYY:
    default:
      break;
  }
  
  // http://support.celerant.com/celwiki/index.php/Reverse_expiration_date suggests that non-US cards
  // might sometimes reverse month and year. Since MMYY and YYMM are distinguishable as of 2013,
  // and since we're going to ignore dates in the past, let's swap ours accordingly.
  if (month > 12 && year > 0 && year <= 12) {
    int temp = month;
    month = year;
    year = temp;
  }
  
  // Only accept dates where month is in [1,12],
  // and the date is >= the current month/year,
  // and the year is within the next 5 years. (TODO: somehow determine whether 5 is a reasonable number)
  // Ignore valid dates if they're no later than the best date we've already found.
  int full_year = year + 2000;
  if (month > 0 && month <= 12 &&
      (full_year > *expiry_year || ((full_year == *expiry_year) && month > *expiry_month))) {
    time_t now = time(NULL);
    struct tm *time_struct = localtime(&now);
    int current_year = time_struct->tm_year + 1900;
    int current_month = time_struct->tm_mon + 1;
    if (full_year < current_year + 5
        && (full_year > current_year || (full_year == current_year && month >= current_month))
        ) {
      *expiry_month = month;
      *expiry_year = full_year;
    }
#if DMZ_DEBUG || CYTHON_DMZ
    else {
      // For current testing, which includes several expired cards, allow dates in the past.
      if (year > 60) {
        full_year= year + 1900;
      }
      if (full_year < current_year + 5) {
        *expiry_month = month;
        *expiry_year = full_year;
      }
      else {
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
        dmz_debug_print("%02d/%04d is a disallowed date.\n", month, full_year);
#endif
      }
    }
#endif
  }
}


DMZ_INTERNAL void get_stable_expiry_month_and_year(GroupedRects &group, int *expiry_month, int *expiry_year) {
  char expiry_string[128];
  memset(expiry_string, 0, sizeof(expiry_string));

  for(uint8_t i = 0; i < group.character_rects.size(); i++) {
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
    switch (group.pattern) {
      case ExpiryPatternMMsYY:
        if (i == 2) {
          dmz_debug_print("- ");
          continue;
        }
        break;
      case ExpiryPatternMMs20YY:
      case ExpiryPatternXXsXXsYY:
      case ExpiryPatternXXsXXs20YY:
      case ExpiryPatternMMdMMsYY:
      case ExpiryPatternMMdMMs20YY:
      case ExpiryPatternMMsYYdMMsYY:
      default:
        break;
    }
#endif
    ExpiryGroupScores::Index r, c;
    float max_score = group.scores.row(i).maxCoeff(&r, &c);
    float sum = group.scores.row(i).sum();
    float stability = max_score / sum;
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
    dmz_debug_print("%d ", (int) ceilf(stability * 100));
#endif
    if (stability < kExpiryMinStability) {
      expiry_string[i] = ' ';
    }
    else {
      expiry_string[i] = (char) ((uint8_t)'0' + (uint8_t)c);
    }
  }
  
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
  dmz_debug_print("\n");
#endif
  
  expiry_string_to_expiry_month_and_year(expiry_string, group, expiry_month, expiry_year);
}


DMZ_INTERNAL void expiry_extract(IplImage *card_y,
                                 GroupedRectsList &expiry_groups,
                                 GroupedRectsList &new_groups,
                                 int *expiry_month,
                                 int *expiry_year) {
  if (new_groups.empty()) {
    return;
  }

  static IplImage *as_float = NULL;
  if (as_float == NULL) {
    as_float = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), IPL_DEPTH_32F, 1);
  }

#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_timer_start(2);
#endif
  
  // For each group identified by expiry_seg, categorize the supposed digits:
  
  for (GroupedRectsListIterator group = new_groups.begin(); group != new_groups.end(); ++group) {
    char expiries_string[8192];
    group->scores = categorize_expiry_digits(card_y, as_float, *group, expiries_string);
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
    dmz_debug_print("\n%s\n", expiries_string);
#endif
  }

  // Aggregate the newly found groups with those we've previously found:
  
  expiry_aggregate_grouped_rects(expiry_groups, new_groups);
  
  // Pick the best month/year from all of the aggregated groups:
  
  for (GroupedRectsListIterator group = expiry_groups.begin(); group != expiry_groups.end(); ++group) {
    if (group->total_seen_count < 3) {
      // If we haven't yet seen this group at least 3 times, let's not trust it yet.
      continue;
    }
    
#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
    dmz_debug_print("Expiry stability (Group %d): ", (int) (group - expiry_groups.begin()));
#endif
    get_stable_expiry_month_and_year(*group, expiry_month, expiry_year);
  }

#if DEBUG_EXPIRY_CATEGORIZATION_PERFORMANCE
  dmz_debug_print("Grand Total for Expiry categorization: %.3f\n", ((float)dmz_debug_timer_stop(2)) / 1000.0);
#endif

#if DEBUG_EXPIRY_CATEGORIZATION_RESULTS
  dmz_debug_print("Returning expiry %02d/%04d\n", *expiry_month, *expiry_year);
#endif
}


#if CYTHON_DMZ
DMZ_INTERNAL void expiry_extract_group(IplImage *card_y,
                                       GroupedRects &group,
                                       ExpiryGroupScores &old_scores,
                                       int *expiry_month,
                                       int *expiry_year) {
  static IplImage *as_float = NULL;
  if (as_float == NULL) {
    as_float = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), IPL_DEPTH_32F, 1);
  }
  
  char expiries_string[8192];
  group.scores = categorize_expiry_digits(card_y, as_float, group, expiries_string);

  group.scores = (old_scores * kExpiryDecayFactor) + (group.scores * (1 - kExpiryDecayFactor));
  
  get_stable_expiry_month_and_year(group, expiry_month, expiry_year);
}
#endif

#endif // COMPILE_DMZ
