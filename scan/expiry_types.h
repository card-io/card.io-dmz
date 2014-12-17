//
//  expiry_types.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef DMZ_SCAN_EXPIRY_TYPES_H
#define DMZ_SCAN_EXPIRY_TYPES_H

#include "dmz_macros.h"
#include "eigen.h"
#include <vector>

#if DMZ_DEBUG
#include "opencv2/core/core_c.h" // needed for IplImage
#endif

#define kSmallCharacterWidth 9
#define kSmallCharacterHeight 15

#define kTrimmedCharacterImageWidth 11
#define kTrimmedCharacterImageHeight 16

#define kMinimumExpiryStripCharacters 5
#define kMinimumNameStripCharacters 5

// possible expiry formats: (s = separator = '/' | '-')
// Separators, and therefore the apparent format, are identified in expiry_seg.
// Therefore, expiry_categorize just worries about identifying the digits, and then the month/year.
// MMsYY
// MMs20YY
// MMsDDsYY or DDsMMsYY, where s1 == s2
// MMsDDs20YY or DDsMMs20YY, where s1 == s2
// MM-MM/YY
// MM-MM/20YY
// MMsYY-MMsYY
#define kExpiryMaxValidLength 11

enum ExpiryPattern
 {
  ExpiryPatternMMsYY,
  ExpiryPatternMMs20YY,
  ExpiryPatternXXsXXsYY,
  ExpiryPatternXXsXXs20YY,
  ExpiryPatternMMdMMsYY,
  ExpiryPatternMMdMMs20YY,
  ExpiryPatternMMsYYdMMsYY,
};

typedef Eigen::Matrix<float, kExpiryMaxValidLength, 10, Eigen::RowMajor> ExpiryGroupScores;

struct CharacterRect
 {
  int   top;
  int   left;
  long  sum;
#if DMZ_DEBUG
  IplImage  *final_image; // be sure to call cvReleaseImage() eventually!
#endif
  
#if DMZ_DEBUG
  CharacterRect() : top(0), left(0), sum(0), final_image(NULL) {};
  CharacterRect(const int top, const int left, const long sum) : top(top), left(left), sum(sum), final_image(NULL) {};
#else
  CharacterRect() : top(0), left(0), sum(0) {};
  CharacterRect(const int top, const int left, const long sum) : top(top), left(left), sum(sum) {};
#endif
};

typedef std::vector<CharacterRect> CharacterRectList;
typedef std::vector<CharacterRect>::iterator CharacterRectListIterator;
typedef std::vector<CharacterRect>::reverse_iterator CharacterRectListReverseIterator;

struct GroupedRects
 {
  int   top;
  int   left;
  int   width;
  int   height;
  bool  grouped_yet;
  long  sum;
  int   character_width;
  CharacterRectList character_rects;
  
  ExpiryPattern pattern;
  ExpiryGroupScores scores;
  
  int   recently_seen_count; // used when aggregating groups across frames
  int   total_seen_count;    // used when aggregating groups across frames
};

typedef std::vector<GroupedRects> GroupedRectsList;
typedef std::vector<GroupedRects>::iterator GroupedRectsListIterator;

// FOR CYTHON USE ONLY
#if CYTHON_DMZ
typedef struct {
  int   top;
  int   left;
} CythonCharacterRect;

typedef float CythonGroupScores[kExpiryMaxValidLength][10];

typedef struct {
  int     top;
  int     left;
  int     width;
  int     height;
  int     character_width;
  
  uint8_t           pattern;
  CythonGroupScores scores;
  int               recently_seen_count;
  int               total_seen_count;
  
  int   number_of_character_rects;
  CythonCharacterRect *character_rects;
} CythonGroupedRects;
#endif

#endif
