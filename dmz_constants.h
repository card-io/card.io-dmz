//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_CONSTANTS_H
#define DMZ_CONSTANTS_H


#define kCreditCardTargetWidth 428
#define kCreditCardTargetHeight 270

#define kPortraitSampleWidth 480
#define kPortraitSampleHeight 640

#define kNumberWidth 19
#define kNumberHeight 27

#define kPortraitVerticalInset ((kPortraitSampleHeight - kCreditCardTargetHeight) / 2)
#define kPortraitVerticalPercentInset ((float)kPortraitVerticalInset / (float)kPortraitSampleHeight)
#define kPortraitHorizontalInset ((kPortraitSampleWidth - kCreditCardTargetWidth) / 2)
#define kPortraitHorizontalPercentInset ((float)kPortraitHorizontalInset / (float)kPortraitSampleWidth)

#define kLandscapeSampleWidth kPortraitSampleHeight
#define kLandscapeSampleHeight kPortraitSampleWidth

#define kLandscapeVerticalInset ((kLandscapeSampleHeight - kCreditCardTargetHeight) / 2)
#define kLandscapeVerticalPercentInset ((float)kLandscapeVerticalInset / (float)kLandscapeSampleHeight)
#define kLandscapeHorizontalInset ((kLandscapeSampleWidth - kCreditCardTargetWidth) / 2)
#define kLandscapeHorizontalPercentInset ((float)kLandscapeHorizontalInset / (float)kLandscapeSampleWidth)

#endif // DMZ_CONSTANTS_H
