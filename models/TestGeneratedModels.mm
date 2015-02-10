//
//  TestGeneratedModels.mm
//  See the file "LICENSE.md" for the full license governing this code.
//


#if TEST_GENERATED_MODELS

#import "TestGeneratedModels.h"

// vert seg mlp models
#import "modelm_befe75da.hpp"

// conv models
#include "modelc_5c241121.hpp"
#include "modelc_01266c1b.hpp"
#include "modelc_b00bf70c.hpp"

#if SCAN_EXPIRY
// expiry models
#include "modelm_730c4cbd.hpp"
#include "modelc_bf4dd6c8.hpp"
//#include "modelm_d38dff65.hpp"
//#include "modelm_f6aa7969.hpp"
//#include "modelm_cb758d40.hpp"
//#include "modelm_9a27fb30.hpp"
//#include "modelm_ad529645.hpp"
//#include "modelm_db226864.hpp"
#endif

@implementation TestGeneratedModels

static BOOL failure = NO;

#define SELF_CHECK_MODEL(model_fn_name) if (model_fn_name()) { \
                                          printf("Model %s passes self-check.\n", #model_fn_name); \
                                        } \
                                        else { \
                                          printf("Model %s fails self-check.\n", #model_fn_name); \
                                          failure = YES; \
                                        }

+ (void)testVSegMlpCategorization {
  SELF_CHECK_MODEL(passm_befe75da);
}

+ (void)testConvCategorization {
  SELF_CHECK_MODEL(passc_5c241121);
  SELF_CHECK_MODEL(passc_01266c1b);
  SELF_CHECK_MODEL(passc_b00bf70c);
}

+ (void)testExpiryModels {
#if SCAN_EXPIRY
  SELF_CHECK_MODEL(passm_730c4cbd);
  SELF_CHECK_MODEL(passc_bf4dd6c8);
//  SELF_CHECK_MODEL(passm_d38dff65);
//  SELF_CHECK_MODEL(passm_f6aa7969);
//  SELF_CHECK_MODEL(passm_cb758d40);
//  SELF_CHECK_MODEL(passm_9a27fb30);
//  SELF_CHECK_MODEL(passm_ad529645);
//  SELF_CHECK_MODEL(passm_db226864);
#endif
}

+ (void)selfCheck {
  [self testVSegMlpCategorization];
  [self testConvCategorization];
  [self testExpiryModels];
  NSAssert(!failure, @"One or more models failed self-check.");
}

@end

#endif