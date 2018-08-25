  #include "compile.h"

#ifndef DMZ_ALL_H
#define DMZ_ALL_H 1


#if COMPILE_DMZ

#include "./cv/canny.cpp"
#include "./cv/conv.cpp"
#include "./cv/convert.cpp"
#include "./cv/hough.cpp"
#include "./cv/image_util.cpp"
#include "./cv/morph.cpp"
#include "./cv/sobel.cpp"
#include "./cv/stats.cpp"
#include "./cv/warp.cpp"
#include "./dmz.cpp"
#include "./dmz_olm.cpp"
#include "./geometry.cpp"
#include "./models/generated/modelc_01266c1b.cpp"
#include "./models/generated/modelc_5c241121.cpp"
#include "./models/generated/modelc_b00bf70c.cpp"
#include "./models/generated/modelm_befe75da.cpp"
#include "./mz.cpp"
#include "./mz_android.cpp"
#include "./processor_support.cpp"
#include "./scan/frame.cpp"
#include "./scan/n_categorize.cpp"
#include "./scan/n_hseg.cpp"
#include "./scan/n_vseg.cpp"
#include "./scan/scan.cpp"
#include "./scan/scan_analytics.cpp"
#include "./scan/ocre.cpp"

  #if SCAN_EXPIRY
    #include "./models/expiry/modelc_bf4dd6c8.cpp"
    #include "./models/expiry/modelm_730c4cbd.cpp"
    #include "./scan/expiry_categorize.cpp"
    #include "./scan/expiry_seg.cpp"
  #endif

#else

  #include "./dmz_olm.cpp"
  #include "./processor_support.cpp"

#endif  // COMPILE_DMZ

#endif // DMZ_ALL_H