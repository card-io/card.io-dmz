//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_DEBUG_H
#define DMZ_DEBUG_H

#ifndef DMZ_MZ_H
#error "mz.h must be included before dmz_debug.h for correct functionality"
#endif

// useful for extreme verbosity.
#if DMZ_TRACE
  #define DMZ_DEBUG 1
  #define dmz_trace_log(format_string, ...) dmz_debug_log(format_string, ##__VA_ARGS__)
#else
  #define dmz_trace_log(format_string, ...)
#endif

#if (DMZ_DEBUG && ANDROID_DMZ)

  #include <android/log.h>

  #define DMZ_DEBUG_TAG "card.io dmz"

  #define dmz_debug_print(format_string, ...) __android_log_print(ANDROID_LOG_DEBUG, DMZ_DEBUG_TAG, format_string, ##__VA_ARGS__)
  #define dmz_debug_log(format_string, ...) __android_log_print(ANDROID_LOG_DEBUG, DMZ_DEBUG_TAG, format_string, ##__VA_ARGS__)
  #define dmz_error_log(format_string, ...) __android_log_print(ANDROID_LOG_ERROR, DMZ_DEBUG_TAG, format_string, ##__VA_ARGS__)

#elif (DMZ_DEBUG && IOS_DMZ)

  #include <stdio.h>

  #define dmz_debug_print(format_string, ...) fprintf(stderr, format_string, ##__VA_ARGS__)
  #define dmz_debug_log(format_string, ...) dmz_debug_print("card.io dmz: " format_string "\n", ##__VA_ARGS__)
  #define dmz_error_log(format_string, ...) dmz_debug_print("card.io error: " format_string "\n", ##__VA_ARGS__)

#else

  #define dmz_debug_print(format_string, ...)
  #define dmz_debug_log(format_string, ...)
  #define dmz_error_log(format_string, ...) // it is tempting to log here, but that could result in logging in released versions of the SDK

#endif


#endif  // DMZ_DEBUG_H
