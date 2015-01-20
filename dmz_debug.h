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

#if CYTHON_DMZ
#define dmz_debug_timer_start(...)
#define dmz_debug_timer_lap(...)
#define dmz_debug_timer_stop(...) 0
#define dmz_debug_timer_print(label, ...)
#define dmz_debug_timer_log(label, ...)
#define dmz_error_timer_log(label, ...)
#elif (DMZ_DEBUG && IOS_DMZ) // will hopefully be fine on Android too -- if so, feel free to remove this IOS_DMZ requirement
#include <sys/time.h>

static suseconds_t dmz_debug_timer_start_microseconds[10];
static suseconds_t dmz_debug_timer_lap_microseconds[10];

void dmz_debug_timer_start(int timer_number = 0) {
  struct timeval time;
  gettimeofday(&time, NULL);
  dmz_debug_timer_start_microseconds[timer_number] = dmz_debug_timer_lap_microseconds[timer_number] = (suseconds_t)(time.tv_sec * 1000000) + time.tv_usec;
}

suseconds_t dmz_debug_timer_lap(int timer_number = 0) {
  struct timeval time;
  gettimeofday(&time, NULL);
  suseconds_t now = (suseconds_t)(time.tv_sec * 1000000) + time.tv_usec;
  suseconds_t interval = now - dmz_debug_timer_lap_microseconds[timer_number];
  dmz_debug_timer_lap_microseconds[timer_number] = now;
  return interval;
}

suseconds_t dmz_debug_timer_stop(int timer_number = 0) {
  struct timeval time;
  gettimeofday(&time, NULL);
  suseconds_t now = (suseconds_t)(time.tv_sec * 1000000) + time.tv_usec;
  suseconds_t interval = now - dmz_debug_timer_start_microseconds[timer_number];
  dmz_debug_timer_start_microseconds[timer_number] = dmz_debug_timer_lap_microseconds[timer_number] = now;
  return interval;
}

suseconds_t dmz_debug_timer_print(const char *label, int timer_number = 0) {
  suseconds_t interval = dmz_debug_timer_lap(timer_number);
  dmz_debug_print("[%d] %.3f %s\n", timer_number, ((float)interval) / 1000.0, label);
  return interval;
}

suseconds_t dmz_debug_timer_log(const char *label, int timer_number = 0) {
  suseconds_t interval = dmz_debug_timer_lap(timer_number);
  dmz_debug_log("[%d] %.3f %s\n", timer_number, ((float)interval) / 1000.0, label);
  return interval;
}

suseconds_t dmz_error_timer_log(const char *label, int timer_number = 0) {
  suseconds_t interval = dmz_debug_timer_lap(timer_number);
  dmz_error_log("[%d] %.3f %s\n", timer_number, ((float)interval) / 1000.0, label);
  return interval;
}
#else
#define dmz_debug_timer_start(...)
#define dmz_debug_timer_lap(...)
#define dmz_debug_timer_stop(...) 0
#define dmz_debug_timer_print(label, ...)
#define dmz_debug_timer_log(label, ...)
#define dmz_error_timer_log(label, ...)
#endif

#endif  // DMZ_DEBUG_H
