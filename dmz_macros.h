//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_MACROS_H
#define DMZ_MACROS_H

#define dmz_likely(x) __builtin_expect(!!(x),1)
#define dmz_unlikely(x) __builtin_expect(!!(x),0)

#define DMZ_INTERNAL static

#if CYTHON_DMZ
#define DMZ_INTERNAL_UNLESS_CYTHON
#else
#define DMZ_INTERNAL_UNLESS_CYTHON static
#endif

#endif  // DMZ_MACROS_H
