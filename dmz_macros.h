//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_MACROS_H
#define DMZ_MACROS_H

#define dmz_likely(x) __builtin_expect(!!(x),1)
#define dmz_unlikely(x) __builtin_expect(!!(x),0)

#define DMZ_INTERNAL static

#endif  // DMZ_MACROS_H
