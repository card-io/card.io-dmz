//  See the file "LICENSE.md" for the full license governing this code.

#ifndef PROCESSOR_SUPPORT_H
#define PROCESSOR_SUPPORT_H

#include "mz.h"
#include "dmz_macros.h"

//
// Wraps client-specific processor support checks
//
// To check for compiletime support, use #if DMZ_HAS_NEON_COMPILETIME.
// To check for runtime support (presence of actual coprocessor), use dmz_has_neon_runtime().
// A normal pattern for NEON usage is:
//
// if(dmz_has_neon_runtime()) {
//     #if DMZ_HAS_NEON_COMPILETIME
//     // NEON implementation
//     #endif
// } else {
//     // Scalar implementation
// }
//

// Enable simple compiletime checks for NEON support
#if IOS_DMZ
    #ifdef _ARM_ARCH_7
        #define DMZ_HAS_NEON_COMPILETIME 1
    #else
        #define DMZ_HAS_NEON_COMPILETIME 0
    #endif
#elif CYTHON_DMZ
    #define DMZ_HAS_NEON_COMPILETIME 0
#elif ANDROID_DMZ
    #if ANDROID_HAS_NEON
        #define DMZ_HAS_NEON_COMPILETIME 1
    #else
        #define DMZ_HAS_NEON_COMPILETIME 0
    #endif
	#ifndef ANDROID_USE_GLES_WARP
		#define ANDROID_USE_GLES_WARP 0
	#endif
#else
    #error "Encountered unknown dmz client. Make sure the right *_DMZ preprocessor macro is set."
#endif

/* For Android ARMv7a architectures:
 * gcc -mfpu=neon <=> DMZ_HAS_NEON_COMPILETME
 * else:
 * gcc -mfpu=vfpv3-d16 <=> limit to 16 VFP registers (normally 32)
 */
extern int dmz_has_neon_runtime(void);
extern int dmz_use_vfp3_16(void);

// Should we use OpenGL ES to do perspective (un)warping?
extern int dmz_use_gles_warp(void);

// Used to set fallback - Set to 0 if a GL error occurs or if the OS knows that OpenGL is not supported.
extern void dmz_set_gles_warp(int newstate);

#endif // PROCESSOR_SUPPORT_H
