//  See the file "LICENSE.md" for the full license governing this code.

#ifndef COMPILE_H
#define COMPILE_H

#include "mz.h"

#include "dmz_debug.h"

#if IOS_DMZ
    #if USE_CAMERA
        #define COMPILE_DMZ 1
    #else
        #define COMPILE_DMZ 0
    #endif
#elif CYTHON_DMZ
    #define COMPILE_DMZ 1
#elif ANDROID_DMZ
    #define COMPILE_DMZ 1

	// really Android? I need to do this?
	#define __STDC_LIMIT_MACROS
	#include <stdint.h>
#else
    #error "Encountered unknown dmz client. Make sure the right *_DMZ preprocessor macro is set."
#endif

#endif // COMPILE_H
