//  See the file "LICENSE.md" for the full license governing this code.

#ifndef DMZ_MZ_H
#define DMZ_MZ_H

#include "dmz_macros.h"

// Preprocessor defines for the (hopefully rare!) case in which platform-specific flags are needed, usually
// around handling compilation for NEON and different architectures.
#ifdef __APPLE__
  #include "TargetConditionals.h"
  #if TARGET_OS_IPHONE
    #define IOS_DMZ 1
  #endif
#endif

// Allocation and initialization of the MZ
// This will be implemented differently on each platform
void *mz_create(void);

// Destruction of the MZ
void mz_destroy(void *mz);

// Perform any necessary operations prior to app backgrounding (e.g., calling glFinish() on any OpenGL contexts)
void mz_prepare_for_backgrounding(void *mz);

// Analogues to this are CYTHON_DMZ and ANDROID_DMZ; they are currently defined explicitly by the
// build systems for those platforms. If we come across built-in defines analogous to TARGET_OS_IPHONE,
// we can define CYTHON_DMZ and ANDROID_DMZ here.


// Python helpers for bridging the gap between OpenCV's Python-wrapped images and OpenCV's C images.
#if CYTHON_DMZ
#include <Python.h>
#include "opencv2/core/core_c.h"

extern IplImage *py_mz_create_from_cv_image_data(char *image_data, int image_size,
                                                 int width, int height,
                                                 int64_t depth, int n_channels,
                                                 int roi_x_offset, int roi_y_offset, int roi_width, int roi_height);

extern void py_mz_release_ipl_image(IplImage *image);

extern void py_mz_get_cv_image_data(IplImage *source,
                                    char **image_data, int *image_size,
                                    int *width, int *height,
                                    int64_t *depth, int *n_channels,
                                    int *roi_x_offset, int *roi_y_offset, int *roi_width, int *roi_height
                                   );
#endif


#endif // DMZ_MZ_H