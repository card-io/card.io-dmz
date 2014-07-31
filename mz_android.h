
#ifndef _MZ_ANDROID_H
#define _MZ_ANDROID_H

#include "opencv2/core/core_c.h"
#include "dmz_macros.h"
#include "dmz.h"

#if ANDROID_USE_GLES_WARP

#include <EGL/egl.h>
#include <GLES/gl.h>

typedef struct {
	GLfloat gl_verts[12];
	GLuint gl_texture;

	int pbufWidth;
	int pbufHeight;

	EGLDisplay egl_display;
	EGLContext egl_context;
	EGLSurface egl_surface;

} llcv_gles_context;

void DMZ_MANGLE_NAME(llcv_gles_setup)(llcv_gles_context *mz, int width, int height);
void DMZ_MANGLE_NAME(llcv_gles_teardown)(llcv_gles_context *mz);

#endif

void DMZ_MANGLE_NAME(llcv_gles_warp_perspective)(void* mz, IplImage *input, const dmz_point corners[4], IplImage *card);


#endif
