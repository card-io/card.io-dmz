#include "compile.h"
#if ANDROID_DMZ

#include "mz_android.h"

#if ANDROID_USE_GLES_WARP

int llcv_gl_error_count() {
	EGLint error = eglGetError();
	if (error != EGL_SUCCESS) {
		dmz_error_log("egl error: %i", error);
		dmz_set_gles_warp(0);
		return 1;
	}

	int err_count = 0;
	GLenum glerror;
	while ((glerror = glGetError())) {
		err_count++;
		dmz_error_log("gl error: 0x%X (%i)", glerror, glerror);
	}
	if (err_count) 	dmz_set_gles_warp(0);
	return err_count;
}


void llcv_gles_setup(llcv_gles_context* mz, int width, int height) {
	mz->pbufWidth = width;
	mz->pbufHeight = height;

	if (dmz_use_gles_warp() && mz->egl_display == NULL) {
		dmz_debug_log("setting up %i x %i rendering surface", mz->pbufWidth, mz->pbufHeight);

		mz->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		EGLint major, minor, error;
		if (eglInitialize(mz->egl_display, &major, &minor) && mz->egl_display != EGL_NO_DISPLAY ) {
			dmz_debug_log("Initialized GLES and got version %i.%i", major, minor);
		}
		else {
			int n_err = llcv_gl_error_count();
			dmz_debug_log("Failed to initialize GLES display with %i errors", n_err);
//			llcv_gl_supported = false;
			return;
		}

		int attribList[] = {
		            EGL_DEPTH_SIZE, 0,
		            EGL_STENCIL_SIZE, 0,
		            EGL_RED_SIZE, 8,
		            EGL_GREEN_SIZE, 8,
		            EGL_BLUE_SIZE, 8,
		            EGL_ALPHA_SIZE, 8,
		            EGL_NONE
		};
		EGLint max_configs = 5;
		EGLint num_configs = 0;
		EGLConfig configs[max_configs];
		eglChooseConfig(mz->egl_display, attribList, configs, max_configs,  &num_configs);
		if (num_configs == 0) {
			llcv_gl_error_count();
			dmz_debug_log("didn't get any EGL configs!");
//			llcv_gl_supported = false;
			return;
		}
		mz->egl_context = eglCreateContext(mz->egl_display, configs[0], EGL_NO_CONTEXT, NULL);
		if (mz->egl_context == EGL_NO_CONTEXT) {
			llcv_gl_error_count();
			dmz_debug_log("Failed to create an EGL context");
//			llcv_gl_supported = false;
			return;
		}
		int pbAttribList[] = {
	            EGL_WIDTH, 	mz->pbufWidth,
	            EGL_HEIGHT, mz->pbufHeight,
	            EGL_NONE
		};
		mz->egl_surface = eglCreatePbufferSurface(mz->egl_display, configs[0], pbAttribList);
		if (! eglMakeCurrent(mz->egl_display, mz->egl_surface, mz->egl_surface, mz->egl_context)) {
			llcv_gl_error_count();
			dmz_debug_log("gles warp not available");
			return;
		}

		// "onSurfaceCreated"
	    glGenTextures(1, &(mz->gl_texture));
	    glBindTexture(GL_TEXTURE_2D, mz->gl_texture);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glEnable(GL_TEXTURE_2D);			//Enable Texture Mapping ( NEW )
		glShadeModel(GL_SMOOTH); 			//Enable Smooth Shading
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f); 	//Black Background

		if (llcv_gl_error_count()) return;

		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

		// "onSurfaceChanged"

		glViewport(0, 0, mz->pbufWidth, mz->pbufHeight);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrthof(-1, 1, -1, 1, -1, 1);

		if (llcv_gl_error_count()) return;

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// onetime setup

		GLfloat h = 1.0f;
		GLfloat w = 1.0f;

//	    gl_verts = {
//				-w, -h,  0.0f,		// V1 - bottom left
//				-w,  h,  0.0f,		// V2 - top left
//				 w, -h,  0.0f,		// V3 - bottom right
//				 w,  h,  0.0f			// V4 - top right
//	    };
		// appease the compiler.
		GLfloat* gl_verts = mz->gl_verts;
		gl_verts[0] = gl_verts[3] = -w;
		gl_verts[1] = gl_verts[7] = -h;
		gl_verts[2] = gl_verts[5] = gl_verts[8] = gl_verts[11] = 0.0f;
		gl_verts[4] = gl_verts[10] = h;
		gl_verts[6] = gl_verts[9] = w;

		if (llcv_gl_error_count()) return;
	}
}


void llcv_gles_teardown(llcv_gles_context *mz) {
	dmz_debug_log("tearing down gles");

	eglMakeCurrent(mz->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	eglDestroyContext(mz->egl_display, mz->egl_context);
	eglDestroySurface(mz->egl_display, mz->egl_surface);
	eglTerminate(mz->egl_display);
	mz->egl_display = NULL;

	llcv_gl_error_count();
}

#endif // ANDROID_USE_GLES_WARP

void llcv_gles_warp_perspective(void* mzv, IplImage *input, const dmz_point corners[4], IplImage *card) {
#if ANDROID_USE_GLES_WARP
	llcv_gles_context* mz = (llcv_gles_context*) mzv;

	// the docs say somewhere that width & height must be even
    assert(input->height % 2 == 0);
    assert(input->width % 2 == 0);
    assert(input->imageData != NULL);

	dmz_debug_log("gles warp: engage!");

	int err_count = 0;

    GLfloat texCorners[]  = {
    		corners[0].x/input->width, corners[0].y/input->height,
    		corners[2].x/input->width, corners[2].y/input->height,
    		corners[1].x/input->width, corners[1].y/input->height,
    		corners[3].x/input->width, corners[3].y/input->height,
    };

#if DMZ_DEBUG
    for (int i=0; i<8; i+=2) {
    	dmz_debug_log("\tsurface corner: (%f, %f)", texCorners[i], texCorners[i+1]);
    }
#endif

    assert(card->width == mz->pbufWidth);
    assert(card->height == mz->pbufHeight);

    GLint bufFormat;
    if (input->nChannels == 1) bufFormat = GL_LUMINANCE;
    else if (input->nChannels == 3) bufFormat = GL_RGB;
    else if (input->nChannels == 4) bufFormat = GL_RGBA;
    else {
    	dmz_debug_log("unexpected number of channels: %i  I expected one of {1, 3, 4}", input->nChannels);
    	return;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, bufFormat, input->width, input->height, 0, bufFormat, GL_UNSIGNED_BYTE, input->imageData);
	if (llcv_gl_error_count()) return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	if (llcv_gl_error_count()) return;

    glBindTexture(GL_TEXTURE_2D, mz->gl_texture);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	if (llcv_gl_error_count()) return;

    glFrontFace(GL_CW);

    err_count = llcv_gl_error_count();
    if (err_count) dmz_debug_log("%i errors after enable client state", err_count);

    glVertexPointer(3, GL_FLOAT, 0, mz->gl_verts);
    glTexCoordPointer(2, GL_FLOAT, 0, texCorners);

    err_count = llcv_gl_error_count();
    if (err_count) dmz_debug_log("%i errors after pointer", err_count);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    err_count = llcv_gl_error_count();
    if (err_count) dmz_debug_log("%i errors after drawArrays", err_count);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    err_count = llcv_gl_error_count();
    if (err_count) dmz_debug_log("%i errors at end of warp", err_count);

    dmz_debug_log("reading pixels %i x %i", card->width, card->height);
    glReadPixels(0, 0, card->width, card->height, GL_RGBA, GL_UNSIGNED_BYTE, card->imageData);

    err_count = llcv_gl_error_count();
    if (err_count) dmz_debug_log("%i errors after readPixels", err_count);
#endif // ANDROID_USE_GLES_WARP
}

void *DMZ_MANGLE_NAME(mz_create)(void) {
	void* mz = NULL;
#if ANDROID_USE_GLES_WARP
	mz = malloc(sizeof(llcv_gles_context));
	llcv_gles_setup((llcv_gles_context*)mz, kCreditCardTargetWidth, kCreditCardTargetHeight);
#endif
	return mz;
}

// Destruction of the MZ
void DMZ_MANGLE_NAME(mz_destroy)(void *mz) {
#if ANDROID_USE_GLES_WARP
	llcv_gles_teardown((llcv_gles_context*)mz);
#endif
	if (mz != NULL) {
		free(mz);
	}
}

#endif
