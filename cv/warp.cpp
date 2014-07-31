//
//  warp.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "warp.h"
#include "dmz_debug.h"
#include "processor_support.h"

#include "eigen.h"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "dmz.h"

#ifdef IOS_DMZ
#include "mz_ios.h"
#endif

#ifdef ANDROID_DMZ
#include "mz_android.h"
#endif

bool llcv_warp_auto_upsamples() {
#ifdef IOS_DMZ
  return true;
#else
  return false;
#endif
}

void llcv_calc_persp_transform(float *matrixData, int matrixDataSize, bool rowMajor, const dmz_point sourcePoints[], const dmz_point destPoints[]) {

  // Set up matrices a and b so we can solve for x from ax = b
  // See http://xenia.media.mit.edu/~cwren/interpolator/ for a
  // good explanation of the basic math behind this.

  typedef Eigen::Matrix<float, 8, 8> Matrix8x8;
  typedef Eigen::Matrix<float, 8, 1> Matrix8x1;

  Matrix8x8 a;
  Matrix8x1 b;

  for(int i = 0; i < 4; i++) {
    a(i, 0) = sourcePoints[i].x;
    a(i, 1) = sourcePoints[i].y;
    a(i, 2) = 1;
    a(i, 3) = 0;
    a(i, 4) = 0;
    a(i, 5) = 0;
    a(i, 6) = -sourcePoints[i].x * destPoints[i].x;
    a(i, 7) = -sourcePoints[i].y * destPoints[i].x;

    a(i + 4, 0) = 0;
    a(i + 4, 1) = 0;
    a(i + 4, 2) = 0;
    a(i + 4, 3) = sourcePoints[i].x;
    a(i + 4, 4) = sourcePoints[i].y;
    a(i + 4, 5) = 1;
    a(i + 4, 6) = -sourcePoints[i].x * destPoints[i].y;
    a(i + 4, 7) = -sourcePoints[i].y * destPoints[i].y;

    b(i, 0) = destPoints[i].x;
    b(i + 4, 0) = destPoints[i].y;
  }

  // Solving ax = b for x, we get the values needed for our perspective
  // matrix. Table of options on the eigen site at
  // /dox/TutorialLinearAlgebra.html#TutorialLinAlgBasicSolve
  //
  // We use householderQr because it places no restrictions on matrix A,
  // is moderately fast, and seems to be sufficiently accurate.
  //
  // partialPivLu() seems to work as well, but I am wary of it because I
  // am unsure of A is invertible. According to the documenation and basic
  // performance testing, they are both roughly equivalent in speed.
  //
  // - @burnto

  Matrix8x1 x = a.householderQr().solve(b);

  // Initialize matrixData
  for (int i = 0; i < matrixDataSize; i++) {
    matrixData[i] = 0.0f;
  }
  int matrixSize = (matrixDataSize >= 16) ? 4 : 3;

  // Initialize a 4x4 eigen matrix. We may not use the final
  // column/row, but that's ok.
  Eigen::Matrix4f perspMatrix = Eigen::Matrix4f::Zero();

  // Assign a, b, d, e, and i
  perspMatrix(0, 0) = x(0, 0); // a
  perspMatrix(0, 1) = x(1, 0); // b
  perspMatrix(1, 0) = x(3, 0); // d
  perspMatrix(1, 1) = x(4, 0); // e
  perspMatrix(2, 2) = 1.0f;    // i

  // For 4x4 matrix used for 3D transform, we want to assign
  // c, f, g, and h to the fourth col and row.
  // So we use an offset for thes values
  int o = matrixSize - 3; // 0 or 1
  perspMatrix(0, 2 + o) = x(2, 0); // c
  perspMatrix(1, 2 + o) = x(5, 0); // f
  perspMatrix(2 + o, 0) = x(6, 0); // g
  perspMatrix(2 + o, 1) = x(7, 0); // h
  perspMatrix(2 + o, 2 + o) = 1.0f; // i

  // Assign perspective matrix to our matrixData buffer,
  // swapping row versus column if needed, and taking care not to
  // overflow if user didn't provide a large enough matrixDataSize.
  for(int c = 0; c < matrixSize; c++) {
    for(int r = 0; r < matrixSize; r++) {
      int index = rowMajor ? (c + r * matrixSize) : (r + c * matrixSize);
      if (index < matrixDataSize) {
        matrixData[index] = perspMatrix(r, c);
      }
    }
  }
  // TODO - instead of copying final values into matrixData return array, do one of:
  // (a) assign directly into matrixData, or
  // (b) use Eigen::Mat so that assignment goes straight into underlying matrixData
  // (see https://github.com/lumberlabs/dmz/issues/5#issuecomment-6715537 for our discussion on this)

}




void llcv_unwarp(dmz_context *dmz, IplImage *input, const dmz_point source_points[4], const dmz_rect to_rect, IplImage *output) {
	//  dmz_point source_points[4], dest_points[4];

#ifdef IOS_DMZ
	ios_gpu_unwarp(dmz, input, source_points, output);
#else

  dmz_trace_log("pre-warp in.width:%i in.height:%i in.widthStep:%i in.nChannels:%i", input->width, input->height, input->widthStep, input->nChannels);
	
  assert(output != null);
  assert(output->imageData != null);
  
  dmz_trace_log("expecting out.width:%i out.height:%i out.widthStep:%i out.nChannels:%i", output->width, output->height, output->widthStep, output->nChannels);

#if ANDROID_USE_GLES_WARP
	if (dmz_use_gles_warp()) {
		llcv_gles_warp_perspective(dmz->mz, input, source_points, output);
	}
#endif
	if (!dmz_use_gles_warp()) {
		/* if dmz_use_gles_warp() has changed from above, then we've encountered an error and are falling back to the old way.*/

		// Old-fashioned openCV
		float matrix[16];
		dmz_point dest_points[4];
		dmz_rect_get_points(to_rect, dest_points);

		// Calculate row-major matrix
		llcv_calc_persp_transform(matrix, 9, true, source_points, dest_points);
		CvMat *cv_persp_mat = cvCreateMat(3, 3, CV_32FC1);
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				CV_MAT_ELEM(*cv_persp_mat, float, r, c) = matrix[3 * r + c];
			}
		}
		cvWarpPerspective(input, output, cv_persp_mat, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
		cvReleaseMat(&cv_persp_mat);
	}
#endif // !IOS_DMZ
}



#endif

