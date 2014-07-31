// The source below is a (perhaps modified) copy from OpenCV. OpenCV's license header:

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "compile.h"
#if COMPILE_DMZ

#include "dmz_macros.h"
#include "processor_support.h"
#include "canny.h"
#include "sobel.h"
#include "opencv2/core/core.hpp" // needed for IplImage
#include "opencv2/core/internal.hpp"

#if DMZ_HAS_NEON_COMPILETIME
  #include <arm_neon.h>
#endif // DMZ_HAS_NEON_COMPILETIME

DMZ_INTERNAL void llcv_canny7_precomputed_sobel(IplImage *srcarr, IplImage *dstarr, IplImage *sobel_dx, IplImage *sobel_dy, double low_thresh, double high_thresh) {
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stack_top = 0, **stack_bottom = 0;

    CvMat srcstub, *src = cvGetMat( srcarr, &srcstub );
    CvMat dststub, *dst = cvGetMat( dstarr, &dststub );
    CvSize size;
    int low, high;
    int* mag_buf[3];
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    CvMat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( low_thresh > high_thresh )
    {
        double t;
        CV_SWAP( low_thresh, high_thresh, t );
    }

    size = cvGetMatSize( src );

    CvMat *dx;
    CvMat *dy;

    CvMat dx_stub, dy_stub;
    dx = cvGetMat(sobel_dx, &dx_stub);
    dy = cvGetMat(sobel_dy, &dy_stub);

    low = cvFloor(low_thresh);
    high = cvFloor(high_thresh);

    buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );

    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map = (uchar*)(mag_buf[2] + size.width + 2);
    mapstep = size.width + 2;

    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    mag_row = cvMat( 1, size.width, CV_32F );

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag = mag_buf[(i > 0) + 1] + 1;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        uchar* _map;
        int64_t x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;

        /*
          // TODO: Needs dmz neon protection
          // TODO: Test and enable this code, if we can get enough other performance benefits from NEON elsewhere
          // in this function to make it worth having a dedicated NEON or assembly version.
#define kVectorSize 8
            uint16_t scalar_cols = size.width % kVectorSize;
            uint16_t vector_chunks = size.width / kVectorSize;
            uint16_t vector_cols = vector_chunks * kVectorSize;

            for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
              uint16_t col_index = vector_index * kVectorSize;

              int16x8_t dx_q = vld1q_s16(_dx + col_index);
              int16x8_t dy_q = vld1q_s16(_dy + col_index);

              int16x4_t dx_d0 = vget_low_s16(dx_q);
              int16x4_t dx_d1 = vget_high_s16(dx_q);
              int16x4_t dy_d0 = vget_low_s16(dy_q);
              int16x4_t dy_d1 = vget_high_s16(dy_q);

              int32x4_t dx_wq0 = vmovl_s16(dx_d0);
              int32x4_t dx_wq1 = vmovl_s16(dx_d1);
              int32x4_t dy_wq0 = vmovl_s16(dy_d0);
              int32x4_t dy_wq1 = vmovl_s16(dy_d1);

              int32x4_t abs_q0 = vaddq_s32(vabsq_s32(dx_wq0), vabsq_s32(dy_wq0));
              int32x4_t abs_q1 = vaddq_s32(vabsq_s32(dx_wq1), vabsq_s32(dy_wq1));

              vst1q_s32(_mag + col_index, abs_q0);
              vst1q_s32(_mag + col_index + (kVectorSize / 2), abs_q1);
            }

            for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
              uint16_t col_index = scalar_index + vector_cols;
              _mag[col_index] = abs(_dx[col_index]) + abs(_dy[col_index]);
            }
#undef kVectorSize 8
          */
          
            for( j = 0; j < size.width; j++ )
                _mag[j] = abs(_dx[j]) + abs(_dy[j]);
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        _dy = (short*)(dy->data.ptr + dy->step*(i-1));

        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15L
            // 0.4142135623730950488016887242097 == tan(22.5 degrees)
            #define TG22  ((int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5))

            x = _dx[j];
            y = _dy[j];
            int s = (x ^ y) < 0 ? -1 : 1;
            int m = _mag[j];

            x = llabs(x);
            y = llabs(y);
            if( m > low )
            {
                int64_t tg22x = x * TG22;
                int64_t tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;

        for( j = 0; j < size.width; j++ )
            _dst[j] = (uchar)-(_map[j] >> 1);
    }
}

DMZ_INTERNAL void llcv_canny7(IplImage *src, IplImage *dst, double low_thresh, double high_thresh) {
  CvSize src_size = cvGetSize(src);

  IplImage *sobel_scratch = cvCreateImage(cvSize(src_size.height, src_size.width), IPL_DEPTH_16S, 1);
  IplImage *dx = cvCreateImage(src_size, IPL_DEPTH_16S, 1);
  IplImage *dy = cvCreateImage(src_size, IPL_DEPTH_16S, 1);
  llcv_sobel7(src, dx, sobel_scratch, 1, 0);
  llcv_sobel7(src, dy, sobel_scratch, 0, 1);
  cvReleaseImage(&sobel_scratch);

  llcv_canny7_precomputed_sobel(src, dst, dx, dy, low_thresh, high_thresh);

  cvReleaseImage(&dx);
  cvReleaseImage(&dy);
}

// Calculate sum of abs(image)
DMZ_INTERNAL double sum_abs_magnitude_c(IplImage *image) {
  IplImage *image_abs = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
  cvAbs(image, image_abs);
  CvScalar sum = cvSum(image_abs);
  cvReleaseImage(&image_abs);
  return sum.val[0];
}

DMZ_INTERNAL double sum_magnitude_c(IplImage *dx, IplImage *dy) {
  CvSize src_size = cvGetSize(dx);

  // Calculate the gradient magnitude
  IplImage *magnitude = cvCreateImage(src_size, IPL_DEPTH_32F, 1);

  IplImage *dx_float = cvCreateImage(src_size, IPL_DEPTH_32F, 1);
  IplImage *dy_float = cvCreateImage(src_size, IPL_DEPTH_32F, 1);
  cvConvertScale(dx, dx_float, 1);
  cvConvertScale(dy, dy_float, 1);

  cvCartToPolar(dx_float, dy_float, magnitude, NULL, true);

  cvReleaseImage(&dx_float);
  cvReleaseImage(&dy_float);

  CvScalar sum = cvSum(magnitude);

  cvReleaseImage(&magnitude);
  return sum.val[0];
}

DMZ_INTERNAL double sum_abs_magnitude_neon(IplImage *image) {
#if DMZ_HAS_NEON_COMPILETIME
#define kVectorSize 8
  CvSize image_size = cvGetSize(image);

  assert(image->depth == IPL_DEPTH_16S);
  assert(image->nChannels == 1);

  const uint8_t *origin = (uint8_t *)(image->imageData);
  if(dmz_unlikely(NULL != image->roi)) {
    origin += image->roi->yOffset * image->widthStep + image->roi->xOffset * sizeof(int16_t);
  }
  uint16_t image_width_step = (uint16_t)image->widthStep;

  uint16_t scalar_cols = (uint16_t)(image_size.width % kVectorSize);
  uint16_t vector_chunks = (uint16_t)(image_size.width / kVectorSize);

  uint16_t vector_cols = vector_chunks * kVectorSize;

  int32x4_t vector_sum = {0, 0, 0, 0};
  int32_t scalar_sum = 0;

  for(uint16_t row_index = 0; row_index < image_size.height; row_index++) {
    const int16_t *row_origin = (int16_t *)(origin + row_index * image_width_step);

    for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
      uint16_t col_index = vector_index * kVectorSize;
      const int16_t *image_data = row_origin + col_index;

      int16x8_t image_q = vld1q_s16(image_data);
      int16x8_t abs_image_q = vqabsq_s16(image_q);
      int16x4_t image_d0 = vget_low_s16(abs_image_q);
      int16x4_t image_d1 = vget_high_s16(abs_image_q);
      vector_sum = vaddw_s16(vector_sum, image_d0);
      vector_sum = vaddw_s16(vector_sum, image_d1);
    }

    for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
      uint16_t col_index = scalar_index + vector_cols;
      scalar_sum += abs((int32_t)(row_origin[col_index]));
    }
  }

  scalar_sum += vgetq_lane_s32(vector_sum, 0);
  scalar_sum += vgetq_lane_s32(vector_sum, 1);
  scalar_sum += vgetq_lane_s32(vector_sum, 2);
  scalar_sum += vgetq_lane_s32(vector_sum, 3);

  return scalar_sum;
#undef kVectorSize
#else
  return 0.0f;
#endif // DMZ_HAS_NEON_COMPILETIME
}

DMZ_INTERNAL double sum_magnitude_neon(IplImage *dx, IplImage *dy) {
#if DMZ_HAS_NEON_COMPILETIME
#define kVectorSize 8
  CvSize image_size = cvGetSize(dx);

  assert(dx->depth == IPL_DEPTH_16S && dy->depth == IPL_DEPTH_16S);
  assert(dx->width == dy->width && dx->height == dy->height);

  const uint8_t *dx_origin = (uint8_t *)(dx->imageData);
  const uint8_t *dy_origin = (uint8_t *)(dy->imageData);
  if(dmz_unlikely(NULL != dx->roi)) {
    dx_origin += dx->roi->yOffset * dx->widthStep + dx->roi->xOffset * sizeof(int16_t);
  }
  if(dmz_unlikely(NULL != dy->roi)) {
    dy_origin += dy->roi->yOffset * dy->widthStep + dy->roi->xOffset * sizeof(int16_t);
  }
  uint16_t dx_width_step = (uint16_t)dx->widthStep;
  uint16_t dy_width_step = (uint16_t)dy->widthStep;

  uint16_t scalar_cols = (uint16_t)(image_size.width % kVectorSize);
  uint16_t vector_chunks = (uint16_t)(image_size.width / kVectorSize);

  uint16_t vector_cols = vector_chunks * kVectorSize;
  float32x4_t vector_sum = {0.0f, 0.0f, 0.0f, 0.0f};
  int32x4_t ones = {1, 1, 1, 1};
  float32_t scalar_sum = 0.0f;
  
  for(uint16_t row_index = 0; row_index < image_size.height; row_index++) {
    const int16_t *dx_row_origin = (int16_t *)(dx_origin + row_index * dx_width_step);
    const int16_t *dy_row_origin = (int16_t *)(dy_origin + row_index * dy_width_step);

    for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
      uint16_t col_index = vector_index * kVectorSize;
      const int16_t *dx_data = dx_row_origin + col_index;
      const int16_t *dy_data = dy_row_origin + col_index;

      int16x8_t dx_q = vld1q_s16(dx_data);
      int16x8_t dy_q = vld1q_s16(dy_data);

      int16x4_t dx_d0 = vget_low_s16(dx_q);
      int16x4_t dx_d1 = vget_high_s16(dx_q);

      int16x4_t dy_d0 = vget_low_s16(dy_q);
      int16x4_t dy_d1 = vget_high_s16(dy_q);

      int32x4_t dx_d0_squared = vmull_s16(dx_d0, dx_d0);
      int32x4_t dx_d1_squared = vmull_s16(dx_d1, dx_d1);

      int32x4_t dy_d0_squared = vmull_s16(dy_d0, dy_d0);
      int32x4_t dy_d1_squared = vmull_s16(dy_d1, dy_d1);

      int32x4_t d0_summed = vhaddq_s32(dx_d0_squared, dy_d0_squared); // halving add to avoid overflow; will compensate for this later
      int32x4_t d1_summed = vhaddq_s32(dx_d1_squared, dy_d1_squared);

      // add 1 across the board to prevent calculating 1 / (0 ** 2).
      d0_summed = vaddq_s32(d0_summed, ones);
      d1_summed = vaddq_s32(d1_summed, ones);
      
      float32x4_t d0_summed_float = vcvtq_f32_s32(d0_summed);
      float32x4_t d1_summed_float = vcvtq_f32_s32(d1_summed);

      float32x4_t approx_inv_root_d0_summed = vrsqrteq_f32(d0_summed_float);
      float32x4_t approx_inv_root_d1_summed = vrsqrteq_f32(d1_summed_float);

      float32x4_t approx_root_d0_summed = vmulq_f32(d0_summed_float, approx_inv_root_d0_summed);
      float32x4_t approx_root_d1_summed = vmulq_f32(d1_summed_float, approx_inv_root_d1_summed);

      vector_sum = vaddq_f32(approx_root_d0_summed, vector_sum);
      vector_sum = vaddq_f32(approx_root_d1_summed, vector_sum);
    }

    for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
      uint16_t col_index = scalar_index + vector_cols;
      int16_t dx_val = dx_row_origin[col_index];
      int16_t dy_val = dy_row_origin[col_index];
      // halve to prevent overflow. will compensate for this at the end.
      int32_t dx_val_squared_halved = (dx_val * dx_val) >> 1;
      int32_t dy_val_squared_halved = (dy_val * dy_val) >> 1;
      float32_t magnitude = sqrtf(dx_val_squared_halved + dy_val_squared_halved);
      scalar_sum += magnitude;
    }
  }

  scalar_sum += vgetq_lane_f32(vector_sum, 0);
  scalar_sum += vgetq_lane_f32(vector_sum, 1);
  scalar_sum += vgetq_lane_f32(vector_sum, 2);
  scalar_sum += vgetq_lane_f32(vector_sum, 3);
  
  // sqrtf(2) is to compensate for the halving adds
  return scalar_sum * sqrtf(2.0f);
#undef kVectorSize
#else
  return 0.0f;
#endif // DMZ_HAS_NEON_COMPILETIME
}

#define TEST_SUM_MAGNITUDE_NEON 0

DMZ_INTERNAL double sum_magnitude(IplImage *dx, IplImage *dy) {
  if(dmz_has_neon_runtime()) {
    double neon_ret = sum_magnitude_neon(dx, dy);
#if TEST_SUM_MAGNITUDE_NEON
    double c_ret = sum_magnitude_c(dx, dy);
    fprintf(stderr, "sum_magnitude C: %f, NEON: %f, DELTA: %f (%f %%)\n", c_ret, neon_ret, c_ret - neon_ret, 100.0f * (c_ret - neon_ret) / c_ret);
#endif
    return neon_ret;
  } else {
    return sum_magnitude_c(dx, dy);
  }
}

#define TEST_SUM_ABS_MAGNITUDE_NEON 0

// TODO: The NEON implementation will be faster if we pass it both images at
// once and let it interleave memory accesses and calculations.
DMZ_INTERNAL double sum_abs_magnitude(IplImage *image) {
    if(dmz_has_neon_runtime()) {
      double neon_ret = sum_abs_magnitude_neon(image);
#if TEST_SUM_ABS_MAGNITUDE_NEON
      double c_ret = sum_abs_magnitude_c(image);
      fprintf(stderr, "sum_abs_magnitude C: %f, NEON: %f, DELTA: %f (%f %%)\n", c_ret, neon_ret, c_ret - neon_ret, 100.0f * (c_ret - neon_ret) / c_ret);
#endif
      return neon_ret;
    } else {
      return sum_abs_magnitude_c(image);
    }
}

DMZ_INTERNAL void llcv_adaptive_canny7_precomputed_sobel(IplImage *src, IplImage *dst, IplImage *dx, IplImage *dy) {
  CvSize src_size = cvGetSize(src);
  // We can use either sum_abs_magnitude (|dx| + |dy|) or sum_magnitude (sqrt(dx^2 + dy^2)) here. They yield
  // comparable results, and sum_abs_magnitude is marginally faster to compute, and can be made faster
  // still than our current implementation.
  double mean = (sum_abs_magnitude(dx) + sum_abs_magnitude(dy)) / (src_size.width * src_size.height);
  // double mean = sum_magnitude(dx, dy) / (src_size.width * src_size.height);

  double low_threshold = mean;
  double high_threshold = 3.0f * low_threshold;

  llcv_canny7_precomputed_sobel(src, dst, dx, dy, low_threshold, high_threshold);
}

#endif
