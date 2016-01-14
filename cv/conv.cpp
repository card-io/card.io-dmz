//
//  conv.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include <stdint.h>
#include <assert.h>

#include "compile.h"
#if COMPILE_DMZ

#include "conv.h"
#include "processor_support.h"

#if DMZ_HAS_NEON_COMPILETIME

#include <arm_neon.h>

#define kChunkSize 4

DMZ_INTERNAL void llcv_conv_3x3_f32_row(const float *inrow0, const float *inrow1, const float *inrow2, float *kernel3x4, float *outrow, uint16_t length) {
  assert(length > 0);
  assert(length % kChunkSize == 0);

  uint16_t n_chunks = length / kChunkSize;
  
  asm
  (
   // load kernel vectors, do other setup
   "mov r0, %[ker]" "\n\t"
   "vld1.32 {q0}, [r0]!" "\n\t"
   "vld1.32 {q1}, [r0]!" "\n\t"
   "vld1.32 {q2}, [r0]!" "\n\t"
   
   "mov r0, %[in0]" "\n\t"
   "mov r1, %[in1]" "\n\t"
   "mov r2, %[in2]" "\n\t"
   "mov r3, %[out]" "\n\t"
   
   "mov r4, %[len]" "\n\t"
   
   // loop beginning
   "1:" "\n\t"
   
   // load image inputs (four of them)
   "vld1.32 {q3}, [r0]" "\n\t"
   "add r0, r0, #4" "\n\t"
   "vld1.32 {q6}, [r0]" "\n\t"
   "add r0, r0, #4" "\n\t"
   "vld1.32 {q9}, [r0]" "\n\t"
   "add r0, r0, #4" "\n\t"
   "vld1.32 {q12}, [r0]" "\n\t"
   "add r0, r0, #4" "\n\t"
   
   "vld1.32 {q4}, [r1]" "\n\t"
   "add r1, r1, #4" "\n\t"
   "vld1.32 {q7}, [r1]" "\n\t"
   "add r1, r1, #4" "\n\t"
   "vld1.32 {q10}, [r1]" "\n\t"
   "add r1, r1, #4" "\n\t"
   "vld1.32 {q13}, [r1]" "\n\t"
   "add r1, r1, #4" "\n\t"
   
   "vld1.32 {q5}, [r2]" "\n\t"
   "add r2, r2, #4" "\n\t"
   "vld1.32 {q8}, [r2]" "\n\t"
   "add r2, r2, #4" "\n\t"
   "vld1.32 {q11}, [r2]" "\n\t"
   "add r2, r2, #4" "\n\t"
   "vld1.32 {q14}, [r2]" "\n\t"
   "add r2, r2, #4" "\n\t"
   
   // multiply/accumulate
   "vmul.f32  q3,  q3, q0" "\n\t"
   "vmul.f32  q6,  q6, q0" "\n\t"
   "vmul.f32  q9,  q9, q0" "\n\t"
   "vmul.f32 q12, q12, q0" "\n\t"
   
   "vmla.f32  q3,  q4, q1" "\n\t"
   "vmla.f32  q6,  q7, q1" "\n\t"
   "vmla.f32  q9, q10, q1" "\n\t"
   "vmla.f32 q12, q13, q1" "\n\t"
   
   "vmla.f32  q3,  q5, q2" "\n\t"
   "vmla.f32  q6,  q8, q2" "\n\t"
   "vmla.f32  q9, q11, q2" "\n\t"
   "vmla.f32 q12, q14, q2" "\n\t"
   
   // pairwise addition for first reduction
   "vpadd.f32  d8,  d6,  d7" "\n\t" //  q3 ==  d6 /  d7
   "vpadd.f32 d14, d12, d13" "\n\t" //  q6 == d12 / d13
   "vpadd.f32 d20, d18, d19" "\n\t" //  q9 == d18 / d19
   "vpadd.f32 d26, d24, d25" "\n\t" // q12 == d24 / d25
   
   // pairwise add for second reduction, including a junk register for the second half
   "vpadd.f32 d10,  d8,  d9" "\n\t"
   "vpadd.f32 d16, d14, d15" "\n\t"
   "vpadd.f32 d22, d20, d21" "\n\t"
   "vpadd.f32 d28, d26, d27" "\n\t"
   
   // the desired float is now the first element of the d registers; store them
   "vst1.32 {d10[0]}, [r3]" "\n\t"
   "add r3, r3, #4" "\n\t"
   "vst1.32 {d16[0]}, [r3]" "\n\t"
   "add r3, r3, #4" "\n\t"
   "vst1.32 {d22[0]}, [r3]" "\n\t"
   "add r3, r3, #4" "\n\t"
   "vst1.32 {d28[0]}, [r3]" "\n\t"
   "add r3, r3, #4" "\n\t"

   // count down n_chunks by one, loop if not zero
   "subs r4, r4, #1" "\n\t"
   "bne 1b" "\n\t"
   
   : // output
   
   : // input
   [ker]"r" (kernel3x4),
   [in0]"r" (inrow0),
   [in1]"r" (inrow1),
   [in2]"r" (inrow2),
   [out]"r" (outrow),
   [len]"r" (n_chunks)
   
   : // clobbered
   "r0", "r1", "r2", "r3", "r4", // absurd that I still can't figure out how to do read/write registers. grrr.
   "q0", "q1", "q2", "q3",
   "q4", "q5", "q6", "q7",
   "q8", "q9", "q10", "q11",
   "q12", "q13", "q14",
   "memory", "cc"
   );
}

#else // !DMZ_HAS_NEON_COMPILETIME

DMZ_INTERNAL void llcv_conv_3x3_f32_row(const float *inrow0, const float *inrow1, const float *inrow2, float *kernel3x4, float *outrow, uint16_t length) {
  assert(false); // not implemented, per documentation in header
}

#endif // DMZ_HAS_NEON_COMPILETIME

#endif // COMPILE_DMZ
