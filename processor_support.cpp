//  See the file "LICENSE.md" for the full license governing this code.

#include "compile.h"
#if COMPILE_DMZ

/*
 * About processor support:
 *
 * http://wiki.debian.org/ArmHardFloatPort/VfpComparison has much useful information about ARM devices.
 *
 * Information from the Android docs have also been included here, as it contains useful info about the ARM family,
 * even though all modern iOS devecs have NEON.
 *
 */

#include "processor_support.h"

#if ANDROID_DMZ
// use runtime checks.
#include <cpu-features.h>
#include <stdint.h>

enum {
  AndroidProcessorUnknown = 0,
  AndroidProcessorHasNeon = 1,
  AndroidProcessorNoSupport = 2,
  AndroidProcessorHasVFP3_16 = 3,
};
typedef uint8_t AndroidProcessorSupport;

static AndroidProcessorSupport androidProcessor = AndroidProcessorUnknown;

AndroidProcessorSupport get_android_processor_support(void) {
    // Note that updates to this static variable are idempotent, so no thread protection required
    if(androidProcessor == AndroidProcessorUnknown) {

      // default to no support
      androidProcessor = AndroidProcessorNoSupport;

      // it is important to check the CPU family before the features, since the results will collide if called on
      // an X86 processor.
      if (android_getCpuFamily() == ANDROID_CPU_FAMILY_ARM) {

          uint64_t cpuFeatures = android_getCpuFeatures();
          if (cpuFeatures & ANDROID_CPU_ARM_FEATURE_NEON) {
            /* From android-ndk-r8/docs/CPU-FEATURES.html:
             *
             * ANDROID_CPU_ARM_FEATURE_NEON
             * Indicates that the device's CPU supports the ARM Advanced SIMD
             * (a.k.a. NEON) vector instruction set extension. Note that ARM
             * mandates that such CPUs also implement VFPv3-D32, which provides
             * 32 hardware FP registers (shared with the NEON unit).
             */
              androidProcessor = AndroidProcessorHasNeon;
          }
          else if (cpuFeatures & ANDROID_CPU_ARM_FEATURE_VFPv3) {
            /* From android-ndk-r8/docs/CPU-FEATURES.html:
             *
             * ANDROID_CPU_ARM_FEATURE_ARMv7
             * Indicates that the device's CPU supports the ARMv7-A instruction
             * set as supported by the "armeabi-v7a" abi (see CPU-ARCH-ABIS.html).
             * This corresponds to Thumb-2 and VFPv3-D16 instructions.
             *
             * ANDROID_CPU_ARM_FEATURE_VFPv3
             * Indicates that the device's CPU supports the VFPv3 hardware FPU
             * instruction set extension. Due to the definition of 'armeabi-v7a',
             * this will always be the case if ANDROID_CPU_ARM_FEATURE_ARMv7 is
             * returned.
             *
             * Note that this corresponds to the minimum profile VFPv3-D16 that
             * _only_ provides 16 hardware FP registers.
             */
              androidProcessor = AndroidProcessorHasVFP3_16;
          }

      } else if(android_getCpuFamily() == ANDROID_CPU_FAMILY_ARM64
             || android_getCpuFamily() == ANDROID_CPU_FAMILY_X86_64) {
          // arm64 bit is NEON by definition, but requires new asm to compile.
          // See https://github.com/card-io/card.io-dmz/pull/20
          androidProcessor = AndroidProcessorHasVFP3_16;
      }
      dmz_debug_log("androidProcessor: %i", androidProcessor);
    }
    return androidProcessor;
}

int dmz_has_neon_runtime(void) {
    return (get_android_processor_support() == AndroidProcessorHasNeon);
}

int dmz_use_vfp3_16(void) {
    return (get_android_processor_support() == AndroidProcessorHasVFP3_16);
}

static int glesWarpAllowed = ANDROID_USE_GLES_WARP;
void dmz_set_gles_warp(int newstate) {
	glesWarpAllowed = ANDROID_USE_GLES_WARP & newstate;
}

int dmz_use_gles_warp(void) {
	return glesWarpAllowed;
}

#elif DMZ_HAS_NEON_COMPILETIME // not ANDROID_DMZ (i.e. iOS)

int dmz_has_neon_runtime(void) {return 1;}
int dmz_use_vfp3_16(void) {return 0;}
int dmz_use_gles_warp(void) {return 1;}

#else // not iOS, not Android. (i.e CYTHON)

int dmz_has_neon_runtime(void) {return 0;}

// technically we could use the vfpv3-d16 on any ARMv7 device,
// but in practice the only non-NEON ARMv7 devices we will see are Android.
int dmz_use_vfp3_16(void) {return 0;}

int dmz_use_gles_warp(void) {return 0;}

#endif


#endif // COMPILE_DMZ
