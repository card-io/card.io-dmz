//
//  rand.h
//  See the file "LICENSE.md" for the full license governing this code.
//

#ifndef icc_rand_h
#define icc_rand_h

#include "dmz_macros.h"

// Seed the Mersenne Twister random function
DMZ_INTERNAL void dmz_mersenne_twister_seed(unsigned long s);

// Generate a random unsinged 32-bit long
DMZ_INTERNAL unsigned long dmz_mersenne_twister_rand32();

#endif
