//
//  neon.h
//  See the file "LICENSE.md" for the full license governing this code.
//

// Contains NEON-related constants

#ifndef DMZ_NEON_H
#define DMZ_NEON_H

#define kQRegisterBits 128
#define kQRegisterElements8 16 // == (kQRegisterBits / sizeof(uint8_t))
#define kQRegisterElements16 8 // == (kQRegisterBits / sizeof(uint16_t))
#define kQRegisterElements32 4 // == (kQRegisterBits / sizeof(float32_t))

#endif
