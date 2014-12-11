# Contribute to the card.io core vision library.

### *Pull requests are welcome!*


General Guidelines
------------------

* **Code style.** Please follow local code style. Ask if you're unsure. 
* **No warnings.** All generated code must compile without warnings. All generated code must pass the XCode static analyzer.
* **Cross-platform.** This module is used by both iOS & Android. With the exception of some minimum wrappers, platform specific code SHOULD generally live with that platform's SDK. Anything platform specific MUST be guarded by appropriate compile macros.
* **Architecture support.** The library should support armv7, armv7s, arm64, i386 and x86_64. armv7 must support the lack of NEON vector instructions. (It is acceptable & encouraged to use NEON provided an alternative is available. Selection SHOULD be controlled by a compile time macro.)
* **ARC agnostic.** No code that depends on the presence or absence of ARC should appear in public header files.
* **No logging in Release builds.** Always use one of the 'dmz_*_log()'' macros provided in 'dmz_debug.h'
* **Testing.** Compile for both iOS and Android, wherever possible. Test on at least one physical device.
