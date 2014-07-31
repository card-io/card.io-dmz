Core, shared client-side image processing code.

Some notes on writing dmz code:

    * For IP protection, any function name, struct name, etc. that could be sensitive should be wrapped with DMZ_MANGLE_NAME. It preprocesses down to nothing in normal use, but marks that symbol (function name, struct name, etc.) as needing mangling during production builds. Any given symbol need only be marked once. Do not mark third party symbols (e.g. Eigen, OpenCV) for mangling. Symbols that will not be externed (e.g. static functions in implementations) do not need to be mangled, but it doesn't hurt anything.
    * To generate the mangling preprocessor instruction file, run `fab mangle` from this directory.
    * All functions that do not *need* to be called from outside the dmz should be declared and implemented prefixed by DMZ_INTERNAL. (This should be as much of the dmz as possible.)
    * If you add a new implementation file to the dmz, run `fab concat` to regenerate `dmz_all.cpp`, which is the only file that actually gets compiled. (Again, this is for ip protection reasons.)

If you're a client of the dmz, you should only ever compile `dmz_all.cpp`. You should run `fab mangle` and `fab concat` before every build (they're fast).

