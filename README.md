Core, shared client-side image processing code.

Some notes on writing dmz code:

* All functions that do not *need* to be called from outside the dmz should be declared and implemented prefixed by DMZ_INTERNAL. (This should be as much of the dmz as possible.)
* If you add a new implementation file to the dmz, run `fab concat` to regenerate `dmz_all.cpp`, which is the only file that actually gets compiled.

If you're a client of the dmz, you should only ever compile `dmz_all.cpp`. You should run `fab concat` before every build (they're fast).

