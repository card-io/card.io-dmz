[![card.io logo](Resources/cardio_logo_220.png "card.io")](https://www.card.io)

### card.io-dmz is intended to be used as a submodule of the main card.io SDK, [card.io-iOS-source](https://github.com/card-io/card.io-iOS-source) or [card.io-Android-source](https://github.com/card-io/card.io-Android-source).

#### The card.io-dmz submodule includes the core, client-side image processing code which is shared between iOS and Android.

As with the main **card.io** source repos, this repo does not yet contain is much in the way of documentation. :crying_cat_face: So please feel free to ask any questions by creating github issues -- we'll gradually build our documentation based on the discussions there.

Note that this is actual production code, which has been iterated upon by multiple developers over several years. If you see something that could benefit from being tidied up, rewritten, or otherwise improved, your Pull Requests will be welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Brought to you by  
[![PayPal logo](Resources/pp_h_rgb.png)](https://paypal.com/ "PayPal")

 
Why "dmz?"
---------
`dmz` stands for "demilitarized zone" -- code that is not platform-specific to iOS, Android, nor any other OS.

Some platform specific code did sneak in, but you'll note that it is in files called `mz`. :smile_cat: A `dmz_context` structure allows each platform to associate its specific data as needed.


Some notes on writing dmz code
------------------------------

* All functions that do not *need* to be called from outside the dmz should be declared and implemented prefixed by DMZ_INTERNAL. (This should be as much of the dmz as possible.)
* If you add a new implementation file to the dmz, run `fab concat` to regenerate `dmz_all.cpp`, which is the only file that actually gets compiled.

If you're a client of the dmz, you should only ever compile `dmz_all.cpp`. You should run `fab concat` before every build.


Contributors
------------

**card.io** was created by [Josh Bleecher Snyder](https://github.com/josharian/).

Subsequent help has come from [Brent Fitzgerald](https://github.com/burnto/), [Tom Whipple](https://github.com/tomwhipple), [Dave Goldman](https://github.com/dgoldman-ebay), and [Roman Punskyy](https://github.com/romk1n).

And from **you**! Pull requests and new issues are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
