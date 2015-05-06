Camera resolution
-----------------

card.io currently uses a camera resolution of 640 x 480 pixels. Much higher resolutions are available on most mobile devices, and a higher-resolution starting image would give us many more pixels per character, which would presumably significantly improve our categorization results.

So why we do we use 640x480?

* At the time this project began, 640x480 was the only camera resolution that was available across all devices. [That might still be true.](http://stackoverflow.com/questions/4486143/supported-camera-preview-sizes-for-popular-android-handsets)
* We have since experimented with higher resolutions, and as a result some of the overall card.io pipeline is now resolution-agnostic. However, the segmentation and categorization stages remain hard-coded to assume images derived from a 640x480 starting resolution. Eliminating these assumptions would require a fair amount of additional work.
* Carrying higher resolution images through the entire pipeline would cause major increases in memory consumption and major decreases in performance. Of course, hardware keeps getting bigger and faster, so these issues might prove tractable.
* Our training images were all gathered at 640x480. This is a huge impediment! Collecting a fresh set of tens of thousands of training images would require considerable time and effort.
* Our existing deep-learning models were all designed for 640x480 resolution. Switching to a higher resolution would mean training new models from scratch -- a trial-and-error process that would probably take several months.
