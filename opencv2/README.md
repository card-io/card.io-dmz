This `opencv2` directory contains a copy of the OpenCV header files.

When building the iOS or Android versions of card.io, this directory is not actually needed. Each of those projects contains its own `opencv_device` directory, which includes these header files as well as the relevant OpenCV static libraries built for that platform.

But if you are trying to use card.io's `dmz` project in a different context, such as building card.io for a platform other than iOS or Android, these header files might come in handy.

In that case, however, you will still need to build your own platform-specific OpenCV static libraries. Specifically these two: `libopencv_core.a` and `libopencv_imgproc.a`.

For some possibly helpful (though possibly not at all helpful) hints on downloading and building OpenCV, see https://github.com/card-io/card.io-iOS-source/blob/master/opencv_device/doc/notes.txt
