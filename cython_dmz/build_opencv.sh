#!/bin/bash

# Use this to build opencv.

if [ -z $1 ]; then
  echo "usage: build_opencv.sh <opencv source directory>"
  exit 1
fi

if [ ! -e "dmz.pyx" ]; then
  echo "build_opencv.sh is meant to be run from the dmz subdirectory of the clearcut project"
  exit 1
fi

PROJECT_PATH="`pwd`"

SRC_DIRNAME=`dirname "$1"`
SRC_BASENAME=`basename "$1"`
SRC_PATH="`cd \"$SRC_DIRNAME\" 2>/dev/null && pwd || echo \"$SRC_DIRNAME\"`/$SRC_BASENAME"

INTERMEDIATE_PATH="`mktemp -d -t opencv_build`"
echo "Using temp dir $INTERMEDIATE_PATH"
OPENCV_INSTALL_DIR=$INTERMEDIATE_PATH/install

LOG_DIR=$INTERMEDIATE_PATH/build_logs
echo "Logging build results to $LOG_DIR"
mkdir -p $LOG_DIR

echo "Configuring"
cd $INTERMEDIATE_PATH
CONFIGURE_LOG=$LOG_DIR/configure.log
cmake -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_DIR \
      -DENABLE_SSE=YES \
      -DENABLE_SSE2=YES \
      -DBUILD_TESTS=OFF \
      -DBUILD_SHARED_LIBS=NO \
      -DBUILD_EXAMPLES=NO \
      -DWITH_EIGEN2=NO \
      -DWITH_PVAPI=NO \
      -DWITH_OPENEXR=NO \
      -DWITH_QT=NO \
      -DWITH_QUICKTIME=NO \
      -G Xcode $SRC_PATH > $CONFIGURE_LOG 2>&1

cd $INTERMEDIATE_PATH

echo "Building"
MACOSX_LOG=$LOG_DIR/mac_os_x.log
xcodebuild -sdk macosx -configuration Release -target install -target opencv_core -target opencv_imgproc -parallelizeTargets > $MACOSX_LOG 2>&1

echo "Built to $OPENCV_INSTALL_DIR"

echo "Copying into project"
cp -R $OPENCV_INSTALL_DIR/include/opencv2 $PROJECT_PATH/opencv/include
for LIB in libopencv_core.a	libopencv_imgproc.a
do
  cp $OPENCV_INSTALL_DIR/lib/$LIB $PROJECT_PATH/opencv/lib
done

for LIB in libopencv_lapack.a
do
  cp $OPENCV_INSTALL_DIR/share/opencv/3rdparty/lib/$LIB $PROJECT_PATH/opencv/lib
done

# TODO: Clean up temp dir
# rm -rf $INTERMEDIATE
