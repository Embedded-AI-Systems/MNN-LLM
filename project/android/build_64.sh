#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_USE_SSE=OFF \
-DMNN_BUILD_TEST=ON \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 $4 $5 $6 $7

make -j4
