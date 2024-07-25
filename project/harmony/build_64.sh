#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$HARMONY_HOME/native/build/cmake/ohos.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DOHOS_ARCH="arm64-v8a" \
-DOHOS_STL=c++_static \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_USE_SSE=OFF \
-DMNN_SUPPORT_BF16=OFF \
-DMNN_BUILD_TEST=ON \
-DOHOS_PLATFORM_LEVEL=9  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

make -j4
