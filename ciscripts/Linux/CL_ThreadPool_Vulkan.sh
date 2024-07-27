./schema/generate.sh
mkdir linuxbuild
cd linuxbuild
cmake ../ -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_SUPPORT_TFLITE_QUAN=ON -DMNN_BUILD_TEST=ON -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_OPENMP=OFF -DMNN_USE_THREAD_POOL=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_SEP_BUILD=OFF
make -j8
