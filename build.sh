rm -rf build
mkdir build
cd build
make clean
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRITON_PYTORCH_DOCKER_IMAGE=nvcr.io/nvidia/pytorch:22.08-py3  \
    -DCONTROLLER_TYPE="deepspeed" ..
make install