export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"

#!/bin/bash

set -e
set -x

sudo apt-get update -qq
sudo apt-get install -qq software-properties-common

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"


sudo apt-get update -qq
sudo apt-get install -qq cuda-toolkit-11-6

export INCLUDE_PATH="$INCLUDE_PATH;$PWD/usr/local/cuda/include"

export CUDA_PATH="/usr/local/cuda"
export CUDA_PATH="$CUDA_PATH:/etc/alternatives/cuda"
export CUDA="/usr/local/cuda"
export CUDA="$CUDA:/etc/alternatives/cuda"

#echo "CUDA_PATH=${CUDA_PATH}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

which nvcc
ls -l /usr/local 
ls -l /usr/local/cuda
ls -lH /usr/local/cuda
ls -lH /usr/local/cuda/include
nvcc --version

#git clone https://github.com/StanfordLegion/legion.git
#CC=gcc CXX=g++ DEBUG=1 USE_GASNET=0 ./legion/language/scripts/setup_env.py
#./install.py
#./legion/language/regent.py test/fft_test.rg