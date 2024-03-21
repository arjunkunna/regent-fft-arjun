export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"
export INCLUDE_PATH="$INCLUDE_PATH:${CUDA_HOME}/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${CUDA_HOME}/lib64"
git clone https://github.com/StanfordLegion/legion.git
CC=gcc CXX=g++ DEBUG=1 USE_GASNET=0 ./legion/language/scripts/setup_env.py

