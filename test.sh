export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"

export CUDA_PATH=$PWD/usr/local/cuda
export CUDA=$PWD/usr/local/cuda

echo "CUDA_PATH=${CUDA_PATH}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"


# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding CUDA to CUDA_PATH, PATH and LD_LIBRARY_PATH"
    echo "CUDA_PATH=${CUDA_PATH}" >> $GITHUB_ENV
    echo "${CUDA_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi

git clone https://github.com/StanfordLegion/legion.git
CC=gcc CXX=g++ DEBUG=1 USE_GASNET=0 ./legion/language/scripts/setup_env.py
./install.py
./legion/language/regent.py test/fft_test.rg

