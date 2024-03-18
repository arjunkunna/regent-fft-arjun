export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"
source env.sh
git clone https://github.com/StanfordLegion/legion.git
CC=gcc CXX=g++ DEBUG=1 USE_GASNET=0 ./legion/language/scripts/setup_env.py
./install.py
./legion/language/regent.py test/fft_test.rg