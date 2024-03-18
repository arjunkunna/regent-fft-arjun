git clone https://github.com/StanfordLegion/legion.git
cd legion/language
./install.py --rdir=auto --debug
cd ../..
legion/language/regent.py fft_test.rg