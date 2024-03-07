# Regent-FFT

This is a fast fourier transform library built in Regent.

For information on Regent, please refer to https://regent-lang.org/
For inforation on FFT, please refer to https://web.stanford.edu/class/cs168/l/l15.pdf or https://see.stanford.edu/Course/EE261

## Description


The library currently supports transforms up to 3 dimensions, and can be configured to run on either a CPU or a GPU.

The CPU mode is supported by FFTW, and the GPU mode by cuFFT

Both Complex-to-Complex and Real-To-Complex transformations are supported.

Both complex64 and complex32 types are supported in GPU mode, but only complex64 in CPU mode. It is possible to use complex32 in CPU mode but it requires some additional setup - please contact me if that is of interest.

## Getting Started

### Installing

First, clone the repo: 
```
git clone https://github.com/arjunkunna/regent-fft-arjun.git
```

Run the install script:
```
./install.py
```

If operating in sapling, these are the instructions:

1. SSH into sapling
```
ssh <username>@sapling.stanford.edu
```
module load slurm mpi cmake cuda llvm
srun -n 1 -N 1 -c 40 -p gpu --exclusive --pty bash --login
<navigate to your .rg file>
source env.sh
../legion/language/regent.py test/fft_test.rg 
```

## Usage

### Executing program

* How to run the program
* Step-by-step bullets\


There are 4 possible modes:
1. GPU vs. CPU
1. Complex-to-Complex vs. Real-to-Complex
2. Float vs. Double (Only supported in GPU mode)


1. Generate an interface. The first argument is the dimension - int1d, int2d, or int3d
2. The second argument is the data type of the input - complex64, complex32, real, or double
3. The third argument is the data type of the input - complex64 or complex32

```
local fft1d = fft.generate_fft_interface(int1d, complex64, complex64)
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
