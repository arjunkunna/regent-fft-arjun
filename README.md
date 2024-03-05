# Regent-FFT

This is a fast fourier transform library built in Regent.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

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

1. SSH into sapling: ssh <username>@sapling.stanford.edu
2. module load slurm mpi cmake cuda llvm
3. srun -n 1 -N 1 -c 40 -p gpu --exclusive --pty bash --login
4. <navigate to your .rg file>
5. source env.sh
6. ../legion/language/regent.py test/fft_test.rg 
``


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
