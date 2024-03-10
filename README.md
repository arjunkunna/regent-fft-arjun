# Regent-FFT

This is a fast fourier transform library built in Regent.


## Description

At a high level, the library takes the input matrix for the DFT in the form of a region, and saves the output in an output region.

The library currently supports transforms up to 3 dimensions, and can be configured to run on either a CPU or a GPU.

The CPU mode is supported by [FFTW](https://www.fftw.org/), and the GPU mode by [cuFFT](https://developer.nvidia.com/cufft).

Both Complex-to-Complex and Real-To-Complex transformations are supported.

Both complex64 and complex32 types are supported in GPU mode, CPU mode is only able to support complex64. It is possible to use complex32 in CPU mode but it requires some additional setup - please contact me if that is of interest.

## Getting Started

### Installing

First, clone the repo: 
```
git clone https://github.com/arjunkunna/regent-fft-arjun.git
```

Run the install script and add environment variables
```
./install.py
source env.sh
```

Then, run your .rg script, which can be set up using the instructions in the 'Executing Program' section
```
../legion/language/regent.py test/fft_test.rg 
```

If operating in sapling, here are some startup instructions:
```
ssh <username>@sapling.stanford.edu
module load slurm mpi cmake cuda llvm
srun -n 1 -N 1 -c 40 -p gpu --exclusive --pty bash --login
<navigate to your .rg file>
source env.sh
../legion/language/regent.py test/fft_test.rg 
```

## Usage

### Executing program

There are 4 possible modes:
1. GPU vs. CPU
2. Complex-to-Complex vs. Real-to-Complex
3. Float vs. Double (Float only supported in GPU mode)

API usage generally follows the following steps. 

First, an FFT interface has to be generated depending on the type of transform you hope to do. Then, we a) create a plan, b) execute said plan, and then c) destroy the plan once we are done. 
There are several sample code snippets in the `fft_test.rg` file for reference as well. 

1. Link the fft.rg file and generate an interface.
   *The first argument is the dimension - `int1d`, `int2d`, or `int3d`
   *The second argument is the data type of the input - `complex64`, `complex32`, `real`, or `double`
   *The third argument is the data type of the input - `complex64` or `complex32`
```
local fft = require("fft")
local fft1d = fft.generate_fft_interface(int1d, complex64, complex64)
```

2. Make a plan

Make_plan takes three arguments: 1. Our input region, `r` 2. Our output region, `s` 3. Our plan region, `p`

```
  fft1d.make_plan(r, s, p)
```

The input region should be initialized with index space of form ispace(<type>, N), where N is the size of the array, and <type> is either int1d/int2d/int3d depending on the dimension of the transform.
The fieldspace of the region is the type supported by the transform - e.g, in a real-to-complex transform with doubles, the input array will have fieldspace double and output array will have fieldspace complex64

```
var r = region(ispace(int1d, 3), double)
var s = region(ispace(int1d, 3), complex64)
```

The plan region always takes the following form, with fieldspace fft.plan

```
 var p = region(ispace(int1d, 1), fft1d.plan)
```
make_plan is a __demand(__inline) task. This means that if the user wants it to execute it in a separate task, they must wrap the task themselves

3. Execute the plan
Next, we execute the plan. This takes the same 3 regions as mentioned above. 

```
   fft1d.execute_plan_task(r, s, p)
```

Note that execute_plan is a __demand(__inline) task (similar to make_plan above). The task execute_plan_task is simply a wrapper around execute_plan for convenience, to avoid needing to define this explicitly.
Important: because execute_plan is a __demand(__inline) task, it will never execute on the GPU (unless the parent task is running on the GPU). Therefore, in most cases it is necessary to use execute_plan_task if one wants to use the GPU.

4 Destroy the plan

When a plan is no longer needed it can be destroyed.
```  
  fft1d.destroy_plan(p)
```

## Authors

* Elliott Slaughter (slaughter@cs.stanford.edu)
* Arjun Kunna (arjunkunna@gmail.com)

## Version History

* 1.0
    * Initial Release - Supports single-GPU transforms for 1, 2, and 3d. Real-to-complex and Complex-to-Complex. 

## Additional Resources
* For information on Regent, please refer to [the Regent website](https://regent-lang.org/)
* For inforation on the FFT transform, please refer to [these set of notes](https://web.stanford.edu/class/cs168/l/l15.pdf) or [this course](https://see.stanford.edu/Course/EE261)

## Acknowledgments

* [FFTW](https://www.fftw.org/)
* [cuFFT](https://developer.nvidia.com/cufft)
* [Regent](https://regent-lang.org/)

