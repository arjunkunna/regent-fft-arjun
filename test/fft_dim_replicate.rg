import "regent"

local fft = require("fft")

local cmapper = require("test_mapper")
local format = require("std/format")

--Import cuFFT API
local cufft_c = terralib.includec("cufftXt.h")
terralib.linklibrary("libcufft.so")

local fft3d_batch = fft.generate_fft_interface(int3d, complex64, complex64)


--demand(__inline)
task test3d_batch()
  -- Initialize input and output arrays
  var r = region(ispace(int3d, { 4, 5, 6 }), complex64)
  var s = region(ispace(int3d, { 4, 5, 6 }), complex64)
  
  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  -- Initialize output array
  fill(s, 0)

  -- Create plan region and call batch_dft
  var p = region(ispace(int1d, 1), fft3d_batch.plan)
  fft3d_batch.batch_dft(r, s, p)
end


-- Main function
task main()
    test3d_batch()
end

--Include mapper
regentlib.start(main, cmapper.register_mappers)