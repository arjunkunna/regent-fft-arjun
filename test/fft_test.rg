import "regent"

local fft = require("fft")

local cmapper = require("test_mapper")
local format = require("std/format")

--Import cuFFT API
local cufft_c = terralib.includec("cufftXt.h")
terralib.linklibrary("libcufft.so")


-- Task to print out input or output array with fieldspace complex64. Takes a region and a string representing the name of the array
__demand(__inline, __leaf)
task print_array(input : region(ispace(int1d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("\n{} = [", arrayName)
  for x in input do
    var currComplex = input[x]
    format.println("{} + {}j, ", currComplex.real, currComplex.imag)
  end
  format.println("]\n")
end

-- Task to print out input or output array with fieldspace complex64. Takes a region and a string representing the name of the array
__demand(__inline, __leaf)
task print_array_float(input : region(ispace(int1d), complex32), arrayName: rawstring)
where reads (input) do
  format.println("\n{} = [", arrayName)
  for x in input do
    var currComplex = input[x]
    format.println("{} + {}j, ", currComplex.real, currComplex.imag)
  end
  format.println("]\n")
end

task print_array_2d(input : region(ispace(int2d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("\n{} = [", arrayName)
  format.println("Bounds = {}", input.bounds)
  for x in input do
    var currComplex = input[x]
    format.println("index {}: {} + {}j, ", x, currComplex.real, currComplex.imag)
  end
  format.println("]\n")
end

task print_array_3d(input : region(ispace(int3d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("\n{} = [", arrayName)
  format.println("Bounds = {}", input.bounds)
  for x in input do
    var currComplex = input[x]
    format.println("index {}: {} + {}j, ", x, currComplex.real, currComplex.imag)
  end
  format.println("]\n")
end



--function fft.generate_fft_interface(itype, dtype): itype = int1d, dtype = complex64, dim = itype.dim =1 
local fft1d = fft.generate_fft_interface(int1d, complex64)
local fft2d = fft.generate_fft_interface(int2d, complex64)
local fft3d = fft.generate_fft_interface(int3d, complex64)

local fft1d_float = fft.generate_fft_interface(int1d, complex32)
local fft2d_float = fft.generate_fft_interface(int2d, complex32)
local fft3d_float = fft.generate_fft_interface(int3d, complex32)

--demand(__inline)
task test1d_float()
  format.println("Running test1d...")
  format.println("Creating input and output arrays...")
  
  -- Initialize input and output arrays
  var r = region(ispace(int1d, 3), complex32)
  var s = region(ispace(int1d, 3), complex32)
  
  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  -- Initialize output array
  fill(s, 0)
  print_array_float(r, "Input array")

  -- Initial plan region
  var p = region(ispace(int1d, 1), fft1d_float.plan)
  format.println("Calling make_plan...")
  fft1d_float.make_plan(r, s, p)

  -- Execute plan
  format.println("Calling execute_plan...\n")
  fft1d_float.execute_plan_task(r, s, p)

  -- Print Output
  print_array_float(s, "Output array")


  -- Destroy plan
  format.println("Calling destroy_plan...\n")
  fft1d_float.destroy_plan(p)
end


--demand(__inline)
task test1d()
  format.println("Running test1d...")
  format.println("Creating input and output arrays...")

  
  -- Initialize input and output arrays
  var r = region(ispace(int1d, 3), complex64)
  var s = region(ispace(int1d, 3), complex64)
  

  for x in r do
    r[x].real = 4
    r[x].imag = 4
  end

  -- Initialize output array
  fill(s, 0)
  print_array(r, "Input array")

  -- Initial plan region
  var p = region(ispace(int1d, 1), fft1d.plan)
  format.println("Calling make_plan...")
  fft1d.make_plan(r, s, p)

  -- Execute plan
  format.println("Calling execute_plan...\n")
  fft1d.execute_plan_task(r, s, p)

  -- Print Output
  print_array(s, "Output array")


  -- Destroy plan
  format.println("Calling destroy_plan...\n")
  fft1d.destroy_plan(p)
end

__demand(__inline)
task test1d_distrib()
  var n = fft1d.get_num_nodes()
  format.println("Num nodes in distrib is {}...", n)
  var r = region(ispace(int1d, 3*n), complex64)
  var r_part = partition(equal, r, ispace(int1d, n))
  var s = region(ispace(int1d, 3*n), complex64)
  var s_part = partition(equal, s, ispace(int1d, n))
  for x in r do
    r[x].real = 4
    r[x].imag = 4
  end
  fill(s, 0)
  print_array(r, "Input array for distrib")
  var p = region(ispace(int1d, n), fft1d.plan)
  var p_part = partition(equal, p, ispace(int1d, n))
  -- Important: this overwrites r and s!
  fft1d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  __demand(__index_launch)
  for i in r_part.colors do
    fft1d.execute_plan_task(r_part[i], s_part[i], p)
  end
  print_array(s, "Output array for distrib")
  fft1d.destroy_plan_distrib(p, p_part)
end


__demand(__inline)
task test2d()
  var r = region(ispace(int2d, { 2, 2 }), complex64)
  var s = region(ispace(int2d, { 2, 2 }), complex64)
  for x in r do
    r[x].real = 2
    r[x].imag = 2
  end
  fill(s, 1)
  print_array_2d(r, "Input array")
  var p = region(ispace(int1d, 1), fft2d.plan)
  fft2d.make_plan(r, s, p)
  fft2d.execute_plan_task(r, s, p)
  print_array_2d(s, "Output array")
  fft2d.destroy_plan(p)
end



__demand(__inline)
task test3d()
  format.println("Running test3d...")
  var r = region(ispace(int3d, { 2, 2, 2 }), complex64)
  var s = region(ispace(int3d, { 2, 2, 2 }), complex64)
  for x in r do
    r[x].real = 4
    r[x].imag = 4
  end
  fill(s, 0)
  print_array_3d(r, "Input array")
  -- Important: this overwrites r and s!
  var p = region(ispace(int1d, 1), fft3d.plan)
  fft3d.make_plan(r, s, p)
  fft3d.execute_plan_task(r, s, p)
  print_array_3d(s, "Output array")
  fft3d.destroy_plan(p)
  format.println("Completed test3d...")

end

-- Main function
task main()
 test1d_float()
 --test1d()
 --test1d_distrib()
 --test2d()
 --test3d()
end

--Include mapper
regentlib.start(main, cmapper.register_mappers)
