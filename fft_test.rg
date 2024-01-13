import "regent"

local c = regentlib.c
local format = require("std/format")
local fftw_c = terralib.includec("fftw3.h")
terralib.linklibrary("libfftw3.so")

fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1
fftw_c.FFTW_ESTIMATE = 2 ^ 6

local fft = {}

--itype should be the dimension of the transform and dtype = complex64
function fft.generate_fft_interface(itype, dtype)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim

  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  assert(dtype == complex64, "currently only complex64 is supported")

  local rect_t = c["legion_rect_" .. dim .. "d_t"]
  local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. dim .. "d"]
  local raw_rect_ptr = c["legion_accessor_array_" .. dim .. "d_raw_rect_ptr"]
  local destroy_accessor = c["legion_accessor_array_" .. dim .. "d_destroy"]

  local iface = {}

  struct iface.plan {
    p : fftw_c.fftw_plan
  }

  --Function to get base pointer of region
  local terra get_base(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)
    var subrect : rect_t
    var offsets : c.legion_byte_offset_t[dim]
    var accessor = get_accessor(physical, field)
    var base_pointer = [&dtype](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))
    regentlib.assert(base_pointer ~= nil, "base pointer is nil")
    escape
      for i = 0, dim-1 do
        emit quote
          regentlib.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
        end
      end
    end


    regentlib.assert(offsets[0].offset == terralib.sizeof(dtype), "stride does not match expected value")

    destroy_accessor(accessor)

    return base_pointer
  end



  __demand(__inline)
  task iface.make_plan(input : region(ispace(int1d), complex64),output : region(ispace(int1d), complex64))
  where reads writes(input, output) do

    --var is = ispace(int1d, 12, -1)
    --is.bounds -- returns rect1d { lo = int1d(-1), hi = int1d(10) }
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")


    -- https://legion.stanford.edu/doxygen/class_legion_1_1_physical_region.html
    --__physical(r.{f, g, ...}) returns an array of physical regions (legion_physical_region_t) for r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    --__fields(r.{f, g, ...}) returns an array of the field IDs (legion_field_id_t) of r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    
    var input_base = get_base(c.legion_rect_1d_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(c.legion_rect_1d_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    -- fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,int sign, unsigned flags)
    -- n is the size of transform, in and out are pointers to the input and output arrays
    -- sign is the sign of the exponent in the transform, can either be FFTW_FORWARD (1) or FFTW_BACKWARD (-1)
    -- flags: FFTW_ESTIMATE, on the contrary, does not run any computation
    
    return iface.plan {
      fftw_c.fftw_plan_dft_1d(input.ispace.volume, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    }
  end

  __demand(__inline)
  task iface.execute_plan(input : region(ispace(int1d), complex64), output : region(ispace(int1d), complex64), p : iface.plan)
  where reads(input), writes(output) do
    fftw_c.fftw_execute(p.p)
  end

  __demand(__inline)
  task iface.destroy_plan(p : iface.plan)
    fftw_c.fftw_destroy_plan(p.p)
  end

  return iface
end


task print_array(input : region(ispace(int1d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("{}, = [", arrayName)
  for x in input do
    var currComplex = input[x]
    format.println("{} + {}j, ", currComplex.real, currComplex.imag)
  end
  format.println("]\n")
end

local fft1d = fft.generate_fft_interface(int1d, complex64)

task main()
  var r = region(ispace(int1d, 3), complex64)
  var s = region(ispace(int1d, 3), complex64)
  fill(s, 1)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  fill(s, 0)
  -- Important: overwrites input/output!

  print_array(r, "Input array")

  var p = fft1d.make_plan(r, s)


  format.println("Executing...\n")
  fft1d.execute_plan(r, s, p)

  print_array(s, "Output array")


  format.println("Destroying...\n")
  fft1d.destroy_plan(p)
  
end

regentlib.start(main)
