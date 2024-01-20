import "regent"


local format = require("std/format")
local data = require("common/data")
local cmapper = require("test_mapper")


local gpuhelper = require("regent/gpu/helper")
local default_foreign = gpuhelper.check_gpu_available() and '0' or '1'

local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
terralib.linklibrary("libfftw3.so")




local cufft_c
if default_foreign then
  cufft_c = terralib.includec("cufft.h")
  terralib.linklibrary("libcufft.so")
end


--Hack: get defines from fftw3.h
fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1

fftw_c.FFTW_MEASURE = 0
fftw_c.FFTW_DESTROY_INPUT = (2 ^ 0)
fftw_c.FFTW_UNALIGNED = (2 ^ 1)
fftw_c.FFTW_CONSERVE_MEMORY = (2 ^ 2)
fftw_c.FFTW_EXHAUSTIVE = (2 ^ 3) -- NO_EXHAUSTIVE is default
fftw_c.FFTW_PRESERVE_INPUT = (2 ^ 4) -- cancels FFTW_DESTROY_INPUT
fftw_c.FFTW_PATIENT = (2 ^ 5) -- IMPATIENT is default
fftw_c.FFTW_ESTIMATE = (2 ^ 6)
fftw_c.FFTW_WISDOM_ONLY = (2 ^ 21)


local fft = {}

--itype should be the dimension of the transform and dtype = complex64
function fft.generate_fft_interface(itype, dtype)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim
  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  assert(dtype == complex64, "currently only complex64 is supported")

  local iface = {}

  local iface_plan
  if default_foreign then 
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      cufft_p : cufft_c.cufftHandle,
      address_space : c.legion_address_space_t,
    }
  else
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      address_space : c.legion_address_space_t,
    }
  end
  
  iface.plan = iface_plan
  iface.plan.__no_field_slicing = true -- don't field slice this struct
     
  local function make_get_base(d, t)

    local rect_t = c["legion_rect_" .. d .. "d_t"]
    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. d .. "d"]
    local raw_rect_ptr = c["legion_accessor_array_" .. d .. "d_raw_rect_ptr"]
    local destroy_accessor = c["legion_accessor_array_" .. d .. "d_destroy"]

    --Function to get base pointer of region
    local terra get_base(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)
      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[d]
      var accessor = get_accessor(physical, field)
      var base_pointer = [&t](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))
      regentlib.assert(base_pointer ~= nil, "base pointer is nil")
      escape
        for i = 0, d-1 do
          emit quote
            regentlib.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
          end
        end
      end

      regentlib.assert(offsets[0].offset == terralib.sizeof(complex64), "stride does not match expected value")
      destroy_accessor(accessor)
      return base_pointer
    end

    return rect_t, get_base
  end
  local rect_plan_t, get_base_plan = make_get_base(1, iface.plan)
  local rect_t, get_base = make_get_base(dim, dtype)

  -- Takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
  local terra get_executing_processor(runtime : c.legion_runtime_t)
    var ctx = c.legion_runtime_get_context()
    var result = c.legion_runtime_get_executing_processor(runtime, ctx)
    c.legion_context_destroy(ctx)
    return result
  end

  -- FIXME: Keep this in sync with default_mapper.h
  local DEFAULT_TUNABLE_NODE_COUNT = 0
  local DEFAULT_TUNABLE_LOCAL_CPUS = 1
  local DEFAULT_TUNABLE_LOCAL_GPUS = 2
  local DEFAULT_TUNABLE_LOCAL_IOS = 3
  local DEFAULT_TUNABLE_LOCAL_OMPS = 4
  local DEFAULT_TUNABLE_LOCAL_PYS = 5
  local DEFAULT_TUNABLE_GLOBAL_CPUS = 6
  local DEFAULT_TUNABLE_GLOBAL_GPUS = 7
  local DEFAULT_TUNABLE_GLOBAL_IOS = 8
  local DEFAULT_TUNABLE_GLOBAL_OMPS = 9
  local DEFAULT_TUNABLE_GLOBAL_PYS = 10


  __demand(__inline)
  task iface.get_tunable(tunable_id : int)
    var f = c.legion_runtime_select_tunable_value(__runtime(), __context(), tunable_id, 0, 0)
    var n = __future(int64, f)

    -- FIXME (Elliott): I thought Regent was supposed to copy on
    -- assignment, but that seems not to happen here, so this would
    -- result in a double destroy if we free here.

    -- c.legion_future_destroy(f)
    return n
  end

  __demand(__inline)
  task iface.get_num_nodes()
    return iface.get_tunable(DEFAULT_TUNABLE_NODE_COUNT)
  end


  __demand(__inline)
  task iface.get_num_local_gpus()
    return iface.get_tunable(DEFAULT_TUNABLE_LOCAL_GPUS)
  end

   __demand(__inline)
  task iface.get_plan(plan : region(ispace(int1d), iface.plan), check : bool) : &iface.plan
  where reads(plan) do
    --Hack: 3Bneed to use raw access to circument CUDA checker here.
    var pr = __physical(plan)[0]
    regentlib.assert(c.legion_physical_region_get_memory_count(pr) == 1, "plan instance has more than one memory?")
    var mem_kind = c.legion_memory_kind(c.legion_physical_region_get_memory(pr, 0))
    regentlib.assert(
      mem_kind == c.SYSTEM_MEM or mem_kind == c.REGDMA_MEM or mem_kind == c.Z_COPY_MEM,
      "plan instance must be stored in sysmem, regmem, or zero copy mem")
    var plan_base = get_base_plan(rect_plan_t(plan.ispace.bounds), __physical(plan)[0], __fields(plan)[0])
    var i = c.legion_processor_address_space(get_executing_processor(__runtime()))
    var p : &iface.plan
    var bounds = plan.ispace.bounds
    --T(x) is a cast from type T to a value x: int1d(1) = number 1 (casted to type int1d)
    if bounds.hi - bounds.lo + 1 > int1d(1) then
      p = plan_base + i
    else
      p = plan_base
    end
    regentlib.assert(not check or p.address_space == i, "plans can only be used on the node where they are originally created")
    return p
  end

  local plan_dft = fftw_c["fftw_plan_dft_" .. dim .. "d"]

  local make_plan_gpu
  if default_foreign then
    __demand(__cuda, __leaf)
    task make_plan_gpu(input : region(ispace(itype), dtype), output : region(ispace(itype), dtype), address_space : c.legion_address_space_t) : cufft_c.cufftHandle
    where reads writes(input, output) do


      --Takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
      var proc = get_executing_processor(__runtime())
      format.println("Make_Plan_GPU: TOC PROC IS {}",c.TOC_PROC)
      format.println("Make_Plan_GPU: Proccessor kind is {}", c.legion_processor_kind(proc))

      if c.legion_processor_kind(proc) == c.TOC_PROC then
        var i = c.legion_processor_address_space(proc)
        regentlib.assert(address_space == i, "make_plan_gpu must be executed on a processor in the same address space")

        var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
        var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
        
        var n : int[dim]
      
        var cufft_p : cufft_c.cufftHandle
        var ok = cufft_c.cufftPlanMany(&cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_C2C, 1)
        regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftPlanMany failed")
        return cufft_p
      else 
        format.println("whee3")
        regentlib.assert(false, "make_plan_gpu must be executed on a GPU processor")
      end
    end
  end
  

  __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), complex64),output : region(ispace(itype), complex64), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do

    var p = iface.get_plan(plan, false)

    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))

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
    
    p.p = fftw_c.fftw_plan_dft_1d(input.ispace.volume, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    p.address_space = address_space

    format.println("whee")

    ;[(function()
         if default_foreign then
          return rquote
            if iface.get_num_local_gpus() > 0 then
              format.println("whee3")
              p.cufft_p = make_plan_gpu(input, output, address_space)
            end
          end
         else
           return rquote end
         end
       end)()]

  end



  __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype), output : region(ispace(itype), dtype), plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do

    var p = iface.get_plan(plan, true)
    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    var proc = get_executing_processor(__runtime())

    format.println("Execute plan: TOC PROC IS {}",c.TOC_PROC)
    format.println("Execute plan: Proccessor kind is {}", c.legion_processor_kind(proc))

    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("execute plan via cuFFT\n")
    else
      c.printf("execute plan via FFTW\n")
      fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))     --void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out))
    end
  end

  __demand(__cuda, __leaf)
  task iface.execute_plan_task(input : region(ispace(itype), dtype),
                               output : region(ispace(itype), dtype),
                               plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    iface.execute_plan(input, output, plan)
  end

  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    var p = iface.get_plan(plan, true)
    fftw_c.fftw_destroy_plan(p.p)
  end

  return iface
end

-- Task to print out input or output array. Takes a region and a string representing the name of the array
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

--demand(__inline)
task test1d()
  var r = region(ispace(int1d, 3), complex64)
  var s = region(ispace(int1d, 3), complex64)

  -- Initialize input and output arrays
  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  -- Make plan
  var p = region(ispace(int1d, 1), fft1d.plan)
  print_array(r, "Input array")
  fft1d.make_plan(r, s, p)

  -- Execute plan
  format.println("Executing...\n")
  __demand(__index_launch)
  fft1d.execute_plan(r, s, p)
  --fft1d.execute_plan(r, s, p)
  print_array(s, "Output array")

  -- Destroy plan
  format.println("Destroying...\n")
  fft1d.destroy_plan(p)
end

task main()
 test1d()
end

regentlib.start(main, cmapper.register_mappers)
