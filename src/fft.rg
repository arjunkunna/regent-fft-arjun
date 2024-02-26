import "regent"


local format = require("std/format")
local data = require("common/data")
local cmapper = require("test_mapper")

local gpuhelper = require("regent/gpu/helper")
local default_foreign = gpuhelper.check_gpu_available()

--Import C and FFTW APIs
local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
local fftwf_c = 
terralib.linklibrary("libfftw3.so")

--Import cuFFT API
local cufft_c
if default_foreign then
  cufft_c = terralib.includec("cufftXt.h")
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

--itype should be the index type of the transform (int1d for 1d/int2d for 2d) and dtype = complex64
function fft.generate_fft_interface(itype, dtype_in, dtype_out)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim
  local dtype_size = terralib.sizeof(dtype_out)
  local real_flag = false
  if dtype_in == double then
    real_flag = true
  end


  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  --assert(dtype == complex64, "currently only complex64 is supported")
  assert(terralib.sizeof(complex64) == terralib.sizeof(cufft_c.cufftDoubleComplex), "Structure used in transform has to match complex64 size")
  assert(terralib.sizeof(complex64) == terralib.sizeof(fftw_c.fftw_complex),"Structure used in transform has to match complex64 size")

  local iface = {}

  -- Create fspaces depending on whether GPUs are used or not
  local iface_plan
 
  if default_foreign then 
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      float_p : fftw_c.fftwf_plan,
      cufft_p : cufft_c.cufftHandle,
      address_space : c.legion_address_space_t,
    }
  else
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      float_p : fftw_c.fftwf_plan,
      address_space : c.legion_address_space_t,
    }
  end
  
  --Store plan fspace in our interface
  iface.plan = iface_plan
  iface.plan.__no_field_slicing = true -- don't field slice this struct
     
  -- d is dimension, t is dtype/region fspace (complex 64)
  local function make_get_base(d, t)

    local rect_t = c["legion_rect_" .. d .. "d_t"]
    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. d .. "d"]
    local raw_rect_ptr = c["legion_accessor_array_" .. d .. "d_raw_rect_ptr"]
    local destroy_accessor = c["legion_accessor_array_" .. d .. "d_destroy"]

    --Function to get base pointer of region: returns base_pointer
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

      --regentlib.assert(offsets[0].offset == terralib.sizeof(complex64), "stride does not match expected value")
      destroy_accessor(accessor)
      return base_pointer
    end

    return rect_t, get_base
  end

  local rect_plan_t, get_base_plan = make_get_base(1, iface.plan) --get_base_plan returns a base_pointer to a region with fspace iface.plan. (always dim = 1 because plan regions are dim 1: 'var p = region(ispace(int1d, 1), fft1d.plan)')
  local rect_t_in, get_base_in = make_get_base(dim, dtype_in) --get_base returns a base pointer to a region with fspace dtype
  local rect_t_out, get_base_out = make_get_base(dim, dtype_out) --get_base returns a base pointer to a region with fspace dtype


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


  --Task to return pointer to plan. Takes plan region and returns pointer to plan
   __demand(__inline)
  task iface.get_plan(plan : region(ispace(int1d), iface.plan), check : bool) : &iface.plan
  where reads(plan) do
    format.println("In get_plan...")

    --Get physical region
    --__physical(r.{f, g, ...}) returns an array of physical regions (legion_physical_region_t) for r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    var pr = __physical(plan)[0] --returns first physical region

    regentlib.assert(c.legion_physical_region_get_memory_count(pr) == 1, "plan instance has more than one memory?")
    
    --Ensure that plan is in the right kind of memory
    var mem_kind = c.legion_memory_kind(c.legion_physical_region_get_memory(pr, 0)) --legion_memory_t legion_physical_region_get_memory(legion_physical_region_t handle, size_t index); --legion_memory_kind_t legion_memory_kind(legion_memory_t mem);
    regentlib.assert(mem_kind == c.SYSTEM_MEM or mem_kind == c.REGDMA_MEM or mem_kind == c.Z_COPY_MEM, "plan instance must be stored in sysmem, regmem, or zero copy mem")

    --Get pointer to plan: get_base_plan returns a base_pointer to a region with fspace iface.plan
    format.println("Getting plan_base...")
    var plan_base = get_base_plan(rect_plan_t(plan.ispace.bounds), __physical(plan)[0], __fields(plan)[0]) --get_base_plan returns a base_pointer to a region with fspace iface.plan
    var i = c.legion_processor_address_space(get_executing_processor(__runtime())) --legion_address_space_t legion_processor_address_space(legion_processor_t proc);

    var p : &iface.plan
    var bounds = plan.ispace.bounds

    --T(x) is a cast from type T to a value x: int1d(1) = number 1 (casted to type int1d)
    if bounds.hi - bounds.lo + 1 > int1d(1) then
      p = plan_base + i
    else
      p = plan_base
    end

    regentlib.assert(not check or p.address_space == i, "plans can only be used on the node where they are originally created")
    format.println("Returning plan_base...")
    return p
  end

  local plan_dft = fftw_c["fftw_plan_dft_" .. dim .. "d"]



  ----MAKE PLAN FUNCTIONS----
  
  --task: Make plan in GPU version. Stores plan in cufft_p
  local make_plan_gpu
  if default_foreign then
    __demand(__cuda, __leaf)

    task make_plan_gpu(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan), address_space : c.legion_address_space_t)
    
    where reads writes(input, output, plan) do
      format.println("In iface.make_plan_gpu...")

      --Get pointer to plan
      var p = iface.get_plan(plan, true)

      --Verify we are in GPU mode by checking TOC_PROC
      --Takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
      var proc = get_executing_processor(__runtime())
      format.println("Make_Plan_GPU: TOC PROC IS {}",c.TOC_PROC)
      format.println("Make_Plan_GPU: Processor kind is {}", c.legion_processor_kind(proc))

      if c.legion_processor_kind(proc) == c.TOC_PROC then
        var i = c.legion_processor_address_space(proc)
        regentlib.assert(address_space == i, "make_plan_gpu must be executed on a processor in the same address space")

        --Get input and output bases
        var input_base = get_base_in(rect_t_in(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
        var output_base = get_base_out(rect_t_out(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
        var lo = input.ispace.bounds.lo:to_point()
        var hi = input.ispace.bounds.hi:to_point()
        var n : int[dim]
        ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      

        --Call cufftPlanMany
        --cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) --rank = dimensionality of transform (1,2,3)
        format.println("Calling cufftPlanMany...")

        var ok = 0

        if real_flag then
          format.println("Calling cufftPlanMany with CUFFT_D2Z ...")
          ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_D2Z, 1)
        else
          format.println("Calling cufftPlanMany with CUFFT_Z2Z ...")
          ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_Z2Z, 1)
        end

        --Check return value of cufftPlanMany
        if ok == cufft_c.CUFFT_INVALID_VALUE then
          format.println("Invalid value in cufftplanmany")
        end

        regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftPlanMany failed")
        format.println("cufftPlanMany Successful")

      else 
        format.println("GPU processor not identified: TOC_PROC not equal to processor kind")
        regentlib.assert(false, "make_plan_gpu must be executed on a GPU processor")
      end
    end
  end
  
  --Takes input, output, and plan regions and makes plan using cufft/FFTW functions
   __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype_in),output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    format.println("In iface.make_plan...")

    --Get plan returns pointer to plan
    format.println("Calling get_plan...")
    var p = iface.get_plan(plan, false)


    --Get_executing process: takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime())) --legion_processor_address_space: takes a legion_processor_t proc and returns a legion_address_space_t

    --Check input/output bounds. Bounds example: 
    --var is = ispace(int1d, 12, -1)
    --is.bounds -- returns rect1d { lo = int1d(-1), hi = int1d(10) }
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")

    -- https://legion.stanford.edu/doxygen/class_legion_1_1_physical_region.html
    --__physical(r.{f, g, ...}) returns an array of physical regions (legion_physical_region_t) for r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    --__fields(r.{f, g, ...}) returns an array of the field IDs (legion_field_id_t) of r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    --local rect_t, get_base = make_get_base(dim, dtype) --get_base returns a base pointer to a region with fspace dtype. dim =1 , dtype = complex64 --local terra get_base(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)

    --Get pointers to input and output regions
    var input_base = get_base_in(rect_t_in(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base_out(rect_t_out(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    --Call fftw_c.fftw_plan_dft_1d: fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,int sign, unsigned flags). n is the size of transform, in and out are pointers to the input and output arrays. Sign is the sign of the exponent in the transform, can either be FFTW_FORWARD (1) or FFTW_BACKWARD (-1). Flags: FFTW_ESTIMATE, on the contrary, does not run any computation
    format.println("Storing fftw_plan in p.p...")
    --p.p = fftw_c.fftw_plan_dft_1d(input.ispace.volume, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    
    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()

    --dtype size is 8 for complex32 and 16 for complex64
    format.println("size of dtype is {}", dtype_size)

    --if dtype_size == 8 then
    --  format.println("p.floatp being saved")
    --  p.float_p = fftw_c.fftwf_plan_dft_1d(3, [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_MEASURE)
    
    --else 
    if real_flag then
      format.println("input is real")
      p.p = fftw_c.fftw_plan_dft_r2c_1d(3, [&double](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_ESTIMATE)
    else
      if dim == 1 then
        var n : int[dim]
        ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
        format.println("n[0] is {}, dim is {}", n[0], dim)
        p.p = plan_dft([data.range(dim):map(function(i) return rexpr hi.x[i] - lo.x[i] + 1 end end)], [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_MEASURE)
      end
    end

    if dim == 2 then
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("n[0] is {}, n[1] is {}, dim is {}", n[0], n[1], dim)
      p.p = fftw_c.fftw_plan_dft_2d(n[0],n[1], [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end

    if dim == 3 then
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("n[0] is {}, n[1] is {}, n[2] is {}, dim is {}", n[0], n[1], n[2], dim)
      p.p = fftw_c.fftw_plan_dft_3d(n[0],n[1],n[2], [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end

    p.address_space = address_space

    --If GPUs, call make_plan_GPU
    rescape
      if default_foreign then
        remit rquote
          format.println("Num_local_gpus is {}", iface.get_num_local_gpus())
          if iface.get_num_local_gpus() > 0 then
            format.println("GPUs identified: calling make_plan_gpu...")
            make_plan_gpu(input, output, plan, p.address_space)
          end
        end
      else
        return rquote end
      end
    end
  end

  task iface.make_plan_task(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    iface.make_plan(input, output, plan)
  end

  --Distribution version of make_plan
  __demand(__inline)
  task iface.make_plan_distrib(input : region(ispace(itype), dtype_in), input_part : partition(disjoint, input, ispace(int1d)), output : region(ispace(itype), dtype_out), output_part : partition(disjoint, output, ispace(int1d)), plan : region(ispace(int1d), iface.plan), plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(input, output, plan) do
    

    --Get number of nodes and check consistency of nodes/colors
    var n = iface.get_num_nodes()
    regentlib.assert(input_part.colors.bounds.hi - input_part.colors.bounds.lo + 1 == int1d(n), "input_part colors size must be equal to the number of nodes")
    regentlib.assert(input_part.colors.bounds == output_part.colors.bounds, "input_part and output_part colors must be equal")
    regentlib.assert(input_part.colors.bounds == plan_part.colors.bounds, "input_part and plan_part colors must be equal")

    var p : iface.plan
    --T(x) is a cast from type T to a value x
    p.p = [fftw_c.fftw_plan](0)
    p.address_space

    if default_foreign then
      p.cufft_p = 0
    end

    fill(plan, p)

    __demand(__index_launch)
    for i in plan_part.colors do
      iface.make_plan_task(input_part[i], output_part[i], plan_part[i])
    end
  end


  ----EXECUTE PLAN FUNCTIONS----

  --Task to execute plan. Calls cufftExecZ2Z if in GPU mode and fftw_execute_dft if in CPU mode
   __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    format.println("In iface.execute_plan...")

    --Get pointer to plan
    var p = iface.get_plan(plan, true) --task iface.get_plan(plan : region(ispace(int1d), iface.plan), check : bool) : &iface.plan

    --Get pointers to input and output regions
    var input_base = get_base_in(rect_t_in(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base_out(rect_t_out(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    --Check if we are in GPU or CPU mode
    var proc = get_executing_processor(__runtime())
    format.println("execute_plan: TOC PROC IS {}",c.TOC_PROC) --TOC = Throughput Oriented Core: Means we have a GPU
    format.println("execute_plan: Processor kind is {}", c.legion_processor_kind(proc))

    format.println("size of dtype is {}", dtype_size)

    --If in GPU mode, usecufftExecZ2Z
    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("execute plan via cuFFT\n")

      var ok = 0
      format.println("size of dtype is {}", dtype_size)

      if real_flag then
        ok = cufft_c.cufftExecD2Z(p.cufft_p, [&cufft_c.cufftDoubleReal](input_base), [&cufft_c.cufftDoubleComplex](output_base))
      else
        ok = cufft_c.cufftExecZ2Z(p.cufft_p, [&cufft_c.cufftDoubleComplex](input_base), [&cufft_c.cufftDoubleComplex](output_base), cufft_c.CUFFT_FORWARD)
      end

      -- TODO: Float verion
      --else
      --  c.printf("float version of cufftExec being called\n")
      --  ok = cufft_c.cufftExecZ2Z(p.cufft_p, [&cufft_c.cufftComplex](input_base), [&cufft_c.cufftComplex](output_base), cufft_c.CUFFT_FORWARD)
      --end

      --Check return values of Exec
      if ok == cufft_c.CUFFT_INVALID_VALUE then
          format.println("Invalid value in cufftExecZ2Z")
      end

      if ok == cufft_c.CUFFT_INVALID_PLAN then
          format.println("Invalid plan passed to cufftExecZ2Z")
      end

      --format.println("cufftExecZ2Z returned {}", ok)
      regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftExecZ2Z failed")
      format.println("cufftExecZ2Z successful")

    --Otherwise, use FFTW if no GPU
    else
      c.printf("execute plan via FFTW\n")

      -- TODO: Float verion
      --if dtype_size == 8 then
      --  format.println("executing float fftw")
      --  fftw_c.fftwf_execute_dft(p.float_p, [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base))     --void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out))
      --else 
      
      if real_flag then
        format.println("executing r2c")
        fftw_c.fftw_execute_dft_r2c(p.p, [&double](input_base), [&fftw_c.fftw_complex](output_base))     --void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out))
      else
        fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))     --void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out))
      end
      --end
    end
  end

  __demand(__cuda, __leaf)
  task iface.execute_plan_task(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), reads writes(output) do
    iface.execute_plan(input, output, plan)
  end


  ----DESTROY PLAN FUNCTIONS----

  --Task to destroy plan. Takes plan region as argument.
  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
   format.println("In iface.destroy_plan...")

    var p = iface.get_plan(plan, true)

    var proc = get_executing_processor(__runtime())

    -- If using GPUs, call cufftDestroy
    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("Destroy plan via cuFFT\n")

      --Function: cufftResult cufftDestroy(cufftHandle plan)
      cufft_c.cufftDestroy(p.cufft_p) 
    end
    
    -- Else, cal fftw_destroy
    c.printf("Destroy plan via FFTW\n")
    fftw_c.fftw_destroy_plan(p.p)
  end


  task iface.destroy_plan_task(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    iface.destroy_plan(plan)
  end

  -- Distributed version of destroy_plan
  __demand(__inline)
  task iface.destroy_plan_distrib(plan : region(ispace(int1d), iface.plan), plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(plan) do
    format.println("In iface.destroy_plan_distrib...")

    for i in plan_part.colors do
      iface.destroy_plan_task(plan_part[i])
    end
  end

  return iface
end

return fft