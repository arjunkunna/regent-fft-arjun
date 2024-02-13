






__demand(__inline)
  task make_plan(input : region(ispace(itype), dtype),output : region(ispace(itype), dtype), plan : region(ispace(int1d), iface.plan))












--demand(__inline)
task test1d()
  format.println("Running test1d...")

  format.println("Creating input and output arrays...")
  var r = region(ispace(int1d, 3), cufft_c.cufftComplex)
  var s = region(ispace(int1d, 3), cufft_c.cufftComplex)

  -- Initialize input and output arrays
  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  fill(s, 0)
  --print_array(r, "Input array")

  -- Initial plan region
  var p = region(ispace(int1d, 1), fft1d.plan)
  
  --format.println("Calling make_plan...")
  make_plan(r, s, p)

  -- Execute plan
  format.println("Calling execute_plan...\n")
  execute_plan_task(r, s, p)


  --print_array(s, "Output array")


  -- Destroy plan
  format.println("Calling destroy_plan...\n")
  destroy_plan(p)
end


__demand(__inline)
task test1d_distrib()
  var n = fft1d.get_num_nodes()
  var r = region(ispace(int1d, 128*n), complex64)
  var r_part = partition(equal, r, ispace(int1d, n))
  var s = region(ispace(int1d, 128*n), complex64)
  var s_part = partition(equal, s, ispace(int1d, n))
  fill(r, 0)
  fill(s, 0)
  var p = region(ispace(int1d, n), fft1d.plan)
  var p_part = partition(equal, p, ispace(int1d, n))
  -- Important: this overwrites r and s!
  fft1d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  fill(r, 0)
  fill(s, 0)
  __demand(__index_launch)
  for i in r_part.colors do
    fft1d.execute_plan_task(r_part[i], s_part[i], p)
  end
  fft1d.destroy_plan_distrib(p, p_part)
end

task main()
 test1d()
end

--regentlib.start(main)
regentlib.start(main, cmapper.register_mappers)