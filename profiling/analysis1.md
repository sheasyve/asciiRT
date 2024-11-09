# Main points
- It appears that 35% or so of the program time is spent doing cudamalloc operations before launching the kernel. Perhaps we can optimize how memory is allocated or perform asynchronous operations because doing the malloc actually takes almost half our excecution time with this computationally easy model (the car).
- The raytracing kernel takes roughly 55% of the runtime, so it is the bulk of the computation. 
It is very unoptimized according to the profiler and there are various optimizations we can do to
acheive a 90% speedup within the kernel according to nvidia nsight. 

# Register use
#### Problem
Register use is too high (max use per thread) leading to low occupancy (14%). We need to get this down, 
as the gpu will not launch more threads if the registers are in use.
#### Solution
- Reduce per thread memory use.
- Minimal local variables, move everything to global memory outside the function where possible or shared memory if possible. 
- Seek other ways to limit register use.

# Partial wave occupancy
#### Problem
Grid has a partial wave which slows the program down by 33%
#### Solution
- Optimize grid and block dimensions without leftover blocks
- So, experiment with grid and block sizes and strides

# Low compute thoroughput and memory bandwidth utilization
#### Problem
17% compute and 2% memory use indicate latency and "load imbalance"
- Latency is probably the real issue
#### Solution
- Memory access patterns are not coalesced properly, so we need to fix that
- Reuse memory where possible to help with registers too
- Minimize warp divergence and memory access alignment

# Load imbalance
#### Problem
- Lots of difference in thread work
- Probably because some rays dont hit an object and have no work,
while others may have shadows and reflections and an object hit which is a lot of work.
#### Solution
- Not sure if there is one.