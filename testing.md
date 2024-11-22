- Render car with all 4

- Render ship with all 4

Car:

c++ (Initial Version):
Runtime: 17.2245 seconds
c++ (Initial Version with current shading model):
Runtime: 22.6327 seconds
c++ (Current Implementation, with new shading model and bvh):
Runtime: 9.98129 seconds
CUDA (Initial Implementation, old shading model):
Runtime: 13.4659 second
CUDA (Interim Report, BVH and new shading):
Runtime: 0.280268 seconds
CUDA (New Implementation, with no eigen, SOA bvh, SMEM Lights):
Runtime: 0.152456 seconds

Pirate Ship:

c++ (Initial Version):
Runtime: 197.683 seconds
c++ (Initial Version with current shading model):
Runtime: 210.277 seconds
c++ (Current Implementation, with new shading model and bvh):
Runtime: 91.1882 seconds

CUDA (Initial Implementation):
Runtime: 122.176 seconds
CUDA (Interim Report, BVH and new shading):
Runtime: 1.80052 seconds
CUDA (New Implementation, with no eigen, SOA bvh, SMEM Lights):
Runtime: 0.472608 seconds
