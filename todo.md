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
CUDA (Current Implementation, including all optimizations, as well as shadows and reflections):
Runtime: 0.280268 seconds

Pirate Ship:

c++ (Initial Version):
Runtime: 197.683 seconds
c++ (Initial Version with current shading model):
Runtime: 210.277 seconds
c++ (Current Implementation, with new shading model and bvh):
Runtime: 91.1882 seconds

CUDA (Initial Implementation):
Runtime: 122.176 seconds
CUDA (Current Implementation, including all optimizations, as well as shadows and reflections):
Runtime: 1.80052 seconds