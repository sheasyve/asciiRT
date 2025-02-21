# <a href = "https://sheasyve.dev/projects/ascii_rt.html"> AsciiRT</a>
![mark](https://github.com/user-attachments/assets/65d30e8a-a69d-47cb-986b-1a8c72d2b514)
[Real Time Animation Demo](https://zenodo.org/records/14524908?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBiM2UxMjFlLTU1MGUtNDkwNi1hNzRhLTY2N2YyNjFkNmUwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJhMzk1NzhlNWU1ODAzYjIyMmZkMTA2NDQ4OGYyMWUxNyJ9.8Y_aLyMM4uLYEOj-QutfOv3Jm_0GT4GhZAmrcECCBsHiym-U9CldCw26N1n54Mhis2SwDVBE36LCCiId9du4hA)

AsciiRT is a highly optimized GPU ray-tracing engine leveraging CUDA to render object models in ASCII characters. Extensive optimizations resulted in a 900x speedup over the initial multi-core version, enabling real-time 60 FPS animations.

Users can load multiple object files (.obj), render still images or animations, and apply transformations such as translation and rotation. The ray-tracing kernel processes the scene, mapping brightness values to ASCII characters for terminal-based rendering. With optimized parallelism, animations maintain smooth 60 FPS output.

### Technical Breakdown  
- Each object model consists of connected points in space, which are:  
  - Transformed, ray-traced, and converted to ASCII in real-time.  
  - Rendered with realistic lighting using reflections, shadows, and perspective projection.  
  - Illuminated by four light sources.  
- The ray-tracing engine:  
  - Casts a ray for every pixel from the camera’s perspective in each frame.  
  - Determines pixel brightness based on ray-object collisions and lighting calculations.  
  - Maps brightness values to ASCII characters for terminal-based rendering.  
- A Bounding Volume Hierarchy (BVH) accelerates ray-object intersection tests, reducing computational overhead.  
- The GPU assigns each ray to a separate thread, maximizing parallel execution for real-time performance.  

### Experience Gained  
- Developed a real-time ray tracer from scratch, balancing performance and accuracy.  
- Implemented efficient ASCII-based rendering techniques for terminal output.  
- Utilized CUDA for massive parallelization, achieving a **900x speedup** over CPU-based implementations.  
- Optimized GPU memory usage and kernel execution with Nvidia Nsight.  
- Constructed a **Bounding Volume Hierarchy (BVH)** for fast ray-object intersection tests.  
- Handled reflections, shadows, and perspective projection to enhance realism in ASCII-rendered scenes.  
- Fine-tuned text-based animations to achieve live rendering.  
