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
    
    CUDA (New Implementation, with Malloc Warming):
    Runtime: 0.0166461 second

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

    CUDA (New Implementation, with Malloc Warming):
    Runtime: 0.135541 seconds


\begin{table}[h!]
\centering
\begin{tabular}{|l|l|l|l|l|}
\hline
\textbf{Scenario} & \textbf{Version}                                      & \textbf{Runtime (s)} & \textbf{C++ Speedup} & \textbf{CUDA Speedup} \\ \hline
\multicolumn{5}{|c|}{\textbf{Car}}                                                                                   \\ \hline
C++                & Initial Version                                      & 17.2245             & 1.0x                &                      \\ \hline
C++                & Initial Version with current shading model          & 22.6327             & 0.76x               &                      \\ \hline
C++                & Current Implementation (new shading, BVH)           & 9.98129             & 1.73x               &                      \\ \hline
CUDA               & Initial Implementation (old shading model)          & 13.4659             & 1.28x               & 1.0x                 \\ \hline
CUDA               & Interim Report (BVH, new shading)                   & 0.280268            & 61.45x              & 48.05x               \\ \hline
CUDA               & New Implementation (no eigen, SOA BVH, SMEM Lights) & 0.152456            & 112.99x             & 88.32x               \\ \hline
CUDA               & New Implementation with Malloc Warming              & 0.0166461           & 1034.56x            & 808.82x              \\ \hline
\multicolumn{5}{|c|}{\textbf{Pirate Ship}}                                                                        \\ \hline
C++                & Initial Version                                      & 197.683             & 1.0x                &                      \\ \hline
C++                & Initial Version with current shading model          & 210.277             & 0.94x               &                      \\ \hline
C++                & Current Implementation (new shading, BVH)           & 91.1882             & 2.17x               &                      \\ \hline
CUDA               & Initial Implementation                               & 122.176             & 1.62x               & 1.0x                 \\ \hline
CUDA               & Interim Report (BVH, new shading)                   & 1.80052             & 109.78x             & 67.87x               \\ \hline
CUDA               & New Implementation (no eigen, SOA BVH, SMEM Lights) & 0.472608            & 418.17x             & 258.49x              \\ \hline
CUDA               & New Implementation with Malloc Warming              & 0.135541            & 1458.87x            & 901.43x              \\ \hline
\end{tabular}
\caption{Performance Comparison for Car and Pirate Ship Renderings with Relative Speedups}
\label{tab:performance_comparison_separated_speedup}
\end{table}