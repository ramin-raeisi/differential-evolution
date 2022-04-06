# differential-evolution

The following is an example of the output of the algorithm in the console:
```
-------------------------------------------------------------------------------------------------------------------------------------------------
Please wait --> running differential evolution on CPU

best cost:          0
best agent:         -3.38306e-09  1.21309e-09   5.87949e-10   3.06625e-09   4.50858e-09   
-------------------------------------------------------------------------------------------------------------------------------------------------
configuration: 
populationSize:     4000
maxIterations:      2000
cpu_time:           9280ms
-------------------------------------------------------------------------------------------------------------------------------------------------
Kernel time     | Total Time      | speedup         | Min cost        | Best Solution                           
-------------------------------------------------------------------------------------------------------------------------------------------------
101 ms          | 200 ms          | 46x             | 0               | -5.28522e-10  3.70555e-09   1.55728e-09   3.13787e-10   -2.60467e-09  
102 ms          | 102 ms          | 90x             | 0               | 1.07425e-09   2.94265e-09   3.46408e-09   -2.30092e-09  7.26217e-10   
100 ms          | 100 ms          | 92x             | 0               | -4.11821e-10  -3.09443e-09  -2.26575e-09  2.25448e-09   -3.05489e-09  
101 ms          | 101 ms          | 91x             | 0               | -1.38828e-09  -1.60295e-09  3.01636e-10   6.51118e-10   -2.80872e-09  
100 ms          | 100 ms          | 92x             | 0               | 2.00182e-10   1.68071e-09   -4.94938e-09  1.80164e-10   -3.41331e-09  
-------------------------------------------------------------------------------------------------------------------------------------------------
```

The speedup is about 92x over the CPU, as you can see. You can adjust the speedup by changing the
`blocksize` of the cuda kernel. For instance, when I set `int kernelBlockSize = 32;` 
in the `main.cu`, the following results can be seen

```
-------------------------------------------------------------------------------------------------------------------------------------------------
Please wait --> running differential evolution on CPU

best cost:          0
best agent:         -3.38306e-09  1.21309e-09   5.87949e-10   3.06625e-09   4.50858e-09   
-------------------------------------------------------------------------------------------------------------------------------------------------
configuration: 
populationSize:     4000
maxIterations:      2000
cpu_time:           9276ms
-------------------------------------------------------------------------------------------------------------------------------------------------
Kernel time     | Total Time      | speedup         | Min cost        | Best Solution                           
-------------------------------------------------------------------------------------------------------------------------------------------------
39 ms           | 139 ms          | 66x             | 0               | 3.20881e-09   -8.43267e-10  4.62413e-10   1.69356e-09   1.59149e-09   
39 ms           | 39 ms           | 237x            | 0               | -2.77717e-09  2.90796e-10   2.99689e-09   -3.13111e-09  2.08979e-09   
40 ms           | 41 ms           | 226x            | 0               | -2.03266e-09  -1.24751e-09  -2.30189e-09  1.39379e-09   2.23023e-09   
39 ms           | 40 ms           | 231x            | 0               | 1.40081e-09   4.47034e-10   4.4781e-09    2.53058e-10   2.60853e-09   
38 ms           | 38 ms           | 244x            | 0               | 5.48091e-11   -1.99432e-09  1.22922e-09   3.41313e-09   1.3266e-09    
-------------------------------------------------------------------------------------------------------------------------------------------------
```

# Supported crossovers

There are two major crossovers supported by the code:

- binomial crossover
- exponential crossover

If you want to include the exponential crossover code, compile it with the `EXPONENTIAL_CROSSOVER` definition.
 or you could simply
uncomment the `add_compile_definitions(EXPONENTIAL_CROSSOVER)` in the `CMakeLists.txt` file. Note that the exponential variant is slightly slower than the binomial