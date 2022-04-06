#include "DifferentialEvolution.cuh"

#include <chrono>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

#define gpuErrorCheck(ans)                                                     \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    getchar();
    if (abort)
      exit(code);
  }
}

namespace deGPU {

DifferentialEvolution::DifferentialEvolution(int populationSize,
                                             int numberOfGeneration, int dim,
                                             double CR, double F, double minX,
                                             double maxX, uint blockDim) {
  if (populationSize < 4) {
    throw std::invalid_argument(
        "populationSize must be greater than 4 (i.e populationSize>=4)");
  }
  _populationSize = populationSize;
  _numberOfGeneration = numberOfGeneration;
  _dim = dim;
  _pCR = CR;
  _pF = F;
  _minX = minX;
  _maxX = maxX;
  _blockDim = blockDim;

  gpuErrorCheck(
      cudaMalloc((void **)&zD, _dim * _populationSize * sizeof(double)));
  gpuErrorCheck(
      cudaMalloc((void **)&rD, _dim * _populationSize * sizeof(double)));
  gpuErrorCheck(
      cudaMalloc((void **)&trialD, _dim * _populationSize * sizeof(double)));
  gpuErrorCheck(cudaMalloc((void **)&populationD,
                           _dim * _populationSize * sizeof(double)));
  gpuErrorCheck(cudaMalloc((void **)&newPopulationD,
                           _dim * _populationSize * sizeof(double)));
  costH = new double[_populationSize];
  gpuErrorCheck(cudaMalloc((void **)&costD, _populationSize * sizeof(double)));
  // create random states
  gpuErrorCheck(
      cudaMalloc((void **)&globalRandStatesD,
                 _populationSize * sizeof(curandStatePhilox4_32_10_t)));
}

optimizeResult DifferentialEvolution::optimize() {

  static constexpr float warpSize = 32;
  int blockDim = static_cast<int>(
      std::ceil(static_cast<float>(_blockDim) / warpSize) * warpSize);

  int gridDim = std::ceil(static_cast<float>(_populationSize) /
                          static_cast<float>(blockDim));

  // initialize population and cost for each agent(we use
  // curandStatePhilox4_32_10_t for performance reason)
  gpu::init<<<gridDim, blockDim>>>(
      populationD, _populationSize, _dim, costD, clock(),
      (curandStatePhilox4_32_10_t *)globalRandStatesD, _minX, _maxX);

  gpuErrorCheck(cudaPeekAtLastError());
  gpuErrorCheck(cudaDeviceSynchronize());

  for (int i = 0; i < _numberOfGeneration; ++i) {
    gpu::SelectionAndCrossing<<<gridDim, blockDim>>>(
        _populationSize, _dim, populationD, newPopulationD, zD, rD, trialD, _pF,
        _pCR, costD, (curandStatePhilox4_32_10_t *)globalRandStatesD, _minX,
        _maxX);
    gpuErrorCheck(cudaPeekAtLastError());
    // swap population(for free)
    double *tmp = populationD;
    populationD = newPopulationD;
    newPopulationD = tmp;
  }

  gpuErrorCheck(cudaDeviceSynchronize());

  // copy the cost from device to host
  gpuErrorCheck(cudaMemcpy(costH, costD, _populationSize * sizeof(double),
                           cudaMemcpyDeviceToHost));

  // find the best solution
  auto [minIndex, minCost] = returnBestSolution();

  // copy the best solution from device to host
  std::vector<double> bestX(_dim);
  cudaMemcpy(bestX.data(), populationD + minIndex * _dim, _dim * sizeof(double),
             cudaMemcpyDeviceToHost);

  return {minCost, bestX};
}

std::pair<int, double> DifferentialEvolution::returnBestSolution() {
  double minCost = costH[0];
  int minIndex = 0;
  for (int i = 1; i < _populationSize; ++i) {
    if (costH[i] < minCost) {
      minCost = costH[i];
      minIndex = i;
    }
  }
  return {minIndex, minCost};
}

DifferentialEvolution::~DifferentialEvolution() {
  gpuErrorCheck(cudaFree(zD));
  gpuErrorCheck(cudaFree(rD));
  gpuErrorCheck(cudaFree(trialD));
  gpuErrorCheck(cudaFree(populationD));
  gpuErrorCheck(cudaFree(newPopulationD));
  gpuErrorCheck(cudaFree(costD));
  gpuErrorCheck(cudaFree(globalRandStatesD));
  delete[] costH;
}

} // namespace deGPU
