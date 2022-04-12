#include "deGPU.cuh"

#include <curand_kernel.h>
#include <curand_mtgp32_kernel.h>
#include <curand_philox4x32_x.h>

namespace gpu {

__device__ double evaluateCost(const double *solution, uint dim, double minX,
                               double maxX) {
  double val = 0.0;
  static constexpr double A = 10.0;

  for (int i = 0; i < dim; i++) {
    if (solution[i] < minX || solution[i] > maxX) {
      return 1e7;
    }
    val += solution[i] * solution[i] - A * cos(2 * M_PI * solution[i]);
  }

  return A * dim + val;
}

__global__ void init(double *population, uint populationSize, uint dim,
                     double *cost, unsigned long seed,
                     curandStatePhilox4_32_10_t *globalRandStates, double minX,
                     double maxX) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= populationSize)
    return;

  curandStatePhilox4_32_10_t *state = &globalRandStates[idx];
  curand_init(seed, idx, 0, state);
  for (int i = 0; i < dim; i++) {
    population[idx * dim + i] =
        (curand_uniform(state) * (maxX - minX)) + minX; // map to [minX, maxX]
  }
  cost[idx] = evaluateCost(&population[idx * dim], dim, minX, maxX);
}

__device__ void crossoverBinomial(const double *population,
                                  const double *mutVec, double *trial, int dim,
                                  curandStatePhilox4_32_10_t *randState,
                                  double CR, uint k, uint idx_dim) {
  for (size_t j = 0; j < dim; ++j) {
    if (curand_uniform(randState) < CR || j == k) {
      trial[idx_dim + j] = mutVec[idx_dim + j];
    } else {
      trial[idx_dim + j] = population[idx_dim + j];
    }
  }
}

__device__ void crossoverExponential(const double *population,
                                     const double *mutVec, double *trial,
                                     int dim,
                                     curandStatePhilox4_32_10_t *randState,
                                     double CR, uint k, uint idx_dim) {
  for (size_t j = 0; j < dim; ++j) {
    trial[idx_dim + j] = population[idx_dim + j];
  }
  uint j = k;
  uint L = 0;
  while (curand_uniform(randState) <= CR && L < dim - 1) {
    trial[idx_dim + j] = mutVec[idx_dim + j];
    j = (j + 1) % dim;
    L = L + 1;
  }
}

__global__ void
SelectionAndCrossing(int populationSize, int dim, const double *population,
                     double *new_population, double *mutVec, double *r,
                     double *trial, double F, double CR, double *cost,
                     curandStatePhilox4_32_10_t *globalRandStates, double minX,
                     double maxX) {
  uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx >= populationSize)
    return;
  curandStatePhilox4_32_10_t *state = &globalRandStates[idx];
  // For x in population select 3 random agents (a, b, c) different from x
  uint a = idx;
  uint b = idx;
  uint c = idx;

  // use uint4 to get more performance than the ordinary call to curand()
  // since it can generate 4 random numbers at once
  uint4 bulkRand;
  // Agents must be different from each other and from x
  while (a == idx || b == idx || c == idx || a == b || a == c || b == c) {
    bulkRand = curand4(state);
    a = bulkRand.x % populationSize;
    b = bulkRand.y % populationSize;
    c = bulkRand.z % populationSize;
  }

  uint idx_dim = idx * dim;

  // mutation
  for (int i = 0; i < dim; ++i) {
    mutVec[idx_dim + i] =
        population[a * dim + i] +
        F * (population[b * dim + i] - population[c * dim + i]);
  }

  // Chose random k in [0,...,Dim-1]
  uint k = bulkRand.w % dim;

#ifndef EXPONENTIAL_CROSSOVER
  crossoverBinomial(population, mutVec, trial, dim, state, CR, k, idx_dim);
#else
  crossoverExponential(population, mutVec, trial, dim, state, CR, k, idx_dim);
#endif

  // select that is trial better than population, if so replace population
  double score = evaluateCost(&trial[idx_dim], dim, minX, maxX);
  if (score < cost[idx]) {
    for (int i = 0; i < dim; ++i) {
      new_population[idx_dim + i] = trial[idx_dim + i];
    }
    cost[idx] = score;
  } else {
    for (int i = 0; i < dim; ++i) {
      new_population[idx_dim + i] = population[idx_dim + i];
    }
  }
}

} // namespace gpu
