#ifndef DE_CUDA_DEGPU_CUH
#define DE_CUDA_DEGPU_CUH

#include <curand_kernel.h>
#include <vector>

namespace gpu {

/// \brief It is used to initializing the population and the first generation,
/// and computing the first generation cost.
__global__ void init(double *population, uint populationSize, uint dims,
                     double *cost, unsigned long seed,
                     curandStatePhilox4_32_10_t *globalRandStates, double minX,
                     double maxX);

/// \brief This is the main function for differential evolution. In this
/// function we will take care of the generation of the new population. and some
/// operations such as the selection, crossover and mutation.
/// \param populationSize The size of the population.
/// \param dim The number of
/// dimensions(dimensions of the problem).
/// \param population The population
/// itself.
/// \param new_population The new population(this population will be the
/// new population if it has better cost to the previous population).
/// \param mutVec The mutation vector.
/// \param r It is used in crossing.
/// \param trial It is the temporary population(if the temporary population has better cost than
/// the previous population, then the new population will be the temporary population).
/// \param F mutation factor
/// \param CR crossover factor
/// \param cost It is The cost of each agent in population.
/// \param seed The seed for the random number generator.
/// \param globalRandStates
/// \param minX The minimum value of the agents in the problem.
/// \param maxX The maximum value of the agents in the problem.
__global__ void
SelectionAndCrossing(int populationSize, int dim, const double *population,
                     double *new_population, double *mutVec, double *r,
                     double *trial, double F, double CR, double *cost,
                     curandStatePhilox4_32_10_t *globalRandStates, double minX,
                     double maxX);

/// @brief This function will evaluate the cost of the agents for `rastigin`
/// problem
/// \param solution The solution to evaluate
__device__ double evaluateCost(const double *solution, uint dim, double minX,
                               double maxX);

/// apply exponential crossover
/// \param population
/// \param mutVec
/// \param trial
/// \param dim
/// \param randState
/// \param CR
/// \param k
/// \param idx_dim it is the offset to know that which location in trial is
/// going to change
__device__ void crossoverExponential(const double *population,
                                     const double *mutVec, double *trial,
                                     int dim,
                                     curandStatePhilox4_32_10_t *randState,
                                     double CR, uint k, uint idx_dim);

/// apply binomial crossover
/// \param population
/// \param mutVec
/// \param trial
/// \param dim
/// \param randState
/// \param CR
/// \param k
/// \param idx_dim it is the offset to know that which location in trial is
/// going to change
__device__ void crossoverBinomial(const double *population, const double *mutVec,
                             double *trial, int dim,
                             curandStatePhilox4_32_10_t *randState, double CR,
                             uint k, uint idx_dim);

} // namespace gpu

#endif // DE_CUDA_DEGPU_CUH
