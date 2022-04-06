#ifndef DE_CUDA_DIFFERENTIALEVOLUTION_CUH
#define DE_CUDA_DIFFERENTIALEVOLUTION_CUH

#include "deGPU.cuh"

namespace deGPU {

/// \brief This struct will hold the parameters for the results of the differential evolution algorithm.
struct optimizeResult {
    double cost;
    std::vector<double> solution;
};


class DifferentialEvolution {
public:
    /// \brief Constructor that will initialize the Differential Evolution, and allocate the memory for the population on the GPU
  DifferentialEvolution(int populationSize, int numberOfGeneration, int dim,
                        double CR, double F, double minX, double maxX,
                        uint blockDim);

    /// \brief Destructor that will free the memory on the GPU
    ~DifferentialEvolution();

    /// \brief Run the Differential Evolution algorithm
    optimizeResult optimize();

    /// \brief After optimization, this function will return the best solution based on the cost.
    std::pair<int,double> returnBestSolution();

private:
    double *populationD;
    double *newPopulationD;
    double *zD;
    double *rD;
    double *trialD;

    // cost variables
    double *costH;
    double *costD;

    // randomness
    void *globalRandStatesD;

    //main parameters
    int _populationSize;
    int _numberOfGeneration;
    int _dim;

    //differential evolution parameters
    double _pF;
    double _pCR;

    //constraints
    double _minX;
    double _maxX;

    uint _blockDim;
};

}//namespace deGPU

#endif//DE_CUDA_DIFFERENTIALEVOLUTION_CUH
