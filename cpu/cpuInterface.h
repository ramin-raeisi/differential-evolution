#ifndef RAST_LIBRARY_H
#define RAST_LIBRARY_H

#include <chrono>
#include <vector>

namespace cpu {

std::pair<double, std::vector<double>> runRastiginOnCPU(int dims, int popSize,
                                                        int interation);

}

#endif // RAST_LIBRARY_H
