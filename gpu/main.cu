#include <chrono>
#include <iomanip>
#include <iostream>

#include "cpuInterface.h"
#include "src/DifferentialEvolution.cuh"

static constexpr int repeatUnderScore = 145;
static const std::string columSeparator = " | ";

void printResult(double minCost, const std::vector<double> &bestSolution,
                 long totalDuration, long kernelDuration, int cpuTime);

void printHeader(int populationSize, int maxIterations, int cpuTime);

long getCpuTime(int dims, int popSize, int numberOfIterations);

int main() {

  using namespace std::chrono;
  // differential evolution parameters
  static constexpr int populationSize = 4000;
  static constexpr int maxIterations = 2000;
  static constexpr int dimension = 5;
  static constexpr double CR = 0.9;
  static constexpr double F = 0.8;
  static constexpr std::pair<double, double> bounds = {-5.12, 5.12};

  int kernelBlockSize = 32;

  static constexpr int testingIterationCount =
      5; // number of iteration for the for loop

  const float cpuTime = getCpuTime(dimension, populationSize, maxIterations);

  printHeader(populationSize, maxIterations, cpuTime);

  for (int i = 0; i < testingIterationCount; i++) {
    auto time1 = high_resolution_clock::now();
    deGPU::DifferentialEvolution de(populationSize, maxIterations, dimension,
                                    CR, F, bounds.first, bounds.second,
                                    kernelBlockSize);

    auto kernel_tick = high_resolution_clock::now();
    auto result = de.optimize();
    auto kernel_tock = high_resolution_clock::now();
    auto kernel_duration =
        duration_cast<milliseconds>(kernel_tock - kernel_tick).count();

    auto time2 = high_resolution_clock::now();
    // get time difference in milliseconds
    auto duration = duration_cast<milliseconds>(time2 - time1).count();
    printResult(result.cost, result.solution, duration, kernel_duration,
                cpuTime);
  }
  std::cout << std::string(repeatUnderScore, '-') << std::endl;
}

void printHeader(int populationSize, int maxIterations, int cpuTime) {
  std::cout << std::string(repeatUnderScore, '-') << std::endl;
  std::cout << "configuration: "
            << "\n"
            << std::left << std::setw(20)
            << "populationSize: " << populationSize << "\n"
            << std::setw(20) << "maxIterations: " << maxIterations << "\n"
            << std::setw(20) << "cpu_time: " << cpuTime << "ms" << std::left
            << "\n";
  std::cout << std::string(repeatUnderScore, '-') << std::endl;
  std::cout << std::setw(15) << "Kernel time" << columSeparator << std::setw(15)
            << "Total Time" << columSeparator;
  std::cout << std::setw(15) << "speedup" << columSeparator;
  std::cout << std::setw(15) << "Min cost" << columSeparator << std::setw(40)
            << "Best Solution" << std::endl;
  std::cout << std::string(repeatUnderScore, '-') << std::endl;
}

void printResult(double minCost, const std::vector<double> &bestSolution,
                 long totalDuration, long kernelDuration, int cpuTime) {
  std::cout << std::setw(15) << std::to_string(kernelDuration) + " ms"
            << columSeparator << std::setw(15)
            << std::to_string(totalDuration) + " ms" << columSeparator;
  std::cout << std::setw(15)
            << std::to_string(static_cast<int>(cpuTime / totalDuration)) + "x"
            << columSeparator;
  // print the best solution
  std::cout << std::setw(15) << minCost << columSeparator;
  for (double i : bestSolution) {
    std::cout << std::setw(14) << i;
  }
  std::cout << std::endl;
}

long getCpuTime(int dims, int popSize, int numberOfIterations) {
  std::cout << std::string(repeatUnderScore, '-') << std::endl;
  std::cout << "Please wait --> running differential evolution on CPU"
            << std::endl;
  using namespace std::chrono;
  auto cpu_tick = high_resolution_clock::now();
  auto [bestCost, bestAgent] = cpu::runRastiginOnCPU(dims, popSize, numberOfIterations);
  auto cpu_tock = high_resolution_clock::now();
  std::cout << std::left << "\n"
            << std::setw(20) << "best cost: " << bestCost << std::endl;
  // best agent
  std::cout << std::left << std::setw(20) << "best agent: ";
  for (auto &agent : bestAgent) {
    std::cout << std::setw(14) << agent;
  }
  std::cout << std::endl;
  return std::chrono::duration_cast<std::chrono::milliseconds>(cpu_tock -
                                                               cpu_tick)
      .count();
}