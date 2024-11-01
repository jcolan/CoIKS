#ifndef IKSOLUTION_SET_HPP
#define IKSOLUTION_SET_HPP

// Eigen
#include <Eigen/Dense>
// C++
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <mutex>
#include <memory>

using JointConfiguration = Eigen::VectorXd;

enum class OptimizationGoal
{
  Maximize,
  Minimize
};

class IKSolutionSet
{
  public:
  struct Solution
  {
    JointConfiguration                      configuration;
    std::unordered_map<std::string, double> metrics;
    double                                  elapsed_time;
    int                                     solver_id;
    int                                     solution_id;
  };

  std::vector<Solution>    solutions;
  std::vector<std::string> metricNames;
  mutable std::mutex       mutex;

  public:
  explicit IKSolutionSet(
      const std::vector<std::string> &initialMetricNames = {})
      : metricNames(initialMetricNames)
  {
  }

  void addMetric(const std::string &metricName)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (std::find(metricNames.begin(), metricNames.end(), metricName) ==
        metricNames.end())
    {
      metricNames.push_back(metricName);
    }
  }

  void addSolution(const JointConfiguration &                     config,
                   const std::unordered_map<std::string, double> &metrics,
                   double elapsed_time, int solver_id, int solution_id)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Solution solution{config, {}, elapsed_time, solver_id, solution_id};
    for (const auto &metricName : metricNames)
    {
      auto it = metrics.find(metricName);
      if (it != metrics.end())
      {
        solution.metrics[metricName] = it->second;
      }
      else
      {
        solution.metrics[metricName] =
            0.0; // Default value if metric not provided
      }
    }
    solutions.push_back(std::move(solution));
  }

  void addSolution(const Solution &solution)
  {
    std::lock_guard<std::mutex> lock(mutex);

    // Create a new Solution object to store in the solutions vector
    Solution newSolution{solution.configuration,
                         {},
                         solution.elapsed_time,
                         solution.solver_id,
                         solution.solution_id};

    // Copy metrics from the input solution, using only the metrics defined in
    // metricNames
    for (const auto &metricName : metricNames)
    {
      auto it = solution.metrics.find(metricName);
      if (it != solution.metrics.end())
      {
        newSolution.metrics[metricName] = it->second;
      }
      else
      {
        newSolution.metrics[metricName] =
            0.0; // Default value if metric not provided
      }
    }

    solutions.push_back(std::move(newSolution));
  }

  void clearSolutions()
  {
    std::lock_guard<std::mutex> lock(mutex);
    solutions.clear();
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return solutions.size();
  }

  std::vector<std::string> getMetricNames() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return metricNames;
  }

  std::vector<JointConfiguration> getConfigurations() const
  {
    std::lock_guard<std::mutex>     lock(mutex);
    std::vector<JointConfiguration> configs;
    configs.reserve(solutions.size());
    for (const auto &solution : solutions)
    {
      configs.push_back(solution.configuration);
    }
    return configs;
  }

  std::vector<double> getMetricValues(const std::string &metricName) const
  {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<double>         values;
    values.reserve(solutions.size());
    for (const auto &solution : solutions)
    {
      auto it = solution.metrics.find(metricName);
      if (it != solution.metrics.end())
      {
        values.push_back(it->second);
      }
      else
      {
        throw std::runtime_error("Metric not found: " + metricName);
      }
    }
    return values;
  }

  std::vector<double> getSolutionTimes() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<double>         elapsed_times;
    elapsed_times.reserve(solutions.size());
    for (const auto &solution : solutions)
    {
      elapsed_times.push_back(solution.elapsed_time);
    }
    return elapsed_times;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex);
    solutions.clear();
  }

  std::pair<bool, std::pair<JointConfiguration,
                            std::unordered_map<std::string, double>>>
  getSolutionWithMetrics(size_t index) const
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (index >= solutions.size())
    {
      return {false, {}};
    }
    return {true, {solutions[index].configuration, solutions[index].metrics}};
  }

  void addSolutions(
      const std::vector<JointConfiguration> &                     configs,
      const std::vector<std::unordered_map<std::string, double>> &metricSets,
      const std::vector<double> &                                 elapsed_times,
      const std::vector<int> &solver_ids, const std::vector<int> &solution_ids)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (configs.size() != metricSets.size() ||
        configs.size() != elapsed_times.size() ||
        configs.size() != solver_ids.size())
    {
      throw std::invalid_argument("Mismatch in input vector sizes");
    }
    solutions.reserve(solutions.size() + configs.size());
    for (size_t i = 0; i < configs.size(); ++i)
    {
      Solution solution{
          configs[i], {}, elapsed_times[i], solver_ids[i], solution_ids[i]};
      for (const auto &metricName : metricNames)
      {
        auto it = metricSets[i].find(metricName);
        if (it != metricSets[i].end())
        {
          solution.metrics[metricName] = it->second;
        }
        else
        {
          solution.metrics[metricName] =
              0.0; // Default value if metric not provided
        }
      }
      solutions.push_back(std::move(solution));
    }
  }

  std::pair<bool, Solution> getBestSolution(const std::string &metricName,
                                            OptimizationGoal   goal) const
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (solutions.empty())
    {
      // std::cout << "No solutions available" << std::endl;
      return {false, {}};
    }

    try
    {
      auto it = std::max_element(
          solutions.begin(), solutions.end(),
          [&metricName, goal](const Solution &a, const Solution &b) {
            auto metricA = a.metrics.find(metricName);
            auto metricB = b.metrics.find(metricName);

            if (metricA == a.metrics.end() || metricB == b.metrics.end())
            {
              throw std::runtime_error("Metric not found: " + metricName);
            }

            if (goal == OptimizationGoal::Maximize)
            {
              return metricA->second < metricB->second;
            }
            else
            {
              return metricA->second > metricB->second;
            }
          });

      return {true, *it}; // Return a copy of the best solution
    }
    catch (const std::exception &e)
    {
      std::cout << "Exception caught in getBestSolution: " << e.what()
                << std::endl;
      return {false, {}};
    }
  }

  void printAllSolutions(std::ostream &out = std::cout) const
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (solutions.empty())
    {
      out << "No solutions available." << std::endl;
      return;
    }

    out << "Total solutions: " << solutions.size() << std::endl;
    out << std::string(50, '-') << std::endl;

    for (size_t i = 0; i < solutions.size(); ++i)
    {
      const auto &solution = solutions[i];

      out << "Solution " << i + 1 << ":" << std::endl;
      out << "Configuration: " << solution.configuration.transpose()
          << std::endl;

      out << "Metrics:" << std::endl;
      for (const auto &metric : solution.metrics)
      {
        out << "  " << metric.first << ": " << metric.second << std::endl;
      }

      out << "Elapsed time: " << solution.elapsed_time << " seconds"
          << std::endl;
      out << "Solver ID: " << solution.solver_id << std::endl;
      out << "Solution ID: " << solution.solution_id << std::endl;
      out << std::string(50, '-') << std::endl;
    }
  }

  void printBestSolution(const std::string &metricName, OptimizationGoal goal,
                         std::ostream &out = std::cout) const
  {
    std::pair<bool, Solution> bestSolutionPair;
    size_t                    totalSolutions;

    {
      std::lock_guard<std::mutex> lock(mutex);
      totalSolutions = solutions.size();
      if (totalSolutions == 0)
      {
        out << "No solutions available." << std::endl;
        return;
      }
    }

    bestSolutionPair = getBestSolution(metricName, goal);

    if (!bestSolutionPair.first)
    {
      out << "No best solution found." << std::endl;
      return;
    }

    const auto &bestSolution = bestSolutionPair.second;
    const auto &bestConfiguration = bestSolution.configuration;
    auto        bestMetricIt = bestSolution.metrics.find(metricName);

    if (bestMetricIt == bestSolution.metrics.end())
    {
      out << "Error: Best metric not found in solution." << std::endl;
      return;
    }

    double bestMetricValue = bestMetricIt->second;

    out << "Best Solution (based on " << metricName << ", "
        << (goal == OptimizationGoal::Maximize ? "maximized" : "minimized")
        << "):" << std::endl;
    out << std::string(50, '-') << std::endl;

    out << "Total Solutions: " << totalSolutions << std::endl;
    out << "Configuration: " << bestConfiguration.transpose() << std::endl;

    out << "Metrics:" << std::endl;
    for (const auto &metric : bestSolution.metrics)
    {
      out << "  " << metric.first << ": " << metric.second << std::endl;
    }
    out << "Elapsed time: " << bestSolution.elapsed_time << " seconds"
        << std::endl;
    out << "Solver ID: " << bestSolution.solver_id << std::endl;
    out << "Solution ID: " << bestSolution.solution_id << std::endl;

    out << metricName << " value: " << bestMetricValue << std::endl;
    out << std::string(50, '-') << std::endl;
  }
};

#endif
