#ifndef SOLVER_OPTIONS_HPP
#define SOLVER_OPTIONS_HPP

// Eigen
#include <Eigen/Dense>
// C++
#include <map>
#include <vector>

using namespace Eigen;

namespace coiks
{

  class SolverOptions
  {
public:
    std::string error_type_;
    std::string pinv_method_;
    std::string step_size_method_;
    std::string seed_method_;
    std::string limiting_method_;
    std::string solve_mode_;

    // INVJ Variables
    double invj_Ke1_;

    // INVJ Multi-solver variables
    int    invj_multi_n_solvers_;
    int    invj_max_stagnation_iterations_;
    double invj_improvement_threshold_;

    // NLO variables
    bool                nlo_concurrent_;
    int                 nlo_concurrent_iterations_;
    std::string         nlo_linear_solver_;
    std::string         nlo_error_type_;
    std::vector<double> cost_coeff_;
    std::string         nlo_warm_start_;

    // QP variables
    // Task-based coefficients
    double qp_K_t1_;
    double qp_Kr_t1_;
    // Priority-based coefficients
    double qp_Kw_p1_;
    double qp_Kd_p1_;
    bool   qp_warm_start_;

    // Logging variables
    int                                verb_level_;
    std::string                        time_stats_;
    std::map<std::string, std::string> other_opts_;
  };
} // namespace coiks
#endif