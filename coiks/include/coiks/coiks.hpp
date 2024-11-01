#ifndef COIKS_HPP
#define COIKS_HPP

#define PINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR

#include <cmath>
#include <memory>
#include <string>
#include <thread>

// Pinocchio
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// Eigen conversions
#include <eigen_conversions/eigen_msg.h>

// Eigen
#include <Eigen/Dense>

// Solver Base Class
#include <coiks/ik_solver.hpp>

// Inverse Jacobian IK Solver
#include <invj/invj_ik.hpp>

// Nonlinear Optimization IK Solver
#include <nlo/nlo_ik.hpp>

// Quadratic Programming IK Solver
#include <qp/qp_ik.hpp>

// Casadi
#include <casadi/casadi.hpp>

// SolverOptions
#include <coiks/solver_options.hpp>
#include <coiks/solution_set.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS : public IkSolver
  {
public:
    COIKS(const std::string &_urdf_file, const std::string &_base_link,
          const std::string &_tip_link, const std::string &_ik_solver,
          SolverOptions solver_opts, double _max_time = 10e-3,
          double _max_error = 1e-4, int _max_iter = 1e2, double _dt = 1.0);

    ~COIKS();

    int IkSolve(const VectorXd q_init, const pin::SE3 &X_EEdes, VectorXd &q_sol,
                IKSolutionSet::Solution &bestSolution);
    // int solveIk(const pin::SE3 X_EEdes);
    int updateFK(const VectorXd q_act);

    int getEEPose(pin::SE3 &x_B_Fid);
    int getFramePose(pin::SE3 &x_B_Fid, const pin::FrameIndex ee_id);
    int getFramePose(pin::SE3 &x_B_Fid, const std::string link_name);

    int  concurrentIkSolve(const VectorXd q_init, const pin::SE3 &X_EEdes,
                           VectorXd &q_out);
    void printStatistics();

    int         get_n_joints() { return n_joints_; }
    std::string get_solver_name() { return ik_solver_name_; }

private:
    //   Pinocchio variables
    pin::Model      model_;
    pin::Data       mdl_data_;
    pin::FrameIndex ee_id_;

    std::string urdf_file_;
    std::string base_link_;
    std::string ee_link_;
    std::string ik_solver_name_;

    int    n_joints_;
    int    max_iter_;
    int    succ_sol_tp_;
    int    succ_sol_nlo_;
    int    succ_sol_qp_;
    int    sol_id_;
    double max_time_;
    double max_error_;
    double dt_;
    bool   initialized_;

    VectorXd q_ul_;
    VectorXd q_ll_;

    std::vector<VectorXd> q_solutions_;
    std::vector<double>   m_solutions_;
    std::vector<double>   errors_;

    std::unique_ptr<IKSolutionSet> solutionSet;
    std::vector<std::string>       metrics;

    bool initialize();
    bool initialize_codcs();
    bool printModelInfo();

    template <typename T1, typename T2, typename T3>
    bool run3Solver(T1 &solver, T2 &other_solver1, T3 &other_solver2,
                    const VectorXd q_init, const pin::SE3 &X_EEdes, int id);

    template <typename T1, typename T2>
    bool run2Solver(T1 &solver, T2 &other_solver1, const VectorXd q_init,
                    const pin::SE3 &X_EEdes, int id);

    template <typename T>
    bool runMultiSolver(std::vector<std::unique_ptr<T>> &solvers,
                        const VectorXd q_init, const pin::SE3 &X_EEdes, int id);

    bool runINVJNLOIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runINVJQPIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runINVJALLIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runNLOINVJIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runNLOQPIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runNLOALLIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runQPINVJIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runQPNLOIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runQPALLIK(const VectorXd q_init, const pin::SE3 &X_EEdes);
    bool runINVJMULTIK(const VectorXd q_init, const pin::SE3 &X_EEdes, int id);

    bool initialize_coiks_invj_nlo();
    bool initialize_coiks_invj_qp();
    bool initialize_coiks_qp_nlo();
    bool initialize_codcs_invj_multi(int n_solvers);

    std::unique_ptr<INVJ_IkSolver<INVJ_BASE>> invj_solver_;
    std::unique_ptr<NLO_IkSolver<NLO_BASE>>   nlo_solver_;
    std::unique_ptr<QP_IkSolver<QP_BASE>>     qp_solver_;

    std::vector<std::unique_ptr<INVJ_IkSolver<INVJ_BASE>>> invj_multi_solvers_;

    std::thread solver1_, solver2_, solver3_;

    std::vector<std::thread> invj_multi_solvers_threads_;

    std::mutex mtx_;

    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_iksolve_time_;

    SolverOptions solver_opts_;
  };

  inline bool COIKS::runINVJNLOIK(const VectorXd  q_init,
                                  const pin::SE3 &X_EEdes)
  {
    return run2Solver(*invj_solver_.get(), *nlo_solver_.get(), q_init, X_EEdes,
                      1);
  }
  inline bool COIKS::runINVJQPIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run2Solver(*invj_solver_.get(), *qp_solver_.get(), q_init, X_EEdes,
                      1);
  }
  inline bool COIKS::runINVJALLIK(const VectorXd  q_init,
                                  const pin::SE3 &X_EEdes)
  {
    return run3Solver(*invj_solver_.get(), *nlo_solver_.get(),
                      *qp_solver_.get(), q_init, X_EEdes, 1);
  }

  inline bool COIKS::runNLOINVJIK(const VectorXd  q_init,
                                  const pin::SE3 &X_EEdes)
  {
    return run2Solver(*nlo_solver_.get(), *invj_solver_.get(), q_init, X_EEdes,
                      2);
  }

  inline bool COIKS::runNLOQPIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run2Solver(*nlo_solver_.get(), *qp_solver_.get(), q_init, X_EEdes,
                      2);
  }

  inline bool COIKS::runNLOALLIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run3Solver(*nlo_solver_.get(), *invj_solver_.get(),
                      *qp_solver_.get(), q_init, X_EEdes, 2);
  }

  inline bool COIKS::runQPINVJIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run2Solver(*qp_solver_.get(), *invj_solver_.get(), q_init, X_EEdes,
                      3);
  }

  inline bool COIKS::runQPNLOIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run2Solver(*qp_solver_.get(), *nlo_solver_.get(), q_init, X_EEdes,
                      3);
  }

  inline bool COIKS::runQPALLIK(const VectorXd q_init, const pin::SE3 &X_EEdes)
  {
    return run3Solver(*qp_solver_.get(), *invj_solver_.get(),
                      *nlo_solver_.get(), q_init, X_EEdes, 3);
  }

  inline bool COIKS::runINVJMULTIK(const VectorXd  q_init,
                                   const pin::SE3 &X_EEdes, int id)
  {
    return runMultiSolver(invj_multi_solvers_, q_init, X_EEdes, id);
  }

} // namespace coiks

#endif
