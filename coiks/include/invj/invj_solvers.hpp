#ifndef INVJ_SOLVERS_HPP
#define INVJ_SOLVERS_HPP

// Source HPP
#include <coiks/solver_base.hpp>
#include <coiks/solver_options.hpp>
#include <coiks/solution_set.hpp>
// Pinocchio
#include <pinocchio/algorithm/joint-configuration.hpp>
// C++
#include <chrono>
#include <random> // Add this include for random number generation

namespace coiks
{

  class COIKS;

  class INVJ_BASE : public SolverBase
  {
    friend class coiks::COIKS;

public:
    INVJ_BASE(const pin::Model &_model, const pin::FrameIndex &_Fid,
              SolverOptions _solver_opts, const double _max_time,
              const double _max_error, const int _max_iter = 1000,
              const double _dt = 1, const int _id = 0);
    ~INVJ_BASE();

    int IkSolve(const VectorXd &q_init, const pin::SE3 &X_des, VectorXd &q_sol);
    int IkSolveSet(const VectorXd &q_init, const pin::SE3 &X_des,
                   VectorXd &q_sol, IKSolutionSet::Solution &solution);

    double computeManipulability(const Eigen::MatrixXd &J,
                                 double                 epsilon = 1e-10);

    MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon,
                           std::string method);
    MatrixXd weightedPseudoInverse(const Eigen::MatrixXd &a,
                                   const Eigen::VectorXd  w);
    void     GetOptions(SolverOptions _solver_opts);

    int computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                           Vector6d &err_ee);
    int computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                           std::string err_type, Vector6d &err_ee);
    int computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                              double &err_pos_ee);

    int computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
                              double &err_ori_ee);

    inline void abort() { aborted_ = true; }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; };
    inline void set_max_error(double _max_error) { max_error_ = _max_error; };

private:
    pin::Model      mdl_;
    pin::Data       mdl_data_;
    pin::FrameIndex id_Fee_;
    pin::FrameIndex id_Fprercm_;
    pin::FrameIndex id_Fpostrcm_;

    int n_q_;
    int max_iter_;
    int verb_level_;
    int id_;

    double max_time_;
    double max_error_;
    double dt_;

    bool aborted_;
    bool success_;

    VectorXd q_ul_;
    VectorXd q_ll_;

    std::string error_type_;
    std::string pinv_method_;
    std::string limiting_method_;
    std::string step_size_method_;
    std::string seed_method_;
    std::string solve_mode_;

    double Kee_;

    // Jacobians
    MatrixXd Jb_task_1_; // EE Tracking
    MatrixXd pinv_Jtask_1_;

    std::default_random_engine       rng_;          // Random number generator
    std::normal_distribution<double> distribution_; // Normal distribution

    // Convergence monitoring variables
    double previous_error_;
    int    stagnation_counter_;
    int    max_stagnation_iterations_;
    double improvement_threshold_;
  };

  void INVJ_BASE::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;

    // Solver parameters
    error_type_ = so.error_type_;
    pinv_method_ = so.pinv_method_;
    limiting_method_ = so.limiting_method_;
    step_size_method_ = so.step_size_method_;
    seed_method_ = so.seed_method_;
    solve_mode_ = so.solve_mode_;

    // INVJ variables
    Kee_ = so.invj_Ke1_;

    // MultiINVJ variables
    max_stagnation_iterations_ = so.invj_max_stagnation_iterations_;
    improvement_threshold_ = so.invj_improvement_threshold_;

    // Logging
    verb_level_ = so.verb_level_;

    std::cout << "\n------\nINVJ " << id_ << " Options summary:" << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Coeff. Kee (EE error): " << Kee_ << std::endl;
    std::cout << "Pinv method: " << pinv_method_ << std::endl;
    std::cout << "Limiting method: " << limiting_method_ << std::endl;
    std::cout << "Step size method: " << step_size_method_ << std::endl;
    std::cout << "Seed method: " << seed_method_ << std::endl;
    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;
    std::cout << "Solver mode: " << solve_mode_ << std::endl;
  }

  INVJ_BASE::INVJ_BASE(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
                       SolverOptions _solver_opts, const double _max_time,
                       const double _max_error, const int _max_iter,
                       const double _dt, const int _id)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt), distribution_(_dt, 0.5), id_(_id)
  {
    std::cout << "----------\nInitializing IK solver COIKS-INVJ" << std::endl;

    GetOptions(_solver_opts);

    mdl_data_ = pin::Data(mdl_);
    q_ll_ = mdl_.lowerPositionLimit;
    q_ul_ = mdl_.upperPositionLimit;

    //* Seed the random number generator
    rng_.seed(0);

    if (error_type_ == "only_p")
    {
      Jb_task_1_.resize(3, n_q_);
      pinv_Jtask_1_.resize(n_q_, 3);
    }
    else
    {
      Jb_task_1_.resize(6, n_q_);
      pinv_Jtask_1_.resize(n_q_, 6);
    }
  }

  INVJ_BASE::~INVJ_BASE() {}

  double INVJ_BASE::computeManipulability(const Eigen::MatrixXd &J,
                                          double                 epsilon)
  {
    Eigen::MatrixXd                                JJT = J * J.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(JJT);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

    double manipulability = 1.0;
    int    validEigenvalues = 0;

    for (int i = 0; i < eigenvalues.size(); ++i)
    {
      if (eigenvalues(i) > epsilon)
      {
        manipulability *= eigenvalues(i);
        validEigenvalues++;
      }
    }

    return (validEigenvalues > 0) ? std::sqrt(manipulability) : 0.0;
  }

  MatrixXd INVJ_BASE::pseudoInverse(const Eigen::MatrixXd &a, double epsilon,
                                    std::string method)
  {
    if (method == "svd")
    {
      // Taken from https://armarx.humanoids.kit.edu/pinv_8hh_source.html
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU |
                                                   Eigen::ComputeThinV);

      double tolerance = epsilon * std::max(a.cols(), a.rows()) *
                         svd.singularValues().array().abs()(0);

      return svd.matrixV() *
             (svd.singularValues().array().abs() > tolerance)
                 .select(svd.singularValues().array().inverse(), 0)
                 .matrix()
                 .asDiagonal() *
             svd.matrixU().adjoint();
    }
    else if (method == "cod")
    {
      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(a);
      cod.setThreshold(epsilon);
      return cod.pseudoInverse();
    }
    else if (method == "qr")
    {
      return a.completeOrthogonalDecomposition().pseudoInverse();
    }
    else if (method == "ldlt")
    {
      if (a.rows() >= a.cols())
      {
        return (a.transpose() * a).ldlt().solve(a.transpose());
      }
      else
      {
        return a.transpose() *
               (a * a.transpose())
                   .ldlt()
                   .solve(MatrixXd::Identity(a.rows(), a.rows()));
      }
    }
    else
    {
      std::cout << "[INVJ " << id_
                << "] Invalid method for pseudoinverse. Using SVD."
                << std::endl;
      return pseudoInverse(a, epsilon, "svd");
    }
  }

  // Taken from https://armarx.humanoids.kit.edu/pinv_8hh_source.html
  MatrixXd INVJ_BASE::weightedPseudoInverse(const Eigen::MatrixXd &a,
                                            const Eigen::VectorXd  w)
  {
    int lenght = w.size();

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Winv(lenght);
    Winv = w.asDiagonal().inverse(); // diag(1./w)

    Eigen::MatrixXd tmp(lenght, lenght);
    tmp = pseudoInverse(a * Winv * a.transpose(), 10E-10, "svd");

    return Winv * a.transpose() * tmp;
  }

  int INVJ_BASE::computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                                    Vector6d &err_ee)
  {

    //? Using log6
    pin::SE3 B_Xerr(X_d.act(B_X_EE.inverse()));
    err_ee = pin::log6(B_Xerr).toVector();
    return 0;
  }

  int INVJ_BASE::computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                                    std::string err_type, Vector6d &err_ee)
  {

    //* Computation Errors
    if (err_type == "log6")
    {
      //? Using log6
      pin::SE3 B_Xerr(X_d.act(B_X_EE.inverse()));
      err_ee = pin::log6(B_Xerr).toVector();
    }
    else if (err_type == "log3")
    {
      //? Using log3
      Vector3d err_tr = (X_d.translation() - B_X_EE.translation());
      Vector3d err_rot =
          pin::log3(X_d.rotation() * B_X_EE.rotation().transpose());
      err_ee << err_tr, err_rot;
    }
    else if (err_type == "euler")
    {
      //? Using euclidean distances
      Vector3d err_tr = (X_d.translation() - B_X_EE.translation());
      // ? Using angular distances
      Vector3d err_rot =
          (X_d.rotation() - B_X_EE.rotation()).eulerAngles(0, 1, 2);
      err_ee << err_tr, err_rot;
    }
    else if (err_type == "only_p")
    {
      //? Using only p error
      Vector3d err_tr = (X_d.translation() - B_X_EE.translation());
      Vector3d err_rot = Vector3d::Zero();
      err_ee << err_tr, err_rot;
    }
    else
    {
      std::cout << "[INVJ " << id_
                << "] Invalid error type for EE task. Using log6." << std::endl;
      return computeEETaskError(B_X_EE, X_d, "log6", err_ee);
    }

    return 0;
  }

  int INVJ_BASE::IkSolve(const VectorXd &q_init, const pin::SE3 &X_d,
                         VectorXd &q_sol)
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time =
        std::chrono::high_resolution_clock::now();

    q_sol = q_init;

    // Seed initialization
    VectorXd q_it;
    if (seed_method_ == "neutral")
    {
      q_it = pin::neutral(mdl_);
    }
    else if (seed_method_ == "middle")
    {
      q_it = (mdl_.upperPositionLimit + mdl_.lowerPositionLimit) / 2.0;
    }
    else if (seed_method_ == "random")
    {
      q_it = pin::randomConfiguration(mdl_);
    }
    else
    {
      q_it = q_init;
    }

    if (verb_level_ >= 1)
      std::cout << "\n[INVJ " << id_
                << "] Solving with optimizer INVJ IK and Initial guess:"
                << q_it.transpose() << std::endl;

    Vector6d err_ee;
    Vector6d err_log3_ee;
    double   err_rcm = 0.0;
    VectorXd q_dot(mdl_.nv);
    double   manip = 0;

    pin::Data::Matrix6x B_Jb_Fee(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fprercm(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fpostrcm(6, n_q_);

    MatrixXd B_Jb_Frcm(1, n_q_);

    B_Jb_Fee.setZero();
    B_Jb_Fprercm.setZero();
    B_Jb_Fpostrcm.setZero();
    B_Jb_Frcm.setZero();

    pin::SE3 B_X_EE, B_x_Fprercm, B_x_Fpostrcm;

    success_ = false;

    Jb_task_1_.setZero();

    //* Updating FK
    pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
    B_X_EE = mdl_data_.oMf[id_Fee_];

    //* Compute initial error
    computeEETaskError(B_X_EE, X_d, err_ee);
    previous_error_ = err_ee.norm();
    stagnation_counter_ = 0;

    int iteration = 0;

    for (int it = 0; it < max_iter_; it++)
    {

      auto   current_time = std::chrono::high_resolution_clock::now();
      double time_elapsed_s =
          std::chrono::duration<double>(current_time - start_time).count();
      iteration++;

      if (time_elapsed_s > max_time_ || aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_
                    << "]  Aborted by other solvers or Maximum time exceeded:"
                    << time_elapsed_s << std::endl;
        break;
      }

      // Step size update
      if (step_size_method_ == "gaussian")
        dt_ = distribution_(rng_);
      else
        dt_ = dt_; // Use fixed step size for all other cases

      // Computing EE Jacobian
      B_Jb_Fee.setZero();
      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fee_,
                                pin::ReferenceFrame::WORLD, B_Jb_Fee);

      // Defining Task Jacobians
      //? Tracking in 3D?
      Jb_task_1_ = (error_type_ == "only_p") ? B_Jb_Fee.topRows(3) : B_Jb_Fee;

      // Computing pseudoinverses
      pinv_Jtask_1_ = pseudoInverse(Jb_task_1_, 1e-10, pinv_method_);

      q_dot.noalias() = pinv_Jtask_1_ * Kee_ * err_ee;

      // VectorXd q_prev = q_it;
      q_it = pin::integrate(mdl_, q_it, q_dot * dt_);

      if (limiting_method_ == "clamping")
      {
        //? Clamps solution to joint limits
        q_it = q_it.cwiseMax(q_ll_).cwiseMin(q_ul_);
      }
      else if (limiting_method_ == "random")
      {
        //? Generate a random configuration if the solution is outside limits
        // Check if any joint is outside its limits
        bool outside_limits = false;
        for (int i = 0; i < n_q_; ++i)
        {
          if (q_it[i] < q_ll_[i] || q_it[i] > q_ul_[i])
          {
            outside_limits = true;
            break;
          }
        }

        // If outside limits, generate a random configuration
        if (outside_limits)
        {
          q_it = pin::randomConfiguration(mdl_);
        }
      }

      //* Computing FK
      pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
      B_X_EE = mdl_data_.oMf[id_Fee_];

      //* Computing EE error
      computeEETaskError(B_X_EE, X_d, err_ee);
      double current_error = err_ee.norm();

      if (verb_level_ >= 1)
      {
        std::cout << "[INVJ " << id_ << "] [" << it
                  << "] error EE: " << err_ee.norm() << " : "
                  << err_ee.transpose() << std::endl;
        std::cout << "[INVJ " << id_ << "] q_it: " << q_it.transpose()
                  << std::endl;
      }

      //* Verifying convergence
      if (err_ee.norm() < max_error_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_ << "]  Solution found" << std::endl;
        success_ = true;
        break;
      }

      //* Check for improvement
      if (previous_error_ - current_error > improvement_threshold_)
      {
        // Improvement found, reset counter
        stagnation_counter_ = 0;
      }
      else
      {
        // No significant improvement, increment counter
        stagnation_counter_++;
      }

      // If stagnation detected, reinitialize randomly
      if (stagnation_counter_ >= max_stagnation_iterations_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_
                    << "] Stagnation detected. Reinitializing randomly."
                    << std::endl;

        q_it = pin::randomConfiguration(mdl_);
        stagnation_counter_ = 0;
      }

      // Update previous error for next iteration
      previous_error_ = current_error;
    }

    auto   current_time = std::chrono::high_resolution_clock::now();
    double time_elapsed_s =
        std::chrono::duration<double>(current_time - start_time).count();

    if (success_)
    {
      q_sol = q_it;

      return 0;
    }
    else
    {

      if (verb_level_ >= 1)
        std::cout << "\n[INVJ " << id_
                  << "] Warning: the iterative algorithm has not reached "
                     "convergence to the desired precision after "
                  << iteration << " iterations." << std::endl;
      return 1;
    }
  }

  int INVJ_BASE::IkSolveSet(const VectorXd &q_init, const pin::SE3 &X_d,
                            VectorXd &q_sol, IKSolutionSet::Solution &solution)
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time =
        std::chrono::high_resolution_clock::now();

    q_sol = q_init;

    // Seed initialization
    VectorXd q_it;
    if (seed_method_ == "neutral")
    {
      q_it = pin::neutral(mdl_);
    }
    else if (seed_method_ == "middle")
    {
      q_it = (mdl_.upperPositionLimit + mdl_.lowerPositionLimit) / 2.0;
    }
    else if (seed_method_ == "random")
    {
      q_it = pin::randomConfiguration(mdl_);
    }
    else
    {
      q_it = q_init;
    }

    if (verb_level_ >= 1)
      std::cout << "\n[INVJ " << id_
                << "] Solving with optimizer INVJ IK and Initial guess:"
                << q_it.transpose() << std::endl;

    Vector6d err_ee;
    Vector6d err_log3_ee;
    double   err_rcm = 0.0;
    VectorXd q_dot(mdl_.nv);
    double   manip = 0;

    pin::Data::Matrix6x B_Jb_Fee(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fprercm(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fpostrcm(6, n_q_);

    MatrixXd B_Jb_Frcm(1, n_q_);

    B_Jb_Fee.setZero();
    B_Jb_Fprercm.setZero();
    B_Jb_Fpostrcm.setZero();
    B_Jb_Frcm.setZero();

    pin::SE3 B_X_EE, B_x_Fprercm, B_x_Fpostrcm;

    success_ = false;

    Jb_task_1_.setZero();

    //* Updating FK
    pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
    B_X_EE = mdl_data_.oMf[id_Fee_];

    //* Compute initial error
    computeEETaskError(B_X_EE, X_d, err_ee);
    previous_error_ = err_ee.norm();
    stagnation_counter_ = 0;

    for (int it = 0; it < max_iter_; it++)
    {

      auto   current_time = std::chrono::high_resolution_clock::now();
      double time_elapsed_s =
          std::chrono::duration<double>(current_time - start_time).count();

      if (time_elapsed_s > max_time_ || aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_
                    << "]  Aborted by other solvers or Maximum time exceeded: "
                    << time_elapsed_s << std::endl;
        break;
      }

      //* Step size update
      if (step_size_method_ == "gaussian")
        dt_ = distribution_(rng_);
      else
        dt_ = dt_; // Use fixed step size for all other cases

      //* Computing EE Jacobian
      B_Jb_Fee.setZero();
      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fee_,
                                pin::ReferenceFrame::WORLD, B_Jb_Fee);

      //* Defining Task Jacobians
      //? Tracking in 3D?
      Jb_task_1_ = (error_type_ == "only_p") ? B_Jb_Fee.topRows(3) : B_Jb_Fee;

      //* Computing pseudoinverses
      pinv_Jtask_1_ = pseudoInverse(Jb_task_1_, 1e-10, pinv_method_);

      q_dot.noalias() = pinv_Jtask_1_ * Kee_ * err_ee;

      if (verb_level_ >= 4)
      {
        std::cout << "[INVJ " << id_ << "] [" << it
                  << "] Jb_task_1_: " << Jb_task_1_ << std::endl;
        std::cout << "[INVJ " << id_ << "] [" << it
                  << "] pinv_Jtask_1_: " << pinv_Jtask_1_ << std::endl;
        std::cout << "[INVJ " << id_ << "] [" << it
                  << "] q_dot: " << q_dot.transpose() << std::endl;
      }

      q_it = pin::integrate(mdl_, q_it, q_dot * dt_);

      if (limiting_method_ == "clamping")
      {
        //? Clamps solution to joint limits
        q_it = q_it.cwiseMax(q_ll_).cwiseMin(q_ul_);
      }
      else if (limiting_method_ == "random")
      {
        //? Generate a random configuration if the solution is outside limits
        // Check if any joint is outside its limits
        bool outside_limits = false;
        for (int i = 0; i < n_q_; ++i)
        {
          if (q_it[i] < q_ll_[i] || q_it[i] > q_ul_[i])
          {
            outside_limits = true;
            break;
          }
        }

        // If outside limits, generate a random configuration
        if (outside_limits)
        {
          q_it = pin::randomConfiguration(mdl_);
        }
      }

      //* Computing FK
      pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
      B_X_EE = mdl_data_.oMf[id_Fee_];

      if (verb_level_ >= 5)
      {
        std::cout << "[INVJ " << id_ << "] [" << it << "] B_X_EE: " << B_X_EE
                  << std::endl;
      }

      //* Computing EE error
      computeEETaskError(B_X_EE, X_d, err_ee);
      double current_error = err_ee.norm();

      if (verb_level_ >= 1)
      {
        std::cout << "[INVJ " << id_ << "] [" << it
                  << "] error EE: " << err_ee.norm() << " : "
                  << err_ee.transpose() << std::endl;
        std::cout << "[INVJ " << id_ << "] q_it: " << q_it.transpose()
                  << std::endl;
      }

      //* Verifying convergence
      if (err_ee.norm() < max_error_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_ << "]  Solution found" << std::endl;
        success_ = true;
        break;
      }

      //* Check for improvement
      if (previous_error_ - current_error > improvement_threshold_)
      {
        // Improvement found, reset counter
        stagnation_counter_ = 0;
      }
      else
      {
        // No significant improvement, increment counter
        stagnation_counter_++;
      }

      // If stagnation detected, reinitialize randomly
      if (stagnation_counter_ >= max_stagnation_iterations_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ " << id_
                    << "] Stagnation detected. Reinitializing randomly."
                    << std::endl;

        q_it = pin::randomConfiguration(mdl_);
        stagnation_counter_ = 0;
      }

      // Update previous error for next iteration
      previous_error_ = current_error;
    }

    auto   current_time = std::chrono::high_resolution_clock::now();
    double time_elapsed_s =
        std::chrono::duration<double>(current_time - start_time).count();

    if (success_)
    {
      q_sol = q_it;

      if (solve_mode_ == "distance")
      {
        solution.configuration = q_sol;
        solution.metrics["error"] = err_ee.norm();
        solution.metrics["distance"] =
            (q_sol - q_init)
                .norm(); // Distance between initial and final configuration
        solution.metrics["time"] = time_elapsed_s;
      }
      else if (solve_mode_ == "manipulability")
      {
        solution.configuration = q_sol;
        solution.metrics["error"] = err_ee.norm();
        solution.metrics["manipulability"] = computeManipulability(B_Jb_Fee);
        solution.metrics["time"] = time_elapsed_s;
      }
      else
      {
        solution.configuration = q_sol;
        solution.metrics["error"] = err_ee.norm();
        solution.metrics["time"] = time_elapsed_s;
      }

      return 0;
    }
    else
    {

      if (verb_level_ >= 1)
        std::cout << "\n[INVJ " << id_
                  << "] Warning: the iterative algorithm has not reached "
                     "convergence to the desired precision after "
                  << max_iter_ << " iterations." << std::endl;
      return 1;
    }
  }

  int INVJ_BASE::computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                                       double &err_pos_ee)
  {
    pin::framesForwardKinematics(mdl_, mdl_data_, q);

    pin::SE3 B_H_Fee_ = mdl_data_.oMf[id_Fee_];

    //* Computation Errors
    //? Using log3
    err_pos_ee = (H_d.translation() - B_H_Fee_.translation()).norm();

    return 0;
  }

  int INVJ_BASE::computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
                                       double &err_ori_ee)
  {
    pin::framesForwardKinematics(mdl_, mdl_data_, q);

    pin::SE3 B_H_Fee_ = mdl_data_.oMf[id_Fee_];

    //* Computation Errors
    Quaterniond B_q_act_Fee(B_H_Fee_.rotation());
    Quaterniond B_q_des_Fee(H_d.rotation());
    err_ori_ee = B_q_act_Fee.angularDistance(B_q_des_Fee);
    return 0;
  }

} // namespace coiks

#endif