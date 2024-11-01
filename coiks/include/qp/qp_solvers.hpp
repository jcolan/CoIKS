#ifndef QP_SOLVERS_HPP
#define QP_SOLVERS_HPP

#include <coiks/solver_base.hpp>
#include <coiks/solver_options.hpp>
// Casadi
#include <casadi/casadi.hpp>
// C++
#include <map>
#include <random> // Add this include for random number generation
// Pinocchio
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace ca = casadi;

namespace coiks
{

  class QP_BASE : public SolverBase
  {
private:
    pin::Model      mdl_;
    pin::Data       mdl_data_;
    pin::FrameIndex id_Fee_;
    pin::FrameIndex id_Fprercm_;
    pin::FrameIndex id_Fpostrcm_;

    int    max_iter_;
    double max_time_;
    double max_error_;
    double dt_;
    int    n_q_;

    // Task-based coefficients
    double qp_K_t1_;
    double qp_Kr_t1_;

    // Priority-based coefficients
    double qp_Kw_p1_;
    double qp_Kd_p1_;

    int verb_level_;

    VectorXd q_ul_;
    VectorXd q_ll_;

    bool aborted_;
    bool success_;

    ca::Function qpsolver1_;

    MatrixXd C_eig_;
    MatrixXd d_eig_;
    // MatrixXd d_ext;

    //* Priority 1
    // Task1
    MatrixXd A_t1_;
    MatrixXd b_t1_;
    MatrixXd C_t1_;
    MatrixXd d_t1_;
    // QP variables
    MatrixXd A_p1_;
    MatrixXd b_p1_;
    MatrixXd Abar_p1_;
    MatrixXd bbar_p1_;
    MatrixXd Cbar_p1_;
    MatrixXd Q_p1_;
    MatrixXd p_p1_;

    MatrixXd param1;
    ca::DM   par1;

    ca::SXDict qp_1;

    std::string error_type_;

    std::string pinv_method_;
    std::string limiting_method_;
    std::string step_size_method_;
    std::string seed_method_;

    bool qp_warm_start_;

    std::vector<double> cost_coeff_;

    std::default_random_engine       rng_;          // Random number generator
    std::normal_distribution<double> distribution_; // Normal distribution

public:
    QP_BASE(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
            SolverOptions _solver_opts, const double _max_time,
            const double _max_error, const int _max_iter, const double _dt);
    ~QP_BASE();

    int IkSolve(const VectorXd q_init, const pin::SE3 &X_d, VectorXd &q_sol);
    int computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                           Vector6d &err_ee);
    int computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                           const std::string err_type, Vector6d &err_ee);

    ca::DM   eig_to_casDM(const VectorXd &eig);
    ca::DM   eigmat_to_casDM(const MatrixXd &eig);
    MatrixXd casDM_to_eig(const casadi::DM &cas);

    MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon,
                           std::string method);
    int      computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                                   double &err_pos_ee);
    int      computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
                                   double &err_ori_ee);

    void generate_qp_solver();
    void GetOptions(SolverOptions _solver_opts);

    inline void abort() { aborted_ = true; }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; }
    inline void set_max_error(double _max_error) { max_error_ = _max_error; }
  };

  void QP_BASE::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;

    // Solver parameters
    error_type_ = so.error_type_;
    pinv_method_ = so.pinv_method_;
    limiting_method_ = so.limiting_method_;
    step_size_method_ = so.step_size_method_;
    seed_method_ = so.seed_method_;
    qp_K_t1_ = so.qp_K_t1_;
    qp_Kr_t1_ = so.qp_Kr_t1_;
    qp_Kw_p1_ = so.qp_Kw_p1_;
    qp_Kd_p1_ = so.qp_Kd_p1_;
    qp_warm_start_ = so.qp_warm_start_;
    verb_level_ = so.verb_level_;

    std::cout << "\n------\nQP Options summary:" << std::endl;
    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Pinv method: " << pinv_method_ << std::endl;
    std::cout << "Limiting method: " << limiting_method_ << std::endl;
    std::cout << "Step size method: " << step_size_method_ << std::endl;
    std::cout << "Seed method: " << seed_method_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;
    std::cout << "Task Coeff. (K) T1:" << qp_K_t1_ << std::endl;
    std::cout << "Residual Coeff. (Kr) T1:" << qp_Kr_t1_ << std::endl;
    std::cout << "Damping Coeff. (Kd) P1:" << qp_Kd_p1_ << std::endl;
    std::cout << "Slack Regularization Coeff. (Kw) P1:" << qp_Kw_p1_
              << std::endl;
    std::cout << "QP warm start : " << qp_warm_start_ << std::endl;
  }

  QP_BASE::QP_BASE(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
                   SolverOptions _solver_opts, const double _max_time,
                   const double _max_error, const int _max_iter,
                   const double _dt)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt), distribution_(_dt, 0.5)
  {
    std::cout << "----------\nInitializing IK solver COIKS-QP" << std::endl;
    GetOptions(_solver_opts);

    //* Seed the random number generator
    rng_.seed(0);

    mdl_data_ = pin::Data(mdl_);

    q_ul_ = mdl_.upperPositionLimit;
    q_ll_ = mdl_.lowerPositionLimit;

    std::cout << "Generating QP solver " << std::endl;
    generate_qp_solver();

    // Initializing parameters for QP-Task1
    C_eig_.resize(2 * n_q_, n_q_);
    C_eig_.setZero();
    C_eig_ << MatrixXd::Identity(n_q_, n_q_), -MatrixXd::Identity(n_q_, n_q_);

    d_eig_.resize(2 * n_q_, 1);
    d_eig_.setZero();

    //* Task 2 : End-effector
    if (error_type_ == "only_p")
    {
      A_t1_.resize(3, n_q_);
      b_t1_.resize(3, 1);
    }
    else
    {
      A_t1_.resize(6, n_q_);
      b_t1_.resize(6, 1);
    }

    A_t1_.setZero();
    b_t1_.setZero();

    C_t1_ = C_eig_;
    d_t1_ = d_eig_;

    //* QP Priority 1
    if (error_type_ == "only_p")
    {

      A_p1_.resize(3 + n_q_, n_q_);
      b_p1_.resize(3 + n_q_, 1);
      Abar_p1_.resize(3 + 3 * n_q_, 3 * n_q_);
      bbar_p1_.resize(3 + 3 * n_q_, 1);
    }
    else
    {
      A_p1_.resize(6 + n_q_, n_q_);
      b_p1_.resize(6 + n_q_, 1);
      Abar_p1_.resize(6 + 3 * n_q_, 3 * n_q_);
      bbar_p1_.resize(6 + 3 * n_q_, 1);
    }

    A_p1_.setZero();
    b_p1_.setZero();
    Abar_p1_.setZero();
    bbar_p1_.setZero();

    Cbar_p1_.resize(2 * n_q_, 3 * n_q_);
    Cbar_p1_.setZero();
    Q_p1_.resize(3 * n_q_, 3 * n_q_);
    Q_p1_.setZero();
    p_p1_.resize(3 * n_q_, 1);
    p_p1_.setZero();

    param1.resize(3 * n_q_, 5 * n_q_ + 2);
    par1.resize(3 * n_q_, 5 * n_q_ + 2);

    // Getting Joints
    std::cout << "Joints lower limits: " << mdl_.lowerPositionLimit.transpose()
              << std::endl;
    std::cout << "Joints upper limits: " << mdl_.upperPositionLimit.transpose()
              << std::endl;
    std::cout << "# of Joints considered for IK solver: " << n_q_ << std::endl;
  }

  QP_BASE::~QP_BASE() {}

  int QP_BASE::computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                                  Vector6d &err_ee)
  {
    //* Computation Errors
    if (error_type_ == "log6")
    {
      //? Using log6
      pin::SE3 B_Xerr(X_d.act(B_X_EE.inverse()));
      err_ee = pin::log6(B_Xerr).toVector();
    }
    else if (error_type_ == "log3")
    {
      //? Using log3
      // Compute translation error
      err_ee.head<3>() = X_d.translation() - B_X_EE.translation();

      // Compute rotation error
      err_ee.tail<3>() =
          pin::log3(X_d.rotation() * B_X_EE.rotation().transpose());
    }
    else
    {
      //? Using only p error
      // Compute translation error
      err_ee.head<3>() = X_d.translation() - B_X_EE.translation();

      // Set rotation error to zero
      err_ee.tail<3>().setZero();
    }
    return 0;
  }

  int QP_BASE::computeEETaskError(const pin::SE3 &B_X_EE, const pin::SE3 &X_d,
                                  const std::string err_type, Vector6d &err_ee)
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
      // Compute translation error
      err_ee.head<3>() = X_d.translation() - B_X_EE.translation();

      // Compute rotation error
      err_ee.tail<3>() =
          pin::log3(X_d.rotation() * B_X_EE.rotation().transpose());
    }
    else
    {
      //? Using only p error
      // Compute translation error
      err_ee.head<3>() = X_d.translation() - B_X_EE.translation();

      // Set rotation error to zero
      err_ee.tail<3>().setZero();
    }

    return 0;
  }

  int QP_BASE::IkSolve(const VectorXd q_init, const pin::SE3 &X_d,
                       VectorXd &q_sol)
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time =
        std::chrono::high_resolution_clock::now();

    q_sol = q_init;

    //* Initialization of the configuration seed (neutral, current, middle)
    VectorXd q_it;
    if (seed_method_ == "neutral")
    {
      q_it = pin::neutral(mdl_);
    }
    else if (seed_method_ == "middle")
    {
      q_it = (mdl_.upperPositionLimit + mdl_.lowerPositionLimit) / 2.0;
    }
    else
    {
      q_it = q_init;
    }

    if (verb_level_ >= 1)
      std::cout << "\n[QP] Solving with optimizer QP IK and Initial guess:"
                << q_init.transpose() << std::endl;

    Vector6d err_ee = Vector6d::Zero();
    Vector6d err_log3_ee;

    pin::Data::Matrix6x B_Jb_Fee = pin::Data::Matrix6x::Zero(6, n_q_);

    MatrixXd qd_opt;
    ca::DM   qd_opt_1;

    success_ = false;
    aborted_ = false;

    pin::SE3 B_X_EE;

    pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
    B_X_EE = mdl_data_.oMf[id_Fee_];

    //* Computing EE error
    computeEETaskError(B_X_EE, X_d, err_ee);

    C_t1_ = C_eig_;

    const double sqrt_qp_K_t2 = std::sqrt(qp_K_t1_);
    const double sqrt_qp_Kd_p2 = std::sqrt(qp_Kd_p1_);
    const double sqrt_qp_Kw_p2 = std::sqrt(qp_Kw_p1_);

    for (int it = 0; it < max_iter_; it++)
    {
      auto   current_time = std::chrono::high_resolution_clock::now();
      double time_elapsed_s =
          std::chrono::duration<double>(current_time - start_time).count();

      if (time_elapsed_s > max_time_ || aborted_)
      {
        if (verb_level_ >= 1)
          std::cout
              << "[QP]  Aborted by other solvers or Maximum time exceeded: "
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

      // if (verb_level_ >= 1)
      //   std::cout << "[QP] Initial EE error: " << err_ee.norm() << " -> "
      //             << err_ee.transpose() << std::endl;

      d_eig_.block(0, 0, n_q_, 1) = q_ul_ - q_it;
      d_eig_.block(n_q_, 0, n_q_, 1) = -(q_ll_ - q_it);
      d_t1_ = d_eig_;

      if (verb_level_ >= 1)
        std::cout << "[QP] qd limits (d): " << d_eig_.transpose() << std::endl;

      if (error_type_ == "only_p")
      {
        A_t1_ = B_Jb_Fee.topRows(3);
        b_t1_ = qp_Kr_t1_ * err_ee.head(3);
      }
      else
      {
        A_t1_ = B_Jb_Fee;
        b_t1_ = qp_Kr_t1_ * err_ee;
      }

      if (error_type_ == "only_p")
      {
        A_p1_.block(0, 0, 3, n_q_) = sqrt(qp_K_t1_) * A_t1_;
        A_p1_.block(3, 0, n_q_, n_q_) =
            sqrt(qp_Kd_p1_) * MatrixXd::Identity(n_q_, n_q_);

        b_p1_.block(0, 0, 3, 1) = sqrt(qp_K_t1_) * b_t1_;
        b_p1_.block(3, 0, n_q_, 1) = MatrixXd::Zero(n_q_, 1);

        Abar_p1_.setZero();
        Abar_p1_.block(0, 0, 3 + n_q_, n_q_) = A_p1_;
        Abar_p1_.block(3 + n_q_, n_q_, 2 * n_q_, 2 * n_q_) =
            sqrt(qp_Kw_p1_) * MatrixXd::Identity(2 * n_q_, 2 * n_q_);

        bbar_p1_.block(0, 0, 3 + n_q_, 1) = b_p1_;
        bbar_p1_.block(3 + n_q_, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
      }
      else
      {
        A_p1_.topLeftCorner(6, n_q_).noalias() = sqrt_qp_K_t2 * A_t1_;
        A_p1_.bottomLeftCorner(n_q_, n_q_)
            .diagonal()
            .setConstant(sqrt_qp_Kd_p2);

        b_p1_.topRows(6).noalias() = sqrt_qp_K_t2 * b_t1_;
        b_p1_.bottomRows(n_q_).setZero();

        Abar_p1_.setZero();
        Abar_p1_.topLeftCorner(6 + n_q_, n_q_) = A_p1_;
        Abar_p1_.bottomRightCorner(2 * n_q_, 2 * n_q_)
            .diagonal()
            .setConstant(sqrt_qp_Kw_p2);

        bbar_p1_.block(0, 0, 6 + n_q_, 1) = b_p1_;
        bbar_p1_.block(6 + n_q_, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
      }

      Cbar_p1_.topLeftCorner(2 * n_q_, n_q_) = C_t1_;
      Cbar_p1_.topRightCorner(2 * n_q_, 2 * n_q_).diagonal().setConstant(-1);

      // Use Eigen::NoAlias() for matrix multiplications
      Q_p1_.noalias() = Abar_p1_.transpose() * Abar_p1_;
      p_p1_.noalias() = -Abar_p1_.transpose() * bbar_p1_;

      param1.topLeftCorner(3 * n_q_, 3 * n_q_) = Q_p1_;
      param1.block(0, 3 * n_q_, 3 * n_q_, 1) = p_p1_;
      param1.block(0, 3 * n_q_ + 1, 3 * n_q_, 2 * n_q_) = Cbar_p1_.transpose();
      param1.block(0, 5 * n_q_ + 1, 2 * n_q_, 1) = d_t1_;

      par1 = eigmat_to_casDM(param1);

      ca::DMDict arg_qpopt = {{"p", par1}, {"lbg", 0}};

      //* Solve QP
      ca::DMDict res_qpsolver1_ = qpsolver1_(arg_qpopt);

      if (verb_level_ >= 2)
        std::cout << "[QP] QP solution: " << res_qpsolver1_["x"] << std::endl;

      qd_opt_1 = res_qpsolver1_["x"](ca::Slice(0, n_q_));
      MatrixXd qd_opt_hat_1 = casDM_to_eig(qd_opt_1);

      if (verb_level_ >= 1)
        std::cout << "[QP] qd optimal (Priority 1): " << qd_opt_1 << std::endl;

      qd_opt = qd_opt_hat_1;

      // Update configuration
      q_it = pin::integrate(mdl_, q_it, VectorXd(qd_opt * dt_));

      if (verb_level_ >= 1)
        std::cout << "[QP] q_it: [" << it << "]: " << q_it.transpose()
                  << std::endl;

      // Ensure solutions are withnin one full rotation [-2pi, 2pi]
      q_it = q_it.unaryExpr([](double x) { return std::fmod(x, 2 * M_PI); });

      if (limiting_method_ == "clamping")
      {
        //? Clamps solution to joint limits
        q_it = q_it.cwiseMax(q_ll_).cwiseMin(q_ul_);
      }
      else if (limiting_method_ == "random")
      {
        //? Generate a random configuration if the solution is outside the joint
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
          if (verb_level_ >= 1)
            std::cout << "[QP] Random configuration generated because of "
                         "joint limits"
                      << std::endl;
          q_it = pin::randomConfiguration(mdl_);
        }
      }

      // Computing updated errors
      // FK
      pin::framesForwardKinematics(mdl_, mdl_data_, q_it);

      // Computing EE error
      B_X_EE = mdl_data_.oMf[id_Fee_];
      computeEETaskError(B_X_EE, X_d, err_ee);

      if (verb_level_ >= 1)
        std::cout << "[QP] EE error: " << err_ee.norm() << " -> "
                  << err_ee.transpose() << std::endl;

      // Check for convergence
      if (err_ee.norm() < max_error_)
      {
        if (verb_level_ >= 1)
        {
          std::cout << "[QP] Iteration: " << it
                    << " EE error: " << err_ee.norm() << std::endl;
          std::cout << "[QP] iteration: " << it << " - Solution found "
                    << q_it.transpose() << std::endl;
        }
        success_ = true;
        break;
      }
    }

    auto   current_time = std::chrono::high_resolution_clock::now();
    double time_elapsed_s =
        std::chrono::duration<double>(current_time - start_time).count();

    if (success_)
    {
      if (verb_level_ >= 1)
        std::cout << "QP: Convergence achieved!" << std::endl;

      q_sol = q_it;
      return 0;
    }

    if (verb_level_ >= 1)
      std::cout << "\nQP: Warning: the iterative algorithm has not reached "
                   "convergence to the desired precision after "
                << max_iter_ << " iterations." << std::endl;
    return 1;
  }

  void QP_BASE::generate_qp_solver()
  {
    // Optimizer options
    ca::Dict opts;
    opts["osqp.max_iter"] = 1000;
    opts["error_on_fail"] = false; // true
    if (verb_level_ >= 1)
    {
      opts["verbose"] = true;      // false
      opts["osqp.verbose"] = true; //
    }
    else
    {
      opts["verbose"] = false;      // false
      opts["osqp.verbose"] = false; //
    }
    opts["warm_start_primal"] = qp_warm_start_; // 0

    // Optimization Variable
    ca::SX x1 = ca::SX::sym("x1", 3 * n_q_, 1);

    // Fixed parameters
    ca::SX par = ca::SX::sym("par", 3 * n_q_, 5 * n_q_ + 2);

    ca::SX Q1 = par(ca::Slice(), ca::Slice(0, 3 * n_q_));
    ca::SX p1 = par(ca::Slice(), 3 * n_q_);
    ca::SX C1 = (par(ca::Slice(), ca::Slice(3 * n_q_ + 1, 5 * n_q_ + 1))).T();
    ca::SX d1 = par(ca::Slice(0, 2 * n_q_), par.size2() - 1);

    ca::SX qdot1 = x1(ca::Slice(0, n_q_));
    ca::SX w1 = x1(ca::Slice(n_q_, 3 * n_q_));

    // Nonlinear problem declaration
    ca::SXDict qp1;

    // Nonlinear problem arguments definition
    qp1 = {
        {"x", x1},
        {"f", 0.5 * ca::SX::mtimes(x1.T(), ca::SX::mtimes(Q1, x1)) +
                  ca::SX::mtimes(p1.T(), x1)},
        {"g", ca::SX(d1) - ca::SX::mtimes(C1, x1)},
        {"p", par},
    };

    qpsolver1_ = qpsol("qpsol1", "osqp", qp1, opts);
  }

  //* Casadi-Eigen conversion functions
  ca::DM QP_BASE::eig_to_casDM(const VectorXd &eig)
  {
    // Create a DM with the same size as the Eigen vector
    ca::DM dm = ca::DM::zeros(eig.size());

    // Use std::copy for faster data transfer
    std::copy(eig.data(), eig.data() + eig.size(), dm.ptr());

    return dm;
  }

  ca::DM QP_BASE::eigmat_to_casDM(const MatrixXd &eig)
  {
    // Create a DM with the same dimensions as the Eigen matrix
    ca::DM dm = ca::DM::zeros(eig.rows(), eig.cols());

    // Use std::copy for faster data transfer
    std::copy(eig.data(), eig.data() + eig.size(), dm.ptr());

    return dm;
  }

  MatrixXd QP_BASE::casDM_to_eig(const casadi::DM &dm)
  {
    // Use Eigen's Map to directly map the CasADi data to an Eigen matrix
    Eigen::Map<const MatrixXd> eig(dm.ptr(), dm.size1(), dm.size2());
    return eig;
  }

  MatrixXd QP_BASE::pseudoInverse(const Eigen::MatrixXd &a, double epsilon,
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
    else if (method == "cholesky")
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
      std::cout << "Invalid method for pseudoinverse. Using SVD." << std::endl;
      return pseudoInverse(a, epsilon, "svd");
    }
  }

  int QP_BASE::computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                                     double &err_pos_ee)
  {
    pin::framesForwardKinematics(mdl_, mdl_data_, q);

    pin::SE3 B_H_Fee_ = mdl_data_.oMf[id_Fee_];

    //* Computation Errors
    //? Using log3
    err_pos_ee = (H_d.translation() - B_H_Fee_.translation()).norm();

    return 0;
  }

  int QP_BASE::computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
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
