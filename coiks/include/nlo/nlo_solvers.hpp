#ifndef NLO_SOLVERS_HPP
#define NLO_SOLVERS_HPP

#include <coiks/solver_base.hpp>
#include <coiks/solver_options.hpp>
// Casadi
#include <casadi/casadi.hpp>
// C++
#include <map>
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

  class NLO_BASE : public SolverBase
  {
private:
    pin::Model      mdl_;
    pin::Data       mdl_data_;
    pin::FrameIndex id_Fee_;

    int    max_iter_;
    double max_time_;
    double max_error_;
    double dt_;

    const int n_q_;
    VectorXd  q_max_;
    VectorXd  q_min_;

    int         verb_level_;
    std::string time_stats_;

    bool aborted_;
    bool success_;

    bool        nlo_concurrent_;
    int         nlo_concurrent_iterations_;
    std::string nlo_linear_solver_;
    std::string nlo_warm_start_;
    std::string error_type_;

    // Penalty gains
    double mu0_;
    double mu1_;
    double mu2_;
    double mu3_;
    double mu4_;

    ca::Function FK_;
    ca::Function ca_perr2_;
    ca::Function ca_elog3_ee_;
    ca::Function ca_log3_;
    ca::Function ca_log6_;
    ca::Function solver_;

    std::vector<double> cost_coeff_;

public:
    NLO_BASE(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
             SolverOptions _solver_opts, const double _max_time,
             const double _max_error, const int _max_iter, const double _dt);
    ~NLO_BASE();

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_d, VectorXd &q_sol);

    ca::DM   eig_to_casDM(const VectorXd &eig);
    ca::DM   eigmat_to_casDM(const MatrixXd &eig);
    MatrixXd casDM_to_eig(const casadi::DM &cas);

    // void generate_ca_RCM_error();
    void generate_ca_EE_error();
    void generate_ca_log3_EE_error();
    void generate_ca_log3();
    void generate_ca_log6();
    void generate_nlsolver();
    void GetOptions(SolverOptions _solver_opts);

    int computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                              double &err_pos_ee);
    int computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
                              double &err_ori_ee);

    inline void abort()
    {
      aborted_ = true;
      if (verb_level_ >= 1)
        std::cout << "Setting NLO abort" << std::endl;
    }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; }
    inline void set_max_error(double _max_error) { max_error_ = _max_error; }
  };

  void NLO_BASE::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;
    cost_coeff_.clear();

    // Solver parameters
    nlo_linear_solver_ = so.nlo_linear_solver_;
    nlo_concurrent_ = so.nlo_concurrent_;
    nlo_concurrent_iterations_ = so.nlo_concurrent_iterations_;
    nlo_warm_start_ = so.nlo_warm_start_;
    error_type_ = so.nlo_error_type_;
    cost_coeff_ = so.cost_coeff_;
    time_stats_ = so.time_stats_;
    verb_level_ = so.verb_level_;

    std::cout << "\n------\n NLO Options summary:" << std::endl;
    std::cout << "NLO concurrent mode enabled: " << nlo_concurrent_
              << std::endl;
    std::cout << "NLO concurrent iterations: " << nlo_concurrent_iterations_
              << std::endl;
    std::cout << "NLO warm start: " << nlo_warm_start_ << std::endl;
    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Linear solver: " << nlo_linear_solver_ << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;
    std::cout << "Time Statstics: " << time_stats_ << std::endl;
    std::cout << "Coeff[0] Pos(log3)/Pose(log6) error: " << cost_coeff_[0]
              << std::endl;
    std::cout << "Coeff[1] Ori(log3): " << cost_coeff_[1] << std::endl;
    std::cout << "Coeff[2] RCM error: " << cost_coeff_[2] << std::endl;
    std::cout << "Coeff[3] Qdelta: " << cost_coeff_[3] << std::endl;
    std::cout << "Coeff[4] Manipulability: " << cost_coeff_[4] << std::endl;
  }

  NLO_BASE::NLO_BASE(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
                     SolverOptions _solver_opts, const double _max_time,
                     const double _max_error, const int _max_iter,
                     const double _dt)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt)
  {
    std::cout << "----------\nInitializing IK solver COIKS-NLO" << std::endl;

    GetOptions(_solver_opts);

    mu0_ = cost_coeff_[0]; // 10
    mu1_ = cost_coeff_[1]; // 0.005
    mu2_ = cost_coeff_[2]; // 0.001
    mu3_ = cost_coeff_[3]; // 100
    mu4_ = cost_coeff_[4]; // 100

    mdl_data_ = pin::Data(mdl_);

    q_max_ = mdl_.upperPositionLimit;
    q_min_ = mdl_.lowerPositionLimit;

    //* Exporting Kinematics Casadi Functions
    // Cast the model into casadi::SX
    pin::ModelTpl<ca::SX> model = mdl_.cast<ca::SX>();

    // Create Data model as casadi::SX
    pinocchio::DataTpl<ca::SX> data(model);

    // Create casadi::SX joint variable
    ca::SX ca_q = ca::SX::sym("ca_q", n_q_, 1);

    // Create associated Eigen matrix
    Eigen::Matrix<ca::SX, Eigen::Dynamic, 1> _q;
    _q.resize(n_q_, 1);

    // Copy casadi::SX into Eigen::Matrix
    pin::casadi::copy(ca_q, _q);

    //* Generate symbolic FK

    std::cout << "Generate Casadi FK function" << std::endl;
    pin::framesForwardKinematics(model, data, _q);

    // Extract Eigen::Matrix results
    Eigen::Matrix<ca::SX, 3, 1> eig_fk_pos = data.oMf.at(id_Fee_).translation();
    Eigen::Matrix<ca::SX, 3, 3> eig_fk_rot = data.oMf.at(id_Fee_).rotation();

    // Create associated casadi::SX variables
    ca::SX ca_fk_tr =
        ca::SX(ca::Sparsity::dense(eig_fk_pos.rows(), eig_fk_pos.cols()));
    ca::SX ca_fk_rot =
        ca::SX(ca::Sparsity::dense(eig_fk_rot.rows(), eig_fk_rot.cols()));

    // Copy Eigen::Matrix into casadi::SX
    pinocchio::casadi::copy(eig_fk_pos, ca_fk_tr);
    pinocchio::casadi::copy(eig_fk_rot, ca_fk_rot);

    // Generate function
    FK_ = ca::Function("forward_kinematics", {ca_q}, {ca_fk_tr, ca_fk_rot},
                       {"q"}, {"ee_pos", "ee_rot"});

    generate_ca_log3();
    generate_ca_log6();
    generate_ca_EE_error();
    generate_ca_log3_EE_error();
    generate_nlsolver();

    // Getting Joints
    std::cout << "Joints considered for IK solver: " << n_q_ << std::endl;
    std::cout << "Joints lower limits: " << mdl_.lowerPositionLimit.transpose()
              << std::endl;
    std::cout << "Joints upper limits: " << mdl_.upperPositionLimit.transpose()
              << std::endl;

    std::cout << "COIKS-NLO initialized\n---------------------------------"
              << std::endl;
  }

  NLO_BASE::~NLO_BASE() {}

  int NLO_BASE::IkSolve(const VectorXd q_init, const pin::SE3 &x_d,
                        VectorXd &q_sol)
  {

    std::chrono::time_point<std::chrono::high_resolution_clock>
                              start_solve_time = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds diff;

    double time_left = 0.0;
    success_ = false;
    double err_ee = 0.0;
    double err_log3_p = 0.0;
    double err_log3_R = 0.0;

    ca::DM qd_opt;
    ca::DM f_opt;

    q_sol = q_init;
    if (verb_level_ >= 1)
      std::cout << "\nSolving NLO IK with optimizer with q_init:"
                << q_init.transpose() << std::endl;

    ca::SX ca_pd = eig_to_casDM(x_d.translation());
    ca::SX ca_Rd = eigmat_to_casDM(x_d.rotation());
    ca::SX ca_qin = eigmat_to_casDM(q_init);

    ca::DMDict arg_nlopt;

    ca::SX ca_qd_min = eig_to_casDM(q_min_ - q_init);
    ca::SX ca_qd_max = eig_to_casDM(q_max_ - q_init);

    auto   Xact = FK_(ca_qin);
    ca::SX p_act = Xact[0];
    ca::SX R_act = Xact[1];

    ca::DM par = ca::DM::zeros(2 * n_q_ + 22);
    ca::DM mu({mu0_, mu1_, mu2_, mu3_, mu4_});

    double dT = 0.01;
    par(ca::Slice(0, n_q_)) = ca_qin;
    par(ca::Slice(n_q_, n_q_ + 3)) = ca_pd;
    par(ca::Slice(n_q_ + 3, n_q_ + 12)) = ca::DM::reshape(ca_Rd, 9, 1);
    par(ca::Slice(n_q_ + 15, n_q_ + 20)) = mu;
    par(n_q_ + 20) = dT;

    arg_nlopt = {{"x0", ca::SX::zeros((n_q_, 1))},
                 {"p", par},
                 {"lbx", ca_qd_min},
                 {"ubx", ca_qd_max}};
    // }

    diff = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_solve_time);
    time_left = max_time_ - diff.count() / 1000000.0;

    for (int it = 0;; it++)
    {
      if (verb_level_ >= 1)
        std::cout << "Starting it " << it << " with warm start "
                  << arg_nlopt["x0"] << std::endl;

      //* Solve NLO IK
      ca::DMDict res_nlopt = solver_(arg_nlopt);

      qd_opt = res_nlopt["x"];
      f_opt = res_nlopt["f"];

      double opt_cost = static_cast<double>(f_opt);

      if (verb_level_ >= 1)
        std::cout << "\nIt solution: " << qd_opt << std::endl;

      // Verify costs
      std::vector<ca::DM> xcost =
          ca_perr2_(std::vector<ca::DM>{ca::DM(ca_qin) + qd_opt, ca_pd, ca_Rd});
      ca::DM c0 = ca::DM::sqrt(xcost[0]); // P cost
      ca::DM c1 = ca::DM::sqrt(xcost[1]); // R cost
      double p_err = static_cast<double>(c0);
      double r_err = static_cast<double>(c1);

      if (error_type_ == "log3")
        err_ee = sqrt(p_err * p_err + r_err * r_err);
      else if (error_type_ == "log6")
        err_ee = p_err;

      if (verb_level_ >= 1)
      {
        ca::DM c2 = ca::DM::mtimes(qd_opt.T(), qd_opt);
        std::cout << "\nAfer optimization:" << err_ee << std::endl;

        std::cout << "\tPose error : " << err_ee << std::endl;
        std::cout << "\tPosition error : " << p_err << std::endl;
        std::cout << "\tOrientation error : " << r_err << std::endl;
        std::cout << "\tCost0 : " << c0
                  << " Squared and Scaled: " << ca::DM(mu0_) * c0 * c0
                  << std::endl;
        std::cout << "\tCost1 : " << c1
                  << " Squared and Scaled: " << ca::DM(mu1_) * c1 * c1
                  << std::endl;
        std::cout << "\tCost2 : " << c2 << " Scaled: " << ca::DM(mu2_) * c2
                  << std::endl;
      }

      if (p_err < max_error_ && r_err < max_error_)
      {
        success_ = true;
        break;
      }
      else
      {
        if (verb_level_ >= 1)
        {
          std::cout << "NLO: Warning: the iterative algorithm has not reached "
                       "convergence to the desired precision."
                    << std::endl;
          success_ = false;
        }
        break;
      }

      if (time_left < 0)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted. Maximum time exceeded: " << time_left
                    << std::endl;
        break;
      }

      if (aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted by other IK solver " << std::endl;
        break;
      }

      if (it >= max_iter_)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted. Maximum number of iteration reached. "
                    << std::endl;
        break;
      }

      arg_nlopt = {
          {"x0", qd_opt}, {"p", par}, {"lbx", ca_qd_min}, {"ubx", ca_qd_max}};

      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      time_left = max_time_ - diff.count() / 1000000.0;
    }

    if (success_)
    {
      // std::cout << "NLO: Convergence achieved!" << std::endl;
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      q_sol = casDM_to_eig(ca::DM(ca_qin) + qd_opt);
      std::vector<ca::DM> err_log3 = ca_elog3_ee_(
          std::vector<ca::DM>{ca::DM(ca_qin) + qd_opt, ca_pd, ca_Rd});
      ca::DM log3_p = err_log3[0]; // P cost
      ca::DM log3_R = err_log3[1]; // R cost

      double err_log3_p = static_cast<double>(log3_p);
      double err_log3_R = static_cast<double>(log3_R);

      auto   Xact = FK_(ca::DM(ca_qin) + qd_opt);
      ca::DM p_act = Xact[0];
      ca::DM R_act = Xact[1];

      return 0;
    }
    else
    {
      if (verb_level_ >= 1)
        std::cout << "\nNLO: Warning: the iterative algorithm has not reached "
                     "convergence to the desired precision "
                  << std::endl;
      return 1;
    }
  }

  void NLO_BASE::generate_nlsolver()
  {
    // Optimization Variable
    ca::SX q_delta = ca::SX::sym("qdelta", n_q_, 1);

    // Fixed parameters
    ca::SX par = ca::SX::sym("par", 2 * n_q_ + 22, 1);

    ca::SX q_in = par(ca::Slice(0, n_q_));
    ca::SX ca_pd = par(ca::Slice(n_q_, n_q_ + 3));
    ca::SX ca_Rd = ca::SX::reshape(par(ca::Slice(n_q_ + 3, n_q_ + 12)), 3, 3);
    ca::SX mu = par(ca::Slice(n_q_ + 15, n_q_ + 20));
    ca::SX dT = par(n_q_ + 20);

    // Optimizer options
    ca::Dict opts;
    opts["verbose"] = false; // false
    opts["print_time"] = 0;
    opts["ipopt.linear_solver"] = nlo_linear_solver_;      //
    opts["ipopt.print_level"] = verb_level_;               // 0
    opts["ipopt.print_timing_statistics"] = time_stats_;   //"no"
    opts["ipopt.warm_start_init_point"] = nlo_warm_start_; //"no"
    opts["ipopt.max_wall_time"] = max_time_;               //"no"
    if (nlo_concurrent_)
      opts["ipopt.max_iter"] = nlo_concurrent_iterations_;
    else
      opts["ipopt.max_iter"] = max_iter_;
    opts["ipopt.tol"] = max_error_ * max_error_; //"no"

    // Objective Function
    ca::SX obj;

    // Inequality constraints
    ca::SX cineq;

    // Nonlinear problem declaration
    ca::SXDict nlp;

    //? Use non-constrained control
    if (verb_level_ >= 1)
      std::cout << "No constrained motion" << std::endl;

    std::vector<ca::SX> xcost =
        ca_perr2_(std::vector<ca::SX>{q_in + q_delta, ca_pd, ca_Rd});
    ca::SX cost0 = xcost[0];
    ca::SX cost1 = xcost[1];
    ca::SX cost2 = ca::SX::mtimes(q_delta.T(), q_delta);
    obj = mu(0) * cost0 + mu(1) * cost1 + mu(2) * cost2;

    nlp = {{"x", q_delta}, {"p", par}, {"f", obj}};

    solver_ = nlpsol("nlpsol", "ipopt", nlp, opts);
  }

  void NLO_BASE::generate_ca_EE_error()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi EE error function" << std::endl;
    // Inputs
    ca::SX R_des = ca::SX::sym("R_act", 3, 3);
    ca::SX p_des = ca::SX::sym("p_act", 3, 1);
    ca::SX q_it = ca::SX::sym("q_it", n_q_, 1);

    ca::SX p_act = FK_(q_it)[0];
    ca::SX R_act = FK_(q_it)[1];

    ca::SX p_e =
        p_des - ca::SX::mtimes(R_des, ca::SX::mtimes(R_act.T(), p_act));
    ca::SX R_e = ca::SX::mtimes(R_des, R_act.T());

    ca::SX err;

    ca::SX p_error;
    ca::SX R_error;
    ca::SX p_error2;
    ca::SX R_error2;

    if (error_type_ == "log6")
    {
      //? Using log6
      std::vector<ca::SX> err_tmp = ca_log6_(std::vector<ca::SX>{p_e, R_e});
      p_error2 = ca::SX::mtimes(err_tmp[0].T(), err_tmp[0]);
      R_error2 = ca::SX(0.0);
    }
    else if (error_type_ == "log3")
    {
      //? Using log3
      p_error = p_des - p_act;
      p_error2 = ca::SX::mtimes(p_error.T(), p_error);
      R_error = ca_log3_(std::vector<ca::SX>{R_e})[0];
      R_error2 = ca::SX::mtimes(R_error.T(), R_error);
    }
    else
    {
      //? Using only p error
      p_error = p_des - p_act;
      p_error2 = ca::SX::mtimes(p_error.T(), p_error);
      R_error2 = ca::SX(0.0);
    }

    ca_perr2_ =
        ca::Function("p_err2", {q_it, p_des, R_des}, {p_error2, R_error2},
                     {"q_it", "pd", "Rd"}, {"p_err2", "R_err2"});
  }

  void NLO_BASE::generate_ca_log3_EE_error()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi Log3 EE error function" << std::endl;
    // Inputs
    ca::SX R_des = ca::SX::sym("R_act", 3, 3);
    ca::SX p_des = ca::SX::sym("p_act", 3, 1);
    ca::SX q_init = ca::SX::sym("q_init", n_q_, 1);

    ca::SX p_act = FK_(q_init)[0];
    ca::SX R_act = FK_(q_init)[1];

    ca::SX R_e = ca::SX::mtimes(R_des, R_act.T());

    ca::SX p_err_vec;
    ca::SX R_err_vec;
    ca::SX p_error;
    ca::SX R_error;

    p_err_vec = p_des - p_act;
    p_error = ca::SX::sqrt(ca::SX::mtimes(p_err_vec.T(), p_err_vec));

    R_err_vec = ca_log3_(std::vector<ca::SX>{R_e})[0];
    R_error = ca::SX::sqrt(ca::SX::mtimes(R_err_vec.T(), R_err_vec));

    ca_elog3_ee_ =
        ca::Function("err_log3", {q_init, p_des, R_des}, {p_error, R_error},
                     {"q_init", "pd", "Rd"}, {"p_error", "R_error"});
  }

  // log3
  void NLO_BASE::generate_ca_log3()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi log3 function" << std::endl;
    ca::SX tolerance = 1e-8;

    ca::SX R = ca::SX::sym("R", 3, 3);
    ca::SX omega = ca::SX::sym("omega", 3, 1);
    ca::SX val = (ca::SX::trace(R) - ca::SX(1)) / ca::SX(2);
    val = ca::SX::if_else(val > ca::SX(1), ca::SX(1),
                          ca::SX::if_else(val < ca::SX(-1), ca::SX(-1), val));
    ca::SX theta = ca::SX::acos(val);
    ca::SX stheta = ca::SX::sin(theta);
    ca::SX tr = ca::SX::if_else(theta < tolerance, ca::SX::zeros((3, 3)),
                                (R - R.T()) * theta / (ca::SX(2) * stheta));
    omega = ca::SX::inv_skew(tr);
    ca_log3_ =
        ca::Function("ca_log3", {R}, {omega, theta}, {"R"}, {"w", "theta"});
  }

  // log6
  void NLO_BASE::generate_ca_log6()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi log6 function" << std::endl;
    ca::SX tolerance = 1e-8;
    ca::SX tolerance2 = 1e-16;

    ca::SX tau = ca::SX::sym("tau", 6, 1);
    ca::SX R = ca::SX::sym("R", 3, 3);
    ca::SX p = ca::SX::sym("p", 3, 1);

    std::vector<ca::SX> log_res = ca_log3_(R);
    ca::SX              omega = log_res[0];
    ca::SX              theta = log_res[1];

    ca::SX stheta = ca::SX::sin(theta);
    ca::SX ctheta = ca::SX::cos(theta);

    ca::SX A_inv = ca::SX::if_else(
        ca::SX::mtimes(p.T(), p) < tolerance2, ca::SX::zeros((3, 3)),
        ca::SX::if_else(
            theta < tolerance, ca::SX::eye(3),
            ca::SX::eye(3) - ca::SX::skew(omega) / ca::SX(2) +
                (ca::SX(2) * stheta - theta * (ca::SX(1) + ctheta)) *
                    (ca::SX::mtimes(ca::SX::skew(omega), ca::SX::skew(omega))) /
                    (ca::SX(2) * (ca::SX::pow(theta, 2)) * stheta)));

    ca::SX v = ca::SX::mtimes(A_inv, p);

    tau(ca::Slice(0, 3)) = v;
    tau(ca::Slice(3, 6)) = omega;

    ca_log6_ = ca::Function("ca_log6", {p, R}, {tau}, {"p", "R"}, {"tau"});
  }

  //* Casadi-Eigen conversion functions

  ca::DM NLO_BASE::eig_to_casDM(const VectorXd &eig)
  {
    auto dm = casadi::DM(casadi::Sparsity::dense(eig.size()));
    for (int i = 0; i < eig.size(); i++)
    {
      dm(i) = eig(i);
    }
    return dm;
  }

  ca::DM NLO_BASE::eigmat_to_casDM(const MatrixXd &eig)
  {
    casadi::DM dm = casadi::DM(casadi::Sparsity::dense(eig.rows(), eig.cols()));
    std::copy(eig.data(), eig.data() + eig.size(), dm.ptr());
    return dm;
  }

  MatrixXd NLO_BASE::casDM_to_eig(const casadi::DM &cas)
  {
    auto     vector_x = static_cast<std::vector<double>>(cas);
    MatrixXd eig = MatrixXd::Zero(cas.size1(), cas.size2());

    for (int i = 0; i < cas.size1(); i++)
    {
      for (int j = 0; j < cas.size2(); j++)
      {
        eig(i, j) = vector_x[i + j * cas.size2()];
      }
    }
    return eig;
  }

  int NLO_BASE::computeEETaskPosError(const pin::SE3 &H_d, const VectorXd &q,
                                      double &err_pos_ee)
  {
    pin::framesForwardKinematics(mdl_, mdl_data_, q);

    pin::SE3 B_H_Fee_ = mdl_data_.oMf[id_Fee_];

    //* Computation Errors
    //? Using log3
    err_pos_ee = (H_d.translation() - B_H_Fee_.translation()).norm();

    return 0;
  }

  int NLO_BASE::computeEETaskOriError(const pin::SE3 &H_d, const VectorXd &q,
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
