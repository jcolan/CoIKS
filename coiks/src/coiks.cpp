#include <coiks/coiks.hpp>

namespace coiks
{
  // Task Types
  enum
  {
    CONSTRAINT = 0,
    OPERATIONAL,
    POSTURE
  };

  // Constraints Types
  enum
  {
    JOINT_LIMITS = 0,
    JOINT_SPEED,
    CART_SPEED
  };

  // Operational Task Subtypes
  enum
  {
    CART_VEL_TRANSLATION = 0,
    CART_VEL_ORIENTATION,
    CART_VEL_POSE,
  };

  typedef struct
  {
    std::string name;
    int         priority;
    std::string type;
    int         type_id;
    std::string subtype;
    int         subtype_id;

  } VarTask;

  double fRand(double min, double max)
  {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
  }

  COIKS::COIKS(const std::string &_urdf_file, const std::string &_base_link,
               const std::string &_ee_link, const std::string &_ik_solver,
               SolverOptions _solver_opts, double _max_time, double _max_error,
               int _max_iter, double _dt)
      : initialized_(false), max_error_(_max_error), max_time_(_max_time),
        max_iter_(_max_iter), urdf_file_(_urdf_file), base_link_(_base_link),
        ee_link_(_ee_link), ik_solver_name_(_ik_solver), dt_(_dt),
        solver_opts_(_solver_opts)
  {
    initialize();
  }

  COIKS::~COIKS()
  {
    if (solver1_.joinable())
      solver1_.join();
    if (solver2_.joinable())
      solver2_.join();
    if (solver3_.joinable())
      solver3_.join();

    std::cout << "COIKS closed" << std::endl;
  }

  int COIKS::IkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                     VectorXd &q_sol, IKSolutionSet::Solution &bestSolution)
  {
    int res = 1;
    sol_id_ = 0;

    solutionSet->clearSolutions();

    if (solver_opts_.verb_level_ >= 2)
      std::cout << "[COIKS] Solving IK with " << ik_solver_name_ << std::endl;
    start_iksolve_time_ = std::chrono::high_resolution_clock::now();
    auto start_cb_time = std::chrono::high_resolution_clock::now();

    if (ik_solver_name_ == "coiks_invj")
      res = invj_solver_->IkSolve(q_init, x_des, q_sol);
    else if (ik_solver_name_ == "coiks_nlo")
      res = nlo_solver_->IkSolve(q_init, x_des, q_sol);
    else if (ik_solver_name_ == "coiks_qp")
      res = qp_solver_->IkSolve(q_init, x_des, q_sol);
    else
      res = concurrentIkSolve(q_init, x_des, q_sol);

    auto stop_cb_time = std::chrono::high_resolution_clock::now();
    auto duration_cb = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_cb_time - start_cb_time);

    bool                    hasSolution;
    IKSolutionSet::Solution solution;

    // After all IK solutions are computed, get the best solution
    if (solver_opts_.solve_mode_ == "distance")
    {
      std::tie(hasSolution, solution) =
          solutionSet->getBestSolution("distance", OptimizationGoal::Minimize);
    }
    else if (solver_opts_.solve_mode_ == "manipulability")
    {
      std::tie(hasSolution, solution) = solutionSet->getBestSolution(
          "manipulability", OptimizationGoal::Maximize);
    }
    else
    {
      std::tie(hasSolution, solution) =
          solutionSet->getBestSolution("error", OptimizationGoal::Minimize);
    }

    if (hasSolution)
    {
      bestSolution = solution;
    }

    if (solver_opts_.verb_level_ >= 2)
    {
      // Print all solutions found
      solutionSet->printAllSolutions();
      if (solver_opts_.solve_mode_ == "distance")
        solutionSet->printBestSolution("distance", OptimizationGoal::Minimize);
      else if (solver_opts_.solve_mode_ == "manipulability")
        solutionSet->printBestSolution("manipulability",
                                       OptimizationGoal::Maximize);
      else
        solutionSet->printBestSolution("error", OptimizationGoal::Minimize);
    }

    if (solver_opts_.verb_level_ >= 2)
    {
      std::cout << "[CODCS-IK] Time IK [us]: " << duration_cb.count() << " usec"
                << std::endl;
      std::cout << "[CODCS-IK] Solution found: " << q_sol.transpose()
                << std::endl;
      std::cout << "[CODCS-IK] Solution found (deg): "
                << q_sol.transpose() * (180 / M_PI) << std::endl;
    }

    return res;
  }

  int COIKS::updateFK(const VectorXd q_act)
  {
    int res = 1;

    pin::forwardKinematics(model_, mdl_data_, q_act);
    pin::updateFramePlacements(model_, mdl_data_);

    return res;
  }

  int COIKS::getEEPose(pin::SE3 &x_B_Fid)
  {
    int res = 1;

    x_B_Fid = mdl_data_.oMf[ee_id_];
    return res;
  }

  int COIKS::getFramePose(pin::SE3 &x_B_Fid, const pin::FrameIndex ee_id)
  {
    int res = 1;

    x_B_Fid = mdl_data_.oMf[ee_id];
    return res;
  }

  int COIKS::getFramePose(pin::SE3 &x_B_Fid, const std::string link_name)
  {
    int res = 1;

    pin::FrameIndex ee_id = model_.getFrameId(link_name);
    x_B_Fid = mdl_data_.oMf[ee_id];
    return res;
  }

  int COIKS::concurrentIkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                               VectorXd &q_sol)
  {
    int res = 1;

    q_solutions_.clear();
    m_solutions_.clear();

    if (ik_solver_name_ == "coiks_invj_nlo")
    {
      invj_solver_->reset();
      nlo_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJNLOIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOINVJIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_invj_qp")
    {
      invj_solver_->reset();
      qp_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJQPIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runQPINVJIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_qp_nlo")
    {
      nlo_solver_->reset();
      qp_solver_->reset();

      solver1_ = std::thread(&COIKS::runQPNLOIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOQPIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_all")
    {
      invj_solver_->reset();
      nlo_solver_->reset();
      qp_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJALLIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOALLIK, this, q_init, x_des);
      solver3_ = std::thread(&COIKS::runQPALLIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
      solver3_.join();
    }

    else if (ik_solver_name_ == "coiks_invj_multi")
    {
      for (int i = 0; i < solver_opts_.invj_multi_n_solvers_; i++)
      {
        invj_multi_solvers_[i]->reset();
      }

      for (int i = 0; i < solver_opts_.invj_multi_n_solvers_; i++)
      {
        invj_multi_solvers_threads_[i] =
            std::thread(&COIKS::runINVJMULTIK, this, q_init, x_des, i);
      }

      for (int i = 0; i < solver_opts_.invj_multi_n_solvers_; i++)
      {
        invj_multi_solvers_threads_[i].join();
      }
    }
    else
    {
      std::cout << "No IK solver found" << std::endl;
      return 1;
    }

    if (!q_solutions_.empty())
    {
      if (solver_opts_.solve_mode_ == "speed")
      {
        q_sol = q_solutions_[0];
      }
      else if (solver_opts_.solve_mode_ == "distance")
      {
        // Find the index of the minimum element in m_solutions_
        auto min_it =
            std::min_element(m_solutions_.begin(), m_solutions_.end());

        // Calculate the index
        size_t min_index = std::distance(m_solutions_.begin(), min_it);

        // Use this index to get the corresponding solution from q_solutions_
        q_sol = q_solutions_[min_index];
      }
      else if (solver_opts_.solve_mode_ == "manipulability")
      {
        // Find the index of the maximum element in m_solutions_
        auto max_it =
            std::max_element(m_solutions_.begin(), m_solutions_.end());

        // Calculate the index
        size_t max_index = std::distance(m_solutions_.begin(), max_it);

        // Get the solution with the index of the largest value
        q_sol = q_solutions_[max_index];
      }
      else
        q_sol = q_solutions_[0];

      if (solver_opts_.verb_level_ >= 2)
        printStatistics();
      res = 0;
    }

    return res;
  }

  void COIKS::printStatistics()
  {
    std::cout << "Solutions found: " << q_solutions_.size() << std::endl;

    std::cout << "TP"
              << " found " << succ_sol_tp_ << " solutions" << std::endl;
    std::cout << "NLO"
              << " found " << succ_sol_nlo_ << " solutions" << std::endl;
    std::cout << "QP"
              << " found " << succ_sol_qp_ << " solutions" << std::endl;
  }

  bool COIKS::initialize()
  {
    std::cout << "Initializing COIKS with Max. Error: " << max_error_
              << " Max. Time:" << max_time_ << " Max. It.:" << max_iter_
              << " Delta-T:" << dt_ << std::endl;
    //   Load the urdf model
    pin::urdf::buildModel(urdf_file_, model_);
    mdl_data_ = pin::Data(model_);

    n_joints_ = model_.nq;
    // q_sol_.resize(n_joints_);
    q_solutions_.clear();
    errors_.clear();
    succ_sol_tp_ = 0;
    succ_sol_nlo_ = 0;
    succ_sol_qp_ = 0;

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    ee_id_ = model_.getFrameId(ee_link_);
    printModelInfo();

    if (solver_opts_.solve_mode_ == "distance")
      metrics = {"error", "error_pose", "error_ori", "time", "distance"};
    else if (solver_opts_.solve_mode_ == "speed")
      metrics = {"error", "error_pose", "error_ori", "time"};
    else if (solver_opts_.solve_mode_ == "manipulability")
      metrics = {"error", "error_pose", "error_ori", "time", "manipulability"};
    else
      metrics = {
          "error",
          "error_pose",
          "error_ori",
          "time",
      };

    solutionSet = std::make_unique<IKSolutionSet>(metrics);

    if (ik_solver_name_ == "coiks_invj")
      invj_solver_.reset(new INVJ_IkSolver<INVJ_BASE>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_nlo")
      nlo_solver_.reset(new NLO_IkSolver<NLO_BASE>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_qp")
      qp_solver_.reset(new QP_IkSolver<QP_BASE>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_invj_nlo")
      initialize_coiks_invj_nlo();
    else if (ik_solver_name_ == "coiks_invj_qp")
      initialize_coiks_invj_qp();
    else if (ik_solver_name_ == "coiks_qp_nlo")
      initialize_coiks_qp_nlo();
    else if (ik_solver_name_ == "coiks_all")
      initialize_codcs();
    else if (ik_solver_name_ == "coiks_invj_multi")
      initialize_codcs_invj_multi(solver_opts_.invj_multi_n_solvers_);
    else
    {
      std::cout << "No IK solver found. Using default: COIKS-INVJ" << std::endl;
      invj_solver_.reset(new INVJ_IkSolver<INVJ_BASE>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    }

    std::cout << "COIKS initialized" << std::endl;
    initialized_ = true;

    unsigned int seed = 42; // Chosen seed value for reproducibility

    // Set the global seed
    std::default_random_engine global_rng(seed);

    return true;
  }

  bool COIKS::initialize_coiks_invj_nlo()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    nlo_solver_.reset(new NLO_IkSolver<NLO_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    return true;
  }

  bool COIKS::initialize_coiks_invj_qp()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    qp_solver_.reset(new QP_IkSolver<QP_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::initialize_coiks_qp_nlo()
  {
    nlo_solver_.reset(new NLO_IkSolver<NLO_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    qp_solver_.reset(new QP_IkSolver<QP_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::initialize_codcs()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    nlo_solver_.reset(new NLO_IkSolver<NLO_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    qp_solver_.reset(new QP_IkSolver<QP_BASE>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::initialize_codcs_invj_multi(int n_solvers)
  {
    std::cout << "Initializing " << n_solvers << " INVJ solvers" << std::endl;
    for (int i = 0; i < n_solvers; i++)
    {
      invj_multi_solvers_.push_back(std::make_unique<INVJ_IkSolver<INVJ_BASE>>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_,
          i));
    }
    invj_multi_solvers_threads_.resize(n_solvers);
    return true;
  }

  bool COIKS::printModelInfo()
  {
    std::cout << "\nPrinting Model Info \n-----------------------" << std::endl;
    std::cout << "Number of Joints found in model: " << model_.njoints << "\n";
    std::cout << "Model nq (positon states): " << model_.nq << "\n";
    std::cout << "Model nv (velocity states): " << model_.nv << "\n";
    std::cout << "Joints lower limits [rad]: "
              << model_.lowerPositionLimit.transpose() << "\n";
    std::cout << "Joints upper limits [rad]: "
              << model_.upperPositionLimit.transpose() << "\n";
    std::cout << "EE link name: " << ee_link_ << std::endl;
    std::cout << "EE link frame id: " << ee_id_ << std::endl;
    return true;
  }

  template <typename T1, typename T2>
  bool COIKS::run2Solver(T1 &solver, T2 &other_solver1, const VectorXd q_init,
                         const pin::SE3 &x_des, int id)
  {
    VectorXd q_sol;
    double   time_left;

    std::chrono::microseconds diff;
    std::chrono::microseconds diff_solver;

    while (true)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_iksolve_time_);
      time_left = max_time_ - diff.count() / 1000000.0;

      if (time_left <= 0)
        break;

      solver.set_max_time(time_left);

      bool res = solver.IkSolve(q_init, x_des, q_sol);

      mtx_.lock();

      if (!res)
      {
        if (id == 1)
        {
          succ_sol_tp_++;
          if (q_solutions_.empty())
            sol_id_ = 1;
        }
        else if (id == 2)
        {
          succ_sol_nlo_++;
          if (q_solutions_.empty())
            sol_id_ = 2;
        }
        else if (id == 3)
        {
          succ_sol_qp_++;
          if (q_solutions_.empty())
            sol_id_ = 3;
        }
        q_solutions_.push_back(q_sol);
      }
      mtx_.unlock();

      if (!q_solutions_.empty())
      {
        break;
      }
    }

    other_solver1.abort();
    solver.set_max_time(max_time_);

    return true;
  }

  template <typename T1, typename T2, typename T3>
  bool COIKS::run3Solver(T1 &solver, T2 &other_solver1, T3 &other_solver2,
                         const VectorXd q_init, const pin::SE3 &x_des, int id)
  {
    VectorXd q_sol;
    double   time_left;

    std::chrono::microseconds diff;
    std::chrono::microseconds diff_solver;

    while (true)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_iksolve_time_);
      time_left = max_time_ - diff.count() / 1000000.0;

      if (time_left <= 0)
        break;

      solver.set_max_time(time_left);

      bool res = solver.IkSolve(q_init, x_des, q_sol);

      mtx_.lock();

      if (!res)
      {
        if (id == 1)
        {
          succ_sol_tp_++;
          if (q_solutions_.empty())
            sol_id_ = 1;
        }
        else if (id == 2)
        {
          succ_sol_nlo_++;
          if (q_solutions_.empty())
            sol_id_ = 2;
        }
        else if (id == 3)
        {
          succ_sol_qp_++;
          if (q_solutions_.empty())
            sol_id_ = 3;
        }
        q_solutions_.push_back(q_sol);
      }
      mtx_.unlock();

      if (!q_solutions_.empty())
      {
        break;
      }
    }

    other_solver1.abort();
    other_solver2.abort();
    solver.set_max_time(max_time_);

    return true;
  }

  template <typename T>
  bool COIKS::runMultiSolver(std::vector<std::unique_ptr<T>> &solvers,
                             const VectorXd q_init, const pin::SE3 &x_des,
                             int id)
  {
    VectorXd q_sol;
    double   time_left;

    std::chrono::microseconds diff;
    std::chrono::microseconds diff_solver;

    bool res;

    IKSolutionSet::Solution solution;

    while (true)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_iksolve_time_);

      time_left = max_time_ - diff.count() / 1000000.0;

      if (solver_opts_.verb_level_ >= 2)
      {
        std::cout << "Time left: " << time_left << " s" << std::endl;
      }
      if (time_left <= 0)
      {
        break;
      }
      // Set the max time for the solver[id]
      solvers[id]->set_max_time(time_left);

      if (solver_opts_.solve_mode_ == "speed" ||
          solver_opts_.solve_mode_ == "distance" ||
          solver_opts_.solve_mode_ == "manipulability")
      {
        res = solvers[id]->IkSolveSet(q_init, x_des, q_sol, solution);
      }
      else
      {
        throw std::invalid_argument("Invalid solve_mode: " +
                                    solver_opts_.solve_mode_);
      }

      mtx_.lock();

      if (!res)
      {
        if (q_solutions_.empty())
          sol_id_ = id;
        // Store the solution
        q_solutions_.push_back(q_sol);

        solution.elapsed_time =
            diff.count() / 1000000.0 + solution.metrics["time"];
        solution.solver_id = id;
        solution.solution_id = q_solutions_.size();
        solutionSet->addSolution(solution);
      }

      mtx_.unlock();

      if (!q_solutions_.empty() and solver_opts_.solve_mode_ == "speed")
      {
        break;
      }
    }

    // Abort all the other solvers
    for (auto &solver : solvers)
    {
      if (solver != solvers[id])
        solver->abort();
    }
    // Set the max time for the solver[id]
    solvers[id]->set_max_time(max_time_);

    return true;
  }
} // namespace coiks