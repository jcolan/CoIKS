
#include <coiks/coiks.hpp>
#include <coiks/ik_solver.hpp>

#include <csignal>
#include <string>

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/JointState.h>

// Orocos KDL
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/velocityprofile_spline.hpp>
#include <kdl/velocityprofile_trap.hpp>

// Eigen conversions
#include <eigen_conversions/eigen_kdl.h>
#include <eigen_conversions/eigen_msg.h>

// Trac_IK
#include <trac_ik/trac_ik.hpp>

// KDL Parser
#include <kdl_parser/kdl_parser.hpp>

#include <coiks/solution_set.hpp>

using namespace coiks;
namespace pin = pinocchio;

class BenchmarkKDL : public IkSolver
{
  public:
  BenchmarkKDL(const std::string &_urdf_file, const std::string &_base_link,
               const std::string &_ee_link, const std::string &_ik_solver,
               double _max_time, double _max_error, int _max_iter, double _dt)
      : initialized_(false), max_error_(_max_error), max_iter_(_max_iter),
        urdf_file_(_urdf_file), base_link_(_base_link), ee_link_(_ee_link),
        dt_(_dt), solver_name_("kdl")
  {
    initialize_kdl();
  }

  ~BenchmarkKDL() {}

  bool initialize_kdl()
  {
    std::cout << "Initializing KDL with Max. Error: " << max_error_
              << " Max. It.:" << max_iter_ << " Delta T:" << dt_ << std::endl;

    double maxtime = 0.005;

    // Parsing URDF
    if (!kdl_parser::treeFromFile(urdf_file_, kdl_tree_))
    {
      ROS_ERROR("Failed to construct kdl tree");
      return false;
    }
    bool exit_value = kdl_tree_.getChain(base_link_, ee_link_, kdl_chain_);
    // Resize variables
    n_joints_ = kdl_chain_.getNrOfJoints();
    qtmp_.resize(kdl_chain_.getNrOfJoints());
    nominal_.resize(kdl_chain_.getNrOfJoints());
    ll_.resize(kdl_chain_.getNrOfJoints());
    ul_.resize(kdl_chain_.getNrOfJoints());

    //   Load the urdf model
    pin::Model model_;
    pin::urdf::buildModel(urdf_file_, model_);

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    // Storing Joint limits
    for (int i = 0; i < kdl_chain_.getNrOfJoints(); i++)
    {
      ll_.data(i) = q_ll_[i];
      ul_.data(i) = q_ul_[i];
    }

    assert(kdl_chain_.getNrOfJoints() == ll_.data.size());
    assert(kdl_chain_.getNrOfJoints() == ul_.data.size());

    kdl_fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(kdl_chain_));
    kdl_vik_solver_.reset(new KDL::ChainIkSolverVel_pinv(kdl_chain_));
    kdl_ik_solver_.reset(new KDL::ChainIkSolverPos_NR_JL(
        kdl_chain_, ll_, ul_, *kdl_fk_solver_, *kdl_vik_solver_, max_iter_,
        max_error_));

    // Initialize nominal vector
    for (uint j = 0; j < nominal_.data.size(); j++)
    {
      nominal_(j) = (ll_(j) + ul_(j)) / 2.0;
    }
    std::cout << "KDL initialized" << std::endl;
    initialized_ = true;

    return true;
  }

  // * KDL solver
  int IkSolve(const VectorXd q_init, const pin::SE3 &x_Fee_d, VectorXd &q_out)
  {
    bool          success = false;
    int           rc;
    KDL::JntArray qd(n_joints_);
    KDL::Frame    ee;

    Affine3d Tdes;
    Tdes.linear() = x_Fee_d.rotation();
    Tdes.translation() = x_Fee_d.translation();

    tf::transformEigenToKDL(Tdes, ee);

    rc = kdl_ik_solver_->CartToJnt(nominal_, ee, qd);

    if (rc >= 0)
    {
      // std::cout << "Solution found" << std::endl;
      q_out = VectorXd::Map(&qd.data[0], qd.data.size());
      // std::cout << "Solution found: " << qd.data * (180 / M_PI) << std::endl;
      // std::cout << "Time IK [us]: " << duration_cb.count() << " usec";

      success = true;
      return 0;
      // ROS_WARN_STREAM("IK solution found: " <<
      // des_joint_pos_.transpose());
    }
    else
    {
      // ROS_WARN_STREAM("No IK solution found");
      success = false;
      return 1;
    }
    // return success;
  }

  int         get_n_joints() { return n_joints_; }
  std::string get_solver_name() { return solver_name_; }

  private:
  // Temporary variables for KDL
  KDL::Tree  kdl_tree_;
  KDL::Chain kdl_chain_;

  KDL::JntArray nominal_;
  KDL::JntArray qtmp_;
  KDL::JntArray ll_, ul_;
  KDL::Frame    xtmp_;
  KDL::Jacobian Jtmp_;

  KDL::Twist    xdot_temp_;
  KDL::JntArray qdot_tmp_;

  // KDL
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> kdl_fk_solver_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv>      kdl_vik_solver_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL>     kdl_ik_solver_;

  std::string urdf_file_;

  std::string base_link_;
  std::string ee_link_;
  VectorXd    q_ul_;
  VectorXd    q_ll_;

  int         n_joints_;
  std::string solver_name_;
  int         max_iter_;
  double      max_error_;

  bool   initialized_;
  double dt_;
};

class BenchmarkTRACIK : public IkSolver
{
  public:
  BenchmarkTRACIK(const std::string &_urdf_file, const std::string &_base_link,
                  const std::string &_ee_link, const std::string &_ik_solver,
                  const std::string &_solver_mode, double _max_time,
                  double _max_error, int _max_iter, double _dt)
      : initialized_(false), max_error_(_max_error), max_time_(_max_time),
        urdf_file_(_urdf_file), base_link_(_base_link), ee_link_(_ee_link),
        dt_(_dt), solver_name_("trac_ik"), solver_mode_(_solver_mode)
  {
    initialize_tracik();
  }

  ~BenchmarkTRACIK() {}

  bool initialize_tracik()
  {
    std::cout << "Initializing TRACIK with Max. Error: " << max_error_
              << " Max. Time:" << max_time_ << " Delta T:" << dt_ << std::endl;

    // Parsing URDF
    if (!kdl_parser::treeFromFile(urdf_file_, kdl_tree_))
    {
      ROS_ERROR("Failed to construct kdl tree");
      return false;
    }
    bool exit_value = kdl_tree_.getChain(base_link_, ee_link_, kdl_chain_);

    // Resize variables
    n_joints_ = kdl_chain_.getNrOfJoints();
    qtmp_.resize(kdl_chain_.getNrOfJoints());
    nominal_.resize(kdl_chain_.getNrOfJoints());
    ll_.resize(kdl_chain_.getNrOfJoints());
    ul_.resize(kdl_chain_.getNrOfJoints());

    //   Load the urdf model
    pin::Model model_;
    pin::urdf::buildModel(urdf_file_, model_);

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    // Storing Joint limits
    for (int i = 0; i < kdl_chain_.getNrOfJoints(); i++)
    {
      ll_.data(i) = q_ll_[i];
      ul_.data(i) = q_ul_[i];
    }

    assert(kdl_chain_.getNrOfJoints() == ll_.data.size());
    assert(kdl_chain_.getNrOfJoints() == ul_.data.size());

    if (solver_mode_ == "speed")
      tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, ll_, ul_, max_time_,
                                                max_error_, TRAC_IK::Speed));
    else if (solver_mode_ == "distance")
      tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, ll_, ul_, max_time_,
                                                max_error_, TRAC_IK::Distance));
    else if (solver_mode_ == "manipulability")
      tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, ll_, ul_, max_time_,
                                                max_error_, TRAC_IK::Manip1));
    else
    {
      ROS_WARN("Invalid solver mode. Using Speed mode");
      tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, ll_, ul_, max_time_,
                                                max_error_, TRAC_IK::Speed));
    }

    // Initialize nominal vector
    for (uint j = 0; j < nominal_.data.size(); j++)
    {
      nominal_(j) = (ll_(j) + ul_(j)) / 2.0;
    }
    std::cout << "TRACIK initialized" << std::endl;

    return true;
  }

  // * TRAC-IK solver
  int IkSolve(const VectorXd q_init, const pin::SE3 &x_Fee_d, VectorXd &q_out)
  {
    bool          success = false;
    int           rc;
    KDL::JntArray qd(n_joints_);
    KDL::Frame    ee;

    Affine3d Tdes;
    Tdes.linear() = x_Fee_d.rotation();
    Tdes.translation() = x_Fee_d.translation();

    tf::transformEigenToKDL(Tdes, ee);

    rc = tracik_solver_->CartToJnt(nominal_, ee, qd);

    if (rc >= 0)
    {
      q_out = VectorXd::Map(&qd.data[0], qd.data.size());
      success = true;
      return 0;
    }
    else
    {
      success = false;
      return 1;
    }
  }

  int         get_n_joints() { return n_joints_; }
  std::string get_solver_name() { return solver_name_; }

  private:
  std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;

  std::string urdf_file_;
  std::string solver_mode_;

  std::string base_link_;
  std::string ee_link_;
  VectorXd    q_ul_;
  VectorXd    q_ll_;

  int         n_joints_;
  std::string solver_name_;
  double      max_error_;
  double      max_time_;

  // Temporary variables for KDL
  KDL::Tree  kdl_tree_;
  KDL::Chain kdl_chain_;

  KDL::JntArray nominal_;
  KDL::JntArray qtmp_;
  KDL::JntArray ll_, ul_;

  bool   initialized_;
  double dt_;
};

double fRand(double min, double max)
{
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

void solveIkFromRandomList(std::string urdf_file_, std::string qinit_file_,
                           std::string target_file_, int num_samples,
                           IkSolver *ik_solver, std::string ee_link_,
                           bool print_all = false, int robot_id = 0)
{
  double                  total_time = 0;
  double                  total_time_it = 0;
  double                  success_time_it = 0;
  uint                    n_success = 0;
  IKSolutionSet::Solution bestSolution;

  //   Pinocchio variables
  pin::Model      model_;
  pin::Data       mdl_data_;
  pin::FrameIndex ee_id_;

  VectorXd q_ul_;
  VectorXd q_ll_;
  VectorXd q_sol_;

  // std::vector<VectorXd> q_solutions_;
  std::vector<double> errors_;
  int                 n_joints_;

  std::fstream fout;
  int          prob_id = 0;

  //   Load the urdf model
  pin::urdf::buildModel(urdf_file_, model_);
  ee_id_ = model_.getFrameId(ee_link_);

  n_joints_ = model_.nq;
  q_sol_.resize(n_joints_);
  // q_solutions_.clear();
  errors_.clear();

  // Getting Joints Limits
  q_ul_ = model_.upperPositionLimit;
  q_ll_ = model_.lowerPositionLimit;

  std::cout << "Solving IK with " << ik_solver->get_solver_name() << " for "
            << num_samples << " random configurations for link " << ee_link_
            << std::endl;

  // Create desired number of valid, random joint configurations
  std::vector<VectorXd> qinit_list;
  std::vector<pin::SE3> xdes_list;
  VectorXd              qinit(ik_solver->get_n_joints());

  // Read qinit values from CSV file
  std::cout << "Reading from file: " << qinit_file_ << std::endl;
  std::string   line;
  std::ifstream file(qinit_file_);

  if (file.is_open())
  {
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line))
    {
      std::stringstream   linestream(line);
      std::string         value;
      std::vector<double> values;

      // Read problem_id
      std::getline(linestream, value, ',');
      int problem_id = std::stoi(value);

      // Read joint values
      for (int i = 0; i < qinit.size(); ++i)
      {
        std::getline(linestream, value, ',');
        qinit(i) = std::stod(value);
      }

      qinit_list.push_back(qinit);
    }
    file.close();
  }
  else
  {
    std::cerr << "Unable to open file: " << qinit_file_ << std::endl;
  }

  // Read target values from CSV file
  std::cout << "Reading from file: " << target_file_ << std::endl;
  std::ifstream file2(target_file_);

  if (file2.is_open())
  {
    // Skip the header line
    std::getline(file2, line);

    while (std::getline(file2, line))
    {
      std::stringstream   linestream(line);
      std::string         value;
      std::vector<double> values;

      // Read problem_id
      std::getline(linestream, value, ',');
      int problem_id = std::stoi(value);

      // Read target values
      pin::SE3 xdes;
      for (int i = 0; i < 3; ++i)
      {
        std::getline(linestream, value, ',');
        xdes.translation()(i) = std::stod(value);
      }
      Eigen::Quaterniond quat;
      std::getline(linestream, value, ',');
      quat.w() = std::stod(value);

      std::getline(linestream, value, ',');
      quat.x() = std::stod(value);

      std::getline(linestream, value, ',');
      quat.y() = std::stod(value);

      std::getline(linestream, value, ',');
      quat.z() = std::stod(value);

      xdes.rotation() = quat.toRotationMatrix();
      xdes_list.push_back(xdes);
    }
    file2.close();
  }
  else
  {
    std::cerr << "Unable to open file: " << target_file_ << std::endl;
  }

  pin::Data mdl_data(model_);

  time_t     currentTime;
  struct tm *localTime;

  time(&currentTime); // Get the current time
  localTime = localtime(&currentTime);
  if (ik_solver->get_solver_name() == "kdl")
  {
    fout.open("/home/colan/kdl_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,time,q_init,q_sol\n";
  }
  else if (ik_solver->get_solver_name() == "trac_ik")
  {
    fout.open("/home/colan/tracik_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,time,q_init,q_sol\n";
  }
  else if (ik_solver->get_solver_name().substr(0, 5) == "coiks")
  {
    fout.open("/home/colan/coiks_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,res,time,solver_id,sol_id,error,q_init,q_sol\n";
  }
  else if (ik_solver->get_solver_name() == "invj")
  {
    fout.open("/home/colan/invj_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,res,time,solver_id,sol_id,error,q_init,q_sol\n";
  }
  else if (ik_solver->get_solver_name() == "nlo")
  {
    fout.open("/home/colan/nlo_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,res,time,solver_id,sol_id,error,q_init,q_sol\n";
  }
  else if (ik_solver->get_solver_name() == "qp")
  {
    fout.open("/home/colan/qp_r" + std::to_string(robot_id) + "_" +
                  std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,res,time,solver_id,sol_id,error,q_init,q_sol\n";
  }

  prob_id = 0;

  auto start_cb_time = std::chrono::high_resolution_clock::now();
  auto start_it_time = std::chrono::high_resolution_clock::now();
  auto stop_it_time = std::chrono::high_resolution_clock::now();

  for (uint i = 0; i < num_samples; i++)
  {
    // Initialize initial configuration
    VectorXd q_init = pin::neutral(model_);
    q_init = qinit_list[i];

    // Initializing solution
    VectorXd q_sol = pin::neutral(model_);

    // Selecting target pose
    pin::SE3 x_des = xdes_list[i];

    prob_id++;
    if (print_all)
      std::cout << "\n[Prob " << i
                << "] Solving for Joints: " << q_init.transpose() << std::endl;

    if (print_all)
      std::cout << "[Prob " << i
                << "] Solving for Pos: " << x_des.translation().transpose()
                << std::endl;
    start_it_time = std::chrono::high_resolution_clock::now();

    // Call IK Solver
    int res = 1;
    if (ik_solver->get_solver_name() == "kdl" ||
        ik_solver->get_solver_name() == "trac_ik")
    {
      res = ik_solver->IkSolve(q_init, x_des, q_sol);
    }
    else
    {
      res = ik_solver->IkSolve(q_init, x_des, q_sol, bestSolution);
    }

    stop_it_time = std::chrono::high_resolution_clock::now();
    auto duration_it = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_it_time - start_it_time);
    total_time_it += duration_it.count();

    if (res == 0 && ik_solver->get_solver_name() != "kdl" &&
        ik_solver->get_solver_name() != "trac_ik" && print_all)
    {
      std::cout << "IK solution found: " << q_sol.transpose() << std::endl;

      // You can now access the best solution details
      std::cout << "Best solution metric: " << bestSolution.metrics["error"]
                << std::endl;
      std::cout << "Best solution joint positions: "
                << bestSolution.configuration.transpose() << std::endl;
      // Access other fields of bestSolution as needed
    }

    if (res && print_all)
      ROS_WARN("Solution not found");

    if (print_all)
    {
      std::cout << "Time: " << duration_it.count() << " [us]" << std::endl;
      if (!res)
      {
        std::cout << "Solution: " << q_sol.transpose() << std::endl;
      }
    }

    if (ik_solver->get_solver_name() != "kdl" &&
        ik_solver->get_solver_name() != "trac_ik")
    {
      if (!res)
      {
        fout << prob_id << "," << res << "," << duration_it.count() << ","
             << bestSolution.solver_id << "," << bestSolution.solution_id << ","
             << bestSolution.metrics["error"] << "," << q_init.transpose()
             << "," << bestSolution.configuration.transpose() << "\n";
      }
      else
        fout << prob_id << "," << res << "," << duration_it.count() << "," << 0
             << "," << 0 << "," << 0 << "," << q_init.transpose() << ","
             << q_init.transpose() << "\n";
    }

    if (!res)
    {
      success_time_it += duration_it.count();
      n_success++;

      if (ik_solver->get_solver_name() == "kdl")
      {
        fout << prob_id << "," << duration_it.count() << ","
             << q_init.transpose() << "," << q_sol.transpose() << "\n";
      }
      else if (ik_solver->get_solver_name() == "trac_ik")
      {
        fout << prob_id << "," << duration_it.count() << ","
             << q_init.transpose() << "," << q_sol.transpose() << "\n";
      }
    }
  }

  auto stop_cb_time = std::chrono::high_resolution_clock::now();
  auto duration_cb = std::chrono::duration_cast<std::chrono::microseconds>(
      stop_cb_time - start_cb_time);

  total_time = duration_cb.count();

  std::cout << "------------------------------" << std::endl;
  std::cout << "Summary:" << std::endl;

  std::cout << ik_solver->get_solver_name() << " found " << n_success
            << " solutions (" << 100.0 * n_success / num_samples
            << "\%) with a total average of " << total_time / num_samples
            << " usec/config. Solving average of "
            << total_time_it / num_samples << " usec/config."
            << "Success solving average of " << success_time_it / n_success
            << " usec/config." << std::endl;
  if (ik_solver->get_solver_name() == "codcs")
  {
    ik_solver->printStatistics();
  }
  std::cout << "------------------------------" << std::endl;

  fout.close();

  return;
}

bool kill_process = false;
void SigIntHandler(int signal)
{
  kill_process = true;
  ROS_INFO_STREAM("SHUTDOWN SIGNAL RECEIVED");
}

int main(int argc, char **argv)
{
  ROS_INFO("Unscontrained IK Benchmarking for COIKS, TRAC-IK and KDL");
  ros::init(argc, argv, "coiks_benchmark");
  ros::NodeHandle nh;
  std::signal(SIGINT, SigIntHandler);

  // ROS parameters
  std::string ik_solver;
  int         max_iter;
  int         n_random_config;
  double      max_time;
  double      max_error;
  double      dt;
  int         print_all;

  if (!nh.getParam("ik_solver", ik_solver))
  {
    ik_solver = "tracik";
  }

  std::string solve_mode;
  if (!nh.getParam("solve_mode", solve_mode))
  {
    solve_mode = "speed";
  }

  std::string ee_link_name;
  if (!nh.getParam("ee_link_name", ee_link_name))
  {
    ee_link_name = "ee_link";
  }
  if (!nh.getParam("max_iter", max_iter))
  {
    max_iter = 10000;
  }
  if (!nh.getParam("max_time", max_time))
  {
    max_time = 5e-3;
  }
  if (!nh.getParam("max_error", max_error))
  {
    max_error = 1e-5;
  }
  if (!nh.getParam("delta_integration", dt))
  {
    dt = 1.0;
  }
  std::string error_type;
  if (!nh.getParam("error_type", error_type))
  {
    error_type = "log6";
  }

  //* Esperiment variables
  int robot_id;
  if (!nh.getParam("robot_id", robot_id))
  {
    robot_id = 0;
  }
  if (!nh.getParam("n_random_config", n_random_config))
  {
    n_random_config = 100;
  }

  // Additional variables
  std::string step_size_method;
  if (!nh.getParam("step_size_method", step_size_method))
  {
    step_size_method = "fixed";
  }

  std::string seed_method;
  if (!nh.getParam("seed_method", seed_method))
  {
    seed_method = "current";
  }

  std::string limiting_method;
  if (!nh.getParam("limiting_method", limiting_method))
  {
    limiting_method = "random";
  }

  std::string pinv_method;
  if (!nh.getParam("pinv_method", pinv_method))
  {
    pinv_method = "cod";
  }

  //* INVJ Variables
  double invj_Ke1;
  if (!nh.getParam("invj_Ke1", invj_Ke1))
  {
    invj_Ke1 = 1.0;
  }

  int invj_multi_n_solvers;
  if (!nh.getParam("invj_multi_n_solvers", invj_multi_n_solvers))
  {
    invj_multi_n_solvers = 2;
  }

  int invj_max_stagnation_iter;
  if (!nh.getParam("invj_max_stagnation_iter", invj_max_stagnation_iter))
  {
    invj_max_stagnation_iter = 10;
  }

  double invj_improvement_threshold;
  if (!nh.getParam("invj_improvement_threshold", invj_improvement_threshold))
  {
    invj_improvement_threshold = 1e-5;
  }

  //* NLO Variables
  std::string nlo_linear_solver;
  if (!nh.getParam("nlo_linear_solver", nlo_linear_solver))
  {
    nlo_linear_solver = "ma57";
  }

  double mu0;
  if (!nh.getParam("nlo_mu0", mu0))
  {
    mu0 = 1.0;
  }

  double mu1;
  if (!nh.getParam("nlo_mu1", mu1))
  {
    mu1 = 0.005;
  }

  double mu2;
  if (!nh.getParam("nlo_mu2", mu2))
  {
    mu2 = 0.001;
  }

  double mu3;
  if (!nh.getParam("nlo_mu3", mu3))
  {
    mu3 = 100.0;
  }

  double mu4;
  if (!nh.getParam("nlo_mu4", mu4))
  {
    mu4 = 0.01;
  }

  bool nlo_concurrent;
  if (!nh.getParam("nlo_concurrent", nlo_concurrent))
  {
    nlo_concurrent = false;
  }
  std::string nlo_error_type;
  if (!nh.getParam("nlo_error_type", nlo_error_type))
  {
    nlo_error_type = "log3";
  }
  int nlo_concurrent_iterations;
  if (!nh.getParam("nlo_concurrent_iterations", nlo_concurrent_iterations))
  {
    nlo_concurrent_iterations = 5;
  }
  std::string nlo_warm_start;
  if (!nh.getParam("nlo_warm_start", nlo_warm_start))
  {
    nlo_warm_start = "yes";
  }

  //* QP variables
  double qp_K_t1;
  if (!nh.getParam("qp_K_t1", qp_K_t1))
  {
    qp_K_t1 = 1.0;
  }
  double qp_Kr_t1;
  if (!nh.getParam("qp_Kr_t1", qp_Kr_t1))
  {
    qp_Kr_t1 = 0.00001;
  }
  double qp_Kw_p1;
  if (!nh.getParam("qp_Kw_p1", qp_Kw_p1))
  {
    qp_Kw_p1 = 0.00001;
  }
  double qp_Kd_p1;
  if (!nh.getParam("qp_Kd_p1", qp_Kd_p1))
  {
    qp_Kd_p1 = 0.00001;
  }
  bool qp_warm_start;
  if (!nh.getParam("qp_warm_start", qp_warm_start))
  {
    qp_warm_start = true;
  }

  //* Printing variables
  std::string time_stats;
  if (!nh.getParam("solv_time_stats", time_stats))
  {
    time_stats = "no";
  }
  int verb_level;
  if (!nh.getParam("solv_verb_level", verb_level))
  {
    verb_level = 0;
  }
  if (!nh.getParam("print_all", print_all))
  {
    print_all = 0;
  }

  bool tracik_enable;
  if (!nh.getParam("tracik_enable", tracik_enable))
  {
    tracik_enable = true;
  }

  bool kdl_enable;
  if (!nh.getParam("kdl_enable", kdl_enable))
  {
    kdl_enable = true;
  }

  // Setting up URDF path
  std::string pkg_path = ros::package::getPath("coiks_benchmark");

  std::string urdf_path;
  std::string qinit_path;
  std::string target_path;

  switch (robot_id)
  {
  case 0:
    urdf_path = pkg_path + std::string("/urdf/") + "toyota/hsr/hsr_arm.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_hsr_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_hsr_random_10000samples.csv";
    break;

  case 1:
    urdf_path = pkg_path + std::string("/urdf/") + "denso/vs050/vs050.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_vs050_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_vs050_random_10000samples.csv";
    break;

  case 2:
    urdf_path = pkg_path + std::string("/urdf/") + "ufactory/xarm6/xarm6.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_xarm6_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_xarm6_random_10000samples.csv";
    break;

  case 3:
    urdf_path = pkg_path + std::string("/urdf/") + "abb/irb140/irb140.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_irb140_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_irb140_random_10000samples.csv";
    break;

  case 4:
    urdf_path = pkg_path + std::string("/urdf/") + "kinova/jaco/jaco_arm.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_jaco_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_jaco_random_10000samples.csv";
    break;

  case 5:
    urdf_path = pkg_path + std::string("/urdf/") + "ur/ur5/ur5.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_ur5_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_ur5_random_10000samples.csv";
    break;

  case 6:
    urdf_path = pkg_path + std::string("/urdf/") + "ur/ur10/ur10.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_ur10_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_ur10_random_10000samples.csv";
    break;

  case 7:
    urdf_path = pkg_path + std::string("/urdf/") + "kinova/gen3/gen3.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_gen3_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_gen3_random_10000samples.csv";
    break;

  case 8:
    urdf_path = pkg_path + std::string("/urdf/") + "kuka/iiwa/iiwa.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_iiwa_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_iiwa_random_10000samples.csv";
    break;

  case 9:
    urdf_path = pkg_path + std::string("/urdf/") + "ufactory/xarm7/xarm7.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_xarm7_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_xarm7_random_10000samples.csv";
    break;

  case 10:
    urdf_path = pkg_path + std::string("/urdf/") + "franka/panda.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_panda_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_panda_random_10000samples.csv";
    break;

  case 11:
    urdf_path = pkg_path + std::string("/urdf/") + "rethink/sawyer/sawyer.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_sawyer_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_sawyer_random_10000samples.csv";
    break;

  case 12:
    urdf_path = pkg_path + std::string("/urdf/") + "abb/yumi/yumi.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_yumi_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_yumi_random_10000samples.csv";
    break;

  case 13:
    urdf_path = pkg_path + std::string("/urdf/") + "fetch/fetch_arm.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_fetch_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_fetch_random_10000samples.csv";
    break;

  case 14:
    urdf_path = pkg_path + std::string("/urdf/") + "tiago/tiago.urdf";
    qinit_path = pkg_path + std::string("/data/") +
                 "qinit_robot_tiago_random_10000samples.csv";
    target_path = pkg_path + std::string("/data/") +
                  "targets_robot_tiago_random_10000samples.csv";
    break;

  default:
    ROS_ERROR("Kinematic Tree ID not recognized.");
    return -1;
  }
  ROS_INFO_STREAM("Using URDF found in: " << urdf_path);

  // Creating solver options class
  SolverOptions so;

  so.solve_mode_ = solve_mode;
  so.error_type_ = error_type;
  so.step_size_method_ = step_size_method;
  so.seed_method_ = seed_method;
  so.limiting_method_ = limiting_method;
  so.pinv_method_ = pinv_method;

  so.invj_Ke1_ = invj_Ke1;

  so.invj_multi_n_solvers_ = invj_multi_n_solvers;
  so.invj_max_stagnation_iterations_ = invj_max_stagnation_iter;
  so.invj_improvement_threshold_ = invj_improvement_threshold;

  so.nlo_linear_solver_ = nlo_linear_solver;
  so.cost_coeff_.push_back(mu0);
  so.cost_coeff_.push_back(mu1);
  so.cost_coeff_.push_back(mu2);
  so.cost_coeff_.push_back(mu3);
  so.cost_coeff_.push_back(mu4);
  so.nlo_concurrent_ = nlo_concurrent;
  so.nlo_error_type_ = nlo_error_type;
  so.nlo_concurrent_iterations_ = nlo_concurrent_iterations;
  so.nlo_warm_start_ = nlo_warm_start;

  so.qp_K_t1_ = qp_K_t1;
  so.qp_Kr_t1_ = qp_Kr_t1;
  so.qp_Kw_p1_ = qp_Kw_p1;
  so.qp_warm_start_ = qp_warm_start;

  so.time_stats_ = time_stats;
  so.verb_level_ = verb_level;

  ROS_INFO_STREAM("Error type: " << so.error_type_);

  // Initialiing KDL
  ROS_INFO("Starting KDL");
  BenchmarkKDL kdl_ik(urdf_path, "base_link", "link_ee", ik_solver, max_time,
                      max_error, max_iter, dt);

  // Initialiing TRAC-IK
  ROS_INFO("Starting TRAC-IK");
  BenchmarkTRACIK trac_ik(urdf_path, "base_link", "link_ee", ik_solver,
                          solve_mode, max_time, max_error, max_iter, dt);

  // Initialiing COIKS
  ROS_INFO("Starting CODCS-IK");
  COIKS coiks(urdf_path, "base_link", "link_ee", ik_solver, so, max_time,
              max_error, max_iter, dt);

  if (kdl_enable)
  {
    ROS_WARN("Running random configurations for KDL");
    solveIkFromRandomList(urdf_path, qinit_path, target_path, n_random_config,
                          &kdl_ik, "link_ee", (print_all & 1));
  }

  if (tracik_enable)
  {
    ROS_WARN("Running random configurations for TRAC-IK");
    solveIkFromRandomList(urdf_path, qinit_path, target_path, n_random_config,
                          &trac_ik, "link_ee", (print_all & 2) >> 1);
  }

  ROS_WARN("Running random configurations for COIKS");
  solveIkFromRandomList(urdf_path, qinit_path, target_path, n_random_config,
                        &coiks, "link_ee", (print_all & 4) >> 2, robot_id);
  ROS_INFO("Benchmark finished");

  return 0;
}
