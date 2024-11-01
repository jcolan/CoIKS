#ifndef IK_SOLVER_H
#define IK_SOLVER_H

// Eigen
#include <Eigen/Dense>

// Pinocchio
#include <pinocchio/algorithm/frames.hpp>

// Source HPP
#include <coiks/solution_set.hpp>

using namespace Eigen;
namespace pin = pinocchio;

class IkSolver
{
  public:
  IkSolver() {}

  virtual int IkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                      VectorXd &q_out)
  {
    return true;
  }

  virtual int IkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                      VectorXd &q_out, IKSolutionSet::Solution &solution)
  {
    return true;
  }

  virtual int         get_n_joints() { return 0; }
  virtual void        printStatistics() { return; }
  virtual std::string get_solver_name() { return ""; }

  private:
  int         n_joints_;
  std::string solver_name_;
};

#endif