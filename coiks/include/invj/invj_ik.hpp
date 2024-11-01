#ifndef INVJIK_HPP
#define INVJIK_HPP

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// INVJ Interfaces
#include <invj/invj_solvers.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS;

  template <typename T> class INVJ_IkSolver
  {
    friend class coiks::COIKS;

public:
    INVJ_IkSolver(const pin::Model &_model, const pin::FrameIndex &_Fid,
                  SolverOptions _solver_opts, const double _max_time,
                  const double _max_error, const int _max_iter = 1000,
                  const double _dt = 1, const int _id = 0)
    {
      invj_solver_.reset(new T(_model, _Fid, _solver_opts, _max_time,
                               _max_error, _max_iter, _dt, _id));
    }
    ~INVJ_IkSolver(){};

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_des, VectorXd &q_out)
    {
      int res;
      res = invj_solver_->IkSolve(q_init, x_des, q_out);
      return res;
    }
    int IkSolveSet(const VectorXd q_init, const pin::SE3 &x_des,
                   VectorXd &q_out, IKSolutionSet::Solution &solution)
    {
      int res;
      res = invj_solver_->IkSolveSet(q_init, x_des, q_out, solution);
      return res;
    }

    MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon)
    {
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
    MatrixXd weightedPseudoInverse(const Eigen::MatrixXd &a,
                                   const Eigen::VectorXd  w)
    {
      int lenght = w.size();

      Eigen::DiagonalMatrix<double, Eigen::Dynamic> Winv(lenght);
      Winv = w.asDiagonal().inverse(); // diag(1./w)

      Eigen::MatrixXd tmp(lenght, lenght);

      //(a*Winv*a.transpose())
      tmp = pseudoInverse(a * Winv * a.transpose(), 10E-10);

      return Winv * a.transpose() * tmp;
    }

    void abort();
    void reset();
    void set_max_time(double _max_time);
    void set_max_error(double _max_error);

private:
    std::unique_ptr<T> invj_solver_;
  };

  template <typename T> inline void INVJ_IkSolver<T>::abort()
  {
    invj_solver_->abort();
  }
  template <typename T> inline void INVJ_IkSolver<T>::reset()
  {
    invj_solver_->reset();
  }
  template <typename T>
  inline void INVJ_IkSolver<T>::set_max_time(double _max_time)
  {
    invj_solver_->set_max_time(_max_time);
  }
  template <typename T>
  inline void INVJ_IkSolver<T>::set_max_error(double _max_error)
  {
    invj_solver_->set_max_error(_max_error);
  }

} // namespace coiks

#endif