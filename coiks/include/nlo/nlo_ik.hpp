#ifndef NLOIK_HPP
#define NLOIK_HPP

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// NLO Interfaces
#include <nlo/nlo_solvers.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS;

  template <typename T> class NLO_IkSolver
  {
    friend class coiks::COIKS;

public:
    NLO_IkSolver(const pin::Model &_model, const pin::FrameIndex &_Fid,
                 SolverOptions _solver_opts, const double _max_time,
                 const double _max_error, const int _max_iter = 1e5,
                 const double _dt = 1)
    {
      nlo_solver_.reset(new T(_model, _Fid, _solver_opts, _max_time, _max_error,
                              _max_iter, _dt));
    }
    ~NLO_IkSolver(){};
    int IkSolve(const VectorXd q_init, const pin::SE3 &x_des, VectorXd &q_out)
    {
      int res;
      res = nlo_solver_->IkSolve(q_init, x_des, q_out);
      return res;
    }

    void abort();
    void reset();
    void set_max_time(double _max_time);
    void set_max_error(double _max_error);

private:
    std::unique_ptr<T> nlo_solver_;
  };

  template <typename T> inline void NLO_IkSolver<T>::abort()
  {
    nlo_solver_->abort();
  }
  template <typename T> inline void NLO_IkSolver<T>::reset()
  {
    nlo_solver_->reset();
  }
  template <typename T>
  inline void NLO_IkSolver<T>::set_max_time(double _max_time)
  {
    nlo_solver_->set_max_time(_max_time);
  }
  template <typename T>
  inline void NLO_IkSolver<T>::set_max_error(double _max_error)
  {
    nlo_solver_->set_max_error(_max_error);
  }

} // namespace coiks

#endif
