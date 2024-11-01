#ifndef QPIK_HPP
#define QPIK_HPP

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// QP Interfaces
#include <qp/qp_solvers.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS;

  template <typename T> class QP_IkSolver
  {
    friend class coiks::COIKS;

public:
    QP_IkSolver(const pin::Model &_model, const pin::FrameIndex &_Fid,
                SolverOptions _solver_opts, const double _max_time,
                const double _max_error, const int _max_iter = 1e4,
                const double _dt = 1)
    {
      qp_solver_.reset(new T(_model, _Fid, _solver_opts, _max_time, _max_error,
                             _max_iter, _dt));
    }
    ~QP_IkSolver(){
        // std::cout << "Closing QP_IK" << std::endl;
        // sleep(5);
    };

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_des, VectorXd &q_sol)
    {
      int res;
      res = qp_solver_->IkSolve(q_init, x_des, q_sol);
      return res;
    }

    void abort();
    void reset();
    void set_max_time(double _max_time);
    void set_max_error(double _max_error);

private:
    std::unique_ptr<T> qp_solver_;
  };

  template <typename T> inline void QP_IkSolver<T>::abort()
  {
    qp_solver_->abort();
  }
  template <typename T> inline void QP_IkSolver<T>::reset()
  {
    qp_solver_->reset();
  }
  template <typename T>
  inline void QP_IkSolver<T>::set_max_time(double _max_time)
  {
    qp_solver_->set_max_time(_max_time);
  }
  template <typename T>
  inline void QP_IkSolver<T>::set_max_error(double _max_error)
  {
    qp_solver_->set_max_error(_max_error);
  }

} // namespace coiks

#endif
