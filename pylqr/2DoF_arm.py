"""
A Inverted Pendulum test for the iLQR implementation
"""
from __future__ import print_function

try:
    import jax.numpy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

from pylqr import PyLQR_iLQRSolver


class TwoR_Robot:
    def __init__(self, T=150):
        # parameters
        self.nDoF = 2
        self.dt = 0.01  # s
        """ arm model parameters """
        self.x0 = np.array([np.deg2rad(5), np.deg2rad(10), 0., 0.])  # initial state vector
        self.l = np.array([0.3, 0.33])  # m. link length
        self.m = np.array([1.4, 1.1])  # kg. link mass
        self.I = np.array([0.025, 0.045])  # kg m^2. MOI about the CG of each of the link, Izz
        self.s = np.array([0.11, 0.16])  # m. distance of CG from each of the joints

        """ pre-compute some constants """
        self.d1 = self.I[0] + self.I[1] + self.m[1] * self.l[0]**2
        self.d2 = self.m[1] * self.l[0] * self.s[1]

        self.dt_ = 0.01

        self.ilqr = None
        self.res = None
        self.T = T

        self.Q = 100 * np.eye(self.nDoF)
        self.R = .01 * np.eye(self.nDoF)

        # terminal Q to regularize final speed
        self.Q_T = .1 * np.eye(self.nDoF)
        self.target = np.array([0.4, 0.2])  # cartesian position of target. At the target, the robot has to stop and thus zero velocity
        self.target_thetas = self.inv_kin(self.target)  # provides the joint angles associated with cart. target position

    def plant_dyn(self, x, u, t, aux):
        q, dq = x[0:self.nDoF], x[self.nDoF:2*self.nDoF]

        M = np.array([[self.d1 + 2 * self.d2 * np.cos(q[1]), self.I[1] + self.d2 * np.cos(q[1])],
                      [self.I[1] + self.d2 * np.cos(q[1]), self.I[1]]])

        # centripital and Coriolis effects
        C = np.array([[-dq[1] * (2 * dq[0] + dq[1])],
                      [dq[0] ** 2]]) * self.d2 * np.sin(q[1])

        # joint friction
        B = np.array([[.05, .025],
                      [.025, .05]])

        # calculate forward dynamics
        kk = np.dot(B, dq).reshape(-1, 1)
        ddq = np.linalg.pinv(M) @ (u.reshape(-1, 1) - C - kk)

        # transfer to next time step
        new_dq = dq + self.dt_ * ddq.reshape(-1)
        new_q = q + self.dt_ * new_dq.reshape(-1)
        x_new = np.hstack((new_q, new_dq))
        return x_new

    """ 
    def plant_dyn_dx(self, x, u, t, aux):
        dfdx = np.array([
            [1, self.dt_],
            [-self.m_ * self.g_ * self.lc_ * np.cos(x[0]) * self.dt_ / self.I_, 1 - self.b_ * self.dt_ / self.I_]
        ])
        return dfdx

    def plant_dyn_du(self, x, u, t, aux):
        dfdu = np.array([
            [0],
            [self.dt_ / self.I_]
        ])
        return dfdu

    def cost_dx(self, x, u, t, aux):
        if t < self.T:
            dldx = np.array(
                [2 * (x[0] - np.pi) * self.Q, 0]
            ).T
        else:
            # terminal cost
            dldx = np.array(
                [2 * (x[0] - np.pi) * self.Q, 2 * x[1] * self.Q_T]
            ).T
        return dldx

    def cost_du(self, x, u, t, aux):
        dldu = np.array(
            [2 * u[0] * self.R]
        )
        return dldu

    def cost_dxx(self, x, u, t, aux):
        if t < self.T:
            dldxx = np.array([
                [2, 0],
                [0, 0]
            ]) * self.Q
        else:
            dldxx = np.array([
                [2 * self.Q, 0],
                [0, 2 * x[1] * self.Q_T]
            ])
        return dldxx

    def cost_duu(self, x, u, t, aux):
        dlduu = np.array([
            [2 * self.R]
        ])
        return dlduu

    def cost_dux(self, x, u, t, aux):
        dldux = np.array(
            [0, 0]
        )
        return dldux
    """

    def forward_kin(self, q):
        x1, y1 = self.l[0] * np.cos(q[0]), self.l[0] * np.sin(q[0])
        x2, y2 = self.l[1] * np.cos(q[0] + q[1]), self.l[1] * np.sin(q[0] + q[1])
        return x1, x2, y1, y2

    def inv_kin(self, target=None):
        if target is None:
            target = self.target
        a = target[0]**2 + target[1]**2 - self.l[0]**2 - self.l[1]**2
        b = 2 * self.l[0] * self.l[1]
        q2 = np.arccos(a/b)
        c = np.arctan2(target[1], target[0])
        q1 = c - np.arctan2(self.l[1] * np.sin(q2), (self.l[0] + self.l[1]*np.cos(q2)))
        return np.array([q1, q2])

    def instaneous_cost(self, x, u, t, aux):
        q, dq = x[0:self.nDoF], x[self.nDoF:2*self.nDoF]
        # x1, x2, y1, y2 = self.forward_kin(q)
        # eef_pos = np.array([x1+x2, y1+y2])
        # pos_err = eef_pos - self.target
        pos_err = q - self.target_thetas
        if t < self.T:
            return pos_err.T @ self.Q @ pos_err + u.T @ self.R @ u
        else:
            return pos_err.T @ self.Q @ pos_err + u.T @ self.R @ u + dq.T @ self.Q_T @ dq

    # grad_types = ['user', 'autograd', 'fd']
    def build_ilqr_problem(self, grad_type=0):
        if grad_type == 0:
            self.ilqr = PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost,
                                         use_autograd=False)
            # not use finite difference, assign the gradient functions
            self.ilqr.plant_dyn_dx = self.plant_dyn_dx
            self.ilqr.plant_dyn_du = self.plant_dyn_du
            self.ilqr.cost_dx = self.cost_dx
            self.ilqr.cost_du = self.cost_du
            self.ilqr.cost_dxx = self.cost_dxx
            self.ilqr.cost_duu = self.cost_duu
            self.ilqr.cost_dux = self.cost_dux
        elif grad_type == 1:
            self.ilqr = PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost,
                                         use_autograd=True)
        else:
            # finite difference
            self.ilqr = PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost,
                                         use_autograd=False)
        return

    def solve_ilqr_problem(self, x0=None, u_init=None, n_itrs=150, verbose=True):
        # prepare initial guess
        if u_init is None:
            u_init = np.ones((self.T, self.nDoF))
        if x0 is None:
            x0 = np.array([0.11, 0.12, 0.0, 0.0])

        if self.ilqr is not None:
            self.res = self.ilqr.ilqr_iterate(x0, u_init, n_itrs=n_itrs, tol=1e-6, verbose=verbose)
        return self.res

    def plot_ilqr_result(self):
        if self.res is not None:
            # draw cost evolution and phase chart
            fig = plt.figure(figsize=(16, 8), dpi=80)
            ax_cost = fig.add_subplot(121)
            n_itrs = len(self.res['J_hist'])
            ax_cost.plot(np.arange(n_itrs), self.res['J_hist'], 'r', linewidth=3.5)
            ax_cost.set_xlabel('Number of Iterations', fontsize=20)
            ax_cost.set_ylabel('Trajectory Cost')

            ax_phase = fig.add_subplot(122)
            theta = self.res['x_array_opt'][:, 0:self.nDoF]
            theta_dot = self.res['x_array_opt'][:, self.nDoF:self.nDoF*2]
            ax_phase.plot(theta, theta_dot, 'k', linewidth=3.5)
            ax_phase.set_xlabel('theta (rad)', fontsize=20)
            ax_phase.set_ylabel('theta_dot (rad/s)', fontsize=20)
            ax_phase.set_title('Phase Plot', fontsize=20)

            ax_phase.plot([theta[-1]], [theta_dot[-1]], 'b*', markersize=16)

            plt.figure()
            plt.grid()
            plt.plot(range(len(theta)), theta, label='angle')
            plt.legend()

            plt.pause(0.02)

            plt.figure()
            plt.grid()
            plt.plot(range(len(theta)), theta_dot, label='ang_velocity')
            plt.legend()

            plt.pause(0.02)

        return

    def animation(self):
        if self.res is not None:
            theta = self.res['x_array_opt'][:, 0:self.nDoF]
            plt.figure()
            for i in range(theta.shape[0]):
                plt.clf()
                plt.grid()
                plt.xlim([-.3, 0.7])
                plt.ylim([-0.3, 0.4])
                plt.plot(self.target[0], self.target[1], 'r*')
                x1, x2, y1, y2 = self.forward_kin(theta[i])
                plt.plot([0, x1], [0, y1], 'b')
                plt.plot([x1, x1 + x2], [y1, y1 + y2], 'b')
                plt.pause(0.02)


if __name__ == '__main__':
    problem = TwoR_Robot()
    x0 = np.array([0.02, 0.11, 0, 0])  # angular positions and angular velocities

    # problem.build_ilqr_problem(grad_type=0) #try real gradients/hessians
    # problem.solve_ilqr_problem(x0)
    # problem.plot_ilqr_result()

    # res_dict = {
    #     'J_hist': np.array(J_hist),
    #     'x_array_opt': np.array(x_array),
    #     'u_array_opt': np.array(u_array),
    #     'k_array_opt': np.array(k_array),
    #     'K_array_opt': np.array(K_array)
    # }
    problem.build_ilqr_problem(grad_type=1)  # try autograd
    result_dict = problem.solve_ilqr_problem(x0)
    X = result_dict['x_array_opt']
    problem.animation()
    problem.plot_ilqr_result()

    plt.show()
    # problem.build_ilqr_problem(grad_type=2)  # try finite difference
    # problem.solve_ilqr_problem(x0)
    # problem.plot_ilqr_result()