import json
import numpy as np


class StarTrackerObserver:
    """
    Nonlinear observer for spacecraft attitude and angular velocity using
    star-tracker image measurements.

    Estimated state:
        xhat = [psi, theta, phi, w_x, w_y, w_z]
    """

    def __init__(
        self,
        stars_filename="urdf/stars.json",
        dt=0.04,
        scope_radius=(0.8 / 2.1),
        gn_damping=1e-3,
        rate_filter=0.2,
        max_gn_iters=3,
    ):
        self.dt = float(dt)
        self.scope_radius = float(scope_radius)
        self.gn_damping = float(gn_damping)
        self.rate_filter = float(rate_filter)
        self.max_gn_iters = int(max_gn_iters)

        with open(stars_filename, "r") as f:
            stars = json.load(f)

        self.star_dirs = []
        for s in stars:
            a = s["alpha"]
            d = s["delta"]
            self.star_dirs.append(
                np.array(
                    [
                        np.cos(a) * np.cos(d),
                        np.sin(a) * np.cos(d),
                        np.sin(d),
                    ],
                    dtype=float,
                )
            )

        self.reset()

    def reset(self):
        self.eta_hat = np.zeros(3, dtype=float)  # [psi, theta, phi]
        self.w_hat = np.zeros(3, dtype=float)    # [w_x, w_y, w_z]
        self.prev_eta = self.eta_hat.copy()
        self.xhat = np.zeros(6, dtype=float)

    def _R_body_in_space(self, eta):
        psi, theta, phi = eta

        cpsi, spsi = np.cos(psi), np.sin(psi)
        cth, sth = np.cos(theta), np.sin(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)

        Rz = np.array(
            [
                [cpsi, -spsi, 0.0],
                [spsi, cpsi, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        Ry = np.array(
            [
                [cth, 0.0, sth],
                [0.0, 1.0, 0.0],
                [-sth, 0.0, cth],
            ],
            dtype=float,
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cphi, -sphi],
                [0.0, sphi, cphi],
            ],
            dtype=float,
        )
        return Rz @ Ry @ Rx

    def _predict_measurement(self, eta):
        R = self._R_body_in_space(eta)
        Rt = R.T
        y = []
        for p_space in self.star_dirs:
            p_body = Rt @ p_space
            x_b = p_body[0]
            if np.abs(x_b) < 1e-6:
                x_b = np.sign(x_b) * 1e-6 if x_b != 0.0 else 1e-6

            y_star = (p_body[1] / x_b) / self.scope_radius
            z_star = (p_body[2] / x_b) / self.scope_radius
            y.extend([y_star, z_star])

        return np.array(y, dtype=float)

    def _measurement_jacobian_fd(self, eta, eps=1e-6):
        y0 = self._predict_measurement(eta)
        H = np.zeros((y0.size, 3), dtype=float)
        for i in range(3):
            de = np.zeros(3, dtype=float)
            de[i] = eps
            yp = self._predict_measurement(eta + de)
            ym = self._predict_measurement(eta - de)
            H[:, i] = (yp - ym) / (2.0 * eps)
        return H

    def _eta_dot_from_w(self, eta, w):
        _, theta, phi = eta
        wx, wy, wz = w

        cth = np.cos(theta)
        cth = np.clip(cth, 1e-4, None)

        psi_dot = (wy * np.sin(phi) + wz * np.cos(phi)) / cth
        theta_dot = wy * np.cos(phi) - wz * np.sin(phi)
        phi_dot = wx + wy * np.sin(phi) * np.tan(theta) + wz * np.cos(phi) * np.tan(theta)
        return np.array([psi_dot, theta_dot, phi_dot], dtype=float)

    def _w_from_eta_dot(self, eta, eta_dot):
        _, theta, phi = eta

        A = np.array(
            [
                [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
                [0.0, np.cos(phi), -np.sin(phi)],
                [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            ],
            dtype=float,
        )
        w, *_ = np.linalg.lstsq(A, eta_dot, rcond=None)
        return w

    def predict_measurement(self, eta=None):
        if eta is None:
            eta = self.eta_hat
        return self._predict_measurement(np.asarray(eta, dtype=float))

    def update(self, star_measurements):
        star_measurements = np.asarray(star_measurements, dtype=float)

        # 1) Predict attitude from previous estimate.
        eta_pred = self.eta_hat + self.dt * self._eta_dot_from_w(self.eta_hat, self.w_hat)

        # 2) Correct attitude with damped Gauss-Newton on image residual.
        eta = eta_pred.copy()
        for _ in range(self.max_gn_iters):
            yhat = self._predict_measurement(eta)
            residual = star_measurements - yhat
            H = self._measurement_jacobian_fd(eta)

            lhs = H.T @ H + self.gn_damping * np.eye(3)
            rhs = H.T @ residual
            deta = np.linalg.solve(lhs, rhs)
            deta = np.clip(deta, -0.2, 0.2)

            eta = eta + deta
            if np.linalg.norm(deta) < 1e-6:
                break

        # 3) Estimate angular rates from attitude increments.
        eta_dot_meas = (eta - self.prev_eta) / self.dt
        w_meas = self._w_from_eta_dot(eta, eta_dot_meas)
        self.w_hat = (1.0 - self.rate_filter) * self.w_hat + self.rate_filter * w_meas

        # 4) Save state estimate.
        self.prev_eta = eta.copy()
        self.eta_hat = eta
        self.xhat = np.concatenate((self.eta_hat, self.w_hat))
        return self.xhat.copy()
