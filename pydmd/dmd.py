# DMD algorithms by Andy Goldschmidt.
#
# TODO:
# - Should we create an ABC interface for DMD?
# - __init__.py and separate files
#
import numpy as np
from numpy.linalg import svd, pinv, eig
from scipy.linalg import expm

from .process import _threshold_svd, dag


class DMD:
    def __init__(self, X2, X1, ts, **kwargs):
        """ X2 = A X1

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2:
            X1:
            U:
            t0: Initial time.
            dt: Step size.
            orig_timesteps: Original times matching X1.
            A: Learned drift operator.
            Atilde: Projected A.
            eigs: Eigenvalues of Atilde.
            modes: DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.X2 = X2
        self.X1 = X1

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # I. Compute SVD
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            U, S, Vt = svd(self.X1, full_matrices=False)
        else:
            threshold_type = kwargs.get('threshold_type', 'percent')
            U, S, Vt = _threshold_svd(self.X1, threshold, threshold_type)

        # II: Compute operators: X2 = A X1 and Atilde = U*AU
        Atilde = dag(U) @ self.X2 @ dag(Vt) @ np.diag(1 / S)
        self.A = self.X2 @ dag(Vt) @ np.diag(1 / S) @ dag(U)

        # III. DMD Modes
        #       Atilde W = W Y (Eigendecomposition)
        self.eigs, W = eig(Atilde)

        # Two versions (eigenvectors of A)
        #       (i)  DMD_exact = X2 V S^-1 W
        #       (ii) DMD_proj = U W
        dmd_modes = kwargs.get('dmd_modes', 'exact')
        if dmd_modes == 'exact':
            self.modes = self.X2 @ dag(Vt) @ np.diag(1 / S) @ W
        elif dmd_modes == 'projected':
            self.modes = U @ W
        else:
            raise ValueError('In DMD initialization, unknown dmd_mode type.')

    @classmethod
    def from_full(cls, X, ts, **kwargs):
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        return cls(X2, X1, ts, **kwargs)

    def time_spectrum(self, ts, system='discrete'):
        """Returns a continuous approximation to the time dynamics of A.

        Note that A_dst = e^(A_cts dt). Suppose (operator, eigs) pairs are denoted (A_dst, Y) for the discrete case
        and (A_cts, Omega) for the continuous case. The eigenvalue correspondence is e^log(Y)/dt = Omega.

        Args:
            ts (:obj:`ndarray` of float): Times.
            system ({'continuous', 'discrete'}): default 'discrete'.

        Returns:
            :obj:`ndarray` of float: Evaluations of modes at ts.
        """
        if np.isscalar(ts):
            # Cast eigs to complex numbers for logarithm
            if system == 'discrete':
                omega = np.log(self.eigs + 0j) / self.dt
            elif system == 'continuous':
                omega = self.eigs + 0j
            else:
                raise ValueError('In time_spectrum, invalid system value.')
            return np.exp(omega * (ts - self.t0))
        else:
            return np.array([self.time_spectrum(it, system=system) for it in ts]).T

    def _predict(self, ts, x0, system):
        left = self.modes
        right = pinv(self.modes) @ x0
        if np.isscalar(ts):
            return left @ np.diag(self.time_spectrum(ts, system)) @ right
        else:
            return np.array([left @ np.diag(self.time_spectrum(it, system)) @ right for it in ts]).T

    def predict_dst(self, ts=None, x0=None):
        """Predict the future state using continuous approximation to the discrete A.

        Args:
            ts (:obj:`ndarray` of float): Array of time-steps to predict. default self.orig_timesteps.
            x0 (:obj:`ndarray` of float): The initial value. default self.x0.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        x0 = self.X1[:, 0] if x0 is None else x0
        ts = self.orig_timesteps if ts is None else ts
        return self._predict(ts, x0, 'discrete')

    def predict_cts(self, ts=None, x0=None):
        """Predict the future state using the continuous operator A.

        Args:
            ts (:obj:`ndarray` of float): Array of time-steps to predict. default self.orig_timesteps.
            x0 (:obj:`ndarray` of float): The initial value. default self.x0.

        Returns:
             :obj:`ndarray` of float: Predicted state for each control input.
        """
        x0 = self.X1[:, 0] if x0 is None else x0
        ts = self.orig_timesteps if ts is None else ts
        return self._predict(ts, x0, 'continuous')


class DMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """ X2 = A X1 + B U

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2:
            X1:
            U:
            Ups: augmented state U*X1.
            t0: Initial time.
            dt: Step size.
            orig_timesteps: Original times matching X1.
            A: Learned drift operator.
            Atilde: Projected A.
            B: Learned control operator.
            Btilde: Projected B.
            eigs: Eigenvalues of Atilde.
            modes: DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.X1 = X1
        self.X2 = X2
        self.U = U if U.shape[1] == self.X1.shape[1] else U[:, :-1]  # ONLY these 2 options
        Omega = np.vstack([self.X1, self.U])

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n, _ = self.X2.shape
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    @classmethod
    def from_full(cls, X, U, ts, **kwargs):
        X2 = X[:, 1:]
        X1 = X[:, :-1]
        return cls(X2, X1, U, ts, **kwargs)

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution X_2 = A X_1 + B u_1.

        Default behavior (control=None) is to use the original control. (If the underlying A is desired,
        format zeros_like u that runs for the desired time.)

        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (:obj:`ndarray` of float): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        U = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0
        res = [xt]
        for ut in U[:, :-1].T:
            xt_1 = self.A @ xt + self.B @ ut
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            X_dot = A X + B u
            x(t+dt) = e^{dt A}(x(t) + dt B u(t))

        Default behavior (control=None) is to use the original control. (If the underlying A is desired,
        format zeros_like u that runs for the desired time.) Be sure that dt matches the train dt if
        using delay embeddings.

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        U = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0
        res = [xt]
        for ut in U[:, :-1].T:
            xt_1 = expm(dt * self.A) @ (xt + dt * self.B @ ut)
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.U.shape[0], n_steps])


class biDMD:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """X2 = A X1 + U B X1

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2:
            X1:
            U:
            Ups: augmented state U*X1.
            t0: Initial time.
            dt: Step size.
            orig_timesteps: Original times matching X1.
            A: Learned drift operator.
            Atilde: Projected A.
            B: Learned nonlinear control operator.
            Btilde: projected B.
            eigs: Eigenvalues of Atilde.
            modes: DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.U = U
        self.X1 = X1
        self.X2 = X2

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the strucutre of the B operator to match our
        #   time-delay function.
        self.shift = kwargs.get('shift', 0)
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(self.shift + 1, -1, n_time),
                             self.X1.reshape(self.shift + 1, -1, n_time)
                             ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups])

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n, _ = self.X2.shape
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution:

            x_1 = A x_0 + B (u.x_0)
                = [A B] [x_0, u.x_0]^T

        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array
        res = [xt]
        for t in range(control.shape[1] - 1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift + 1, -1)
            _xt = xt.reshape(self.shift + 1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A @ xt + self.B @ ups_t
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            x_{t+1} = e^{A dt + u B dt } x_t

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0] // delay_dim
        measure_1_dim = self.X1.shape[0] // delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1] - 1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls)
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim),
                             control[:, t].reshape(delay_dim, control_dim)
                             ).reshape(to_dim, delay_dim * measure_1_dim)

            xt_1 = expm((self.A + _uBt) * dt) @ xt
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])


class biDMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """ X2 = A X1 + U B X1 + D U

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.
            
        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2:
            X1:
            U:
            Ups: augmented state U*X1.
            t0: Initial time.
            dt: Step size.
            orig_timesteps: Original times matching X1.
            A: Learned drift operator.
            Atilde: Projected A.
            B: Learned nonlinear control operator.
            Btilde: projected B.
            D: Learned control operator.
            eigs: Eigenvalues of Atilde.
            modes: DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.U = U
        self.X1 = X1
        self.X2 = X2

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)
        self.shift = kwargs.get('shift', 0)
        delay_dim = self.shift + 1

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the structure of the B operator to match our
        #   time-delay function.
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(delay_dim, -1, n_time),
                             self.X1.reshape(delay_dim, -1, n_time)
                             ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups, self.U])

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        c = self.U.shape[0] // delay_dim
        n = self.X1.shape[0]
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        # Omega = X + uX + u => dim'ns: n + c*n + c
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:(c + 1) * n, :])
        self.D = left @ dag(Ug[(c + 1) * n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.Dtilde = dag(U) @ self.D
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution,

            x_1 = A x_0 + B (u*x_0) + D u
                = [A B D] [x_0, u*x_0, u ]^T
        
        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array
        res = [xt]
        for t in range(control.shape[1] - 1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift + 1, -1)
            _xt = xt.reshape(self.shift + 1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A @ xt + self.B @ ups_t + self.D @ control[:, t]
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            x_{t+1} = e^{A dt + u B dt } (x_t + dt * D u_t}

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0] // delay_dim
        measure_1_dim = self.X1.shape[0] // delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1] - 1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls) 
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim),
                             control[:, t].reshape(delay_dim, control_dim)
                             ).reshape(to_dim, delay_dim * measure_1_dim)

            xt_1 = expm(dt * (self.A + _uBt)) @ (xt + dt * self.D @ control[:, t])
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])

