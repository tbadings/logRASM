from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces, logger
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


class TripleIntegrator(BaseEnvironment, gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):

        self.variable_names = ['absement', 'position', 'velocity']

        self.max_torque = np.array([1])

        self.A = np.array([
            [1, 0.045, 0],
            [0, 1, 0.045],
            [0, 0, 0.9],
        ])
        self.state_dim = len(self.A)
        self.plot_dim = [0, 1]
        self.B = np.array([
            [0.35],
            [0.45],
            [0.5]
        ])
        self.W = np.diag([0.01, 0.01, 0.005])

        # Lipschitz coefficient of linear dynamical system is maximum sum of columns in A and B matrix.
        self.lipschitz_f_l1 = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=0)))
        self.lipschitz_f_linfty = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=1)))

        self.lipschitz_f_l1_A = float(np.max(np.sum(self.A, axis=0)))
        self.lipschitz_f_linfty_A = float(np.max(np.sum(self.A, axis=1)))
        self.lipschitz_f_l1_B = float(np.max(np.sum(self.B, axis=0)))
        self.lipschitz_f_linfty_B = float(np.max(np.sum(self.B, axis=1)))

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(len(self.max_torque),), dtype=np.float32
        )

        # Set observation / state space
        high = np.array([1, 1, 1], dtype=np.float32)
        self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        if args.deterministic:
            high = np.array([0, 0, 0], dtype=np.float32)
        else:
            high = np.array([1, 1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = len(high)

        # Set target set
        self.target_space = RectangularSet(low=np.array([-0.2, -0.2, -0.2]), high=np.array([0.2, 0.2, 0.2]),
                                           dtype=np.float32)

        self.init_space = MultiRectangularSet([
            RectangularSet(low=np.array([-0.25, -0.25, -0.1]), high=np.array([-0.2, -0.2, 0.1]), dtype=np.float32),
            RectangularSet(low=np.array([0.2, 0.2, -0.1]), high=np.array([0.25, 0.25, 0.1]), dtype=np.float32)
        ])

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-1, -1, -1]), high=np.array([-0.9, -0.9, 0]), dtype=np.float32),
            RectangularSet(low=np.array([0.9, 0.9, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        ])

        self.init_unsafe_dist = 0.65 + 0.65

        self.num_steps_until_reset = 100

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.jit(jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0)))

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(TripleIntegrator, self).__init__()

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(self.noise_dim), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(self.noise_dim))

    def sample_noise_numpy(self, size=None):
        return np.random.triangular(self.noise_space.low * np.ones(self.noise_dim),
                                    np.zeros(self.noise_dim),
                                    self.noise_space.high * np.ones(self.noise_dim),
                                    size)

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with StableBaselines3).
        '''

        assert self.state is not None, "Call reset before using step method."

        u = np.clip(u, -self.max_torque, self.max_torque)
        w = self.sample_noise_numpy()
        self.state = self.A @ self.state + self.B @ u + self.W @ w
        self.last_u = u  # for rendering

        terminated = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True) +
            self.state_space.not_contains(np.array([self.state]), return_indices=True)
        )

        if not terminated:
            costs = -1 + (self.state[0] ** 2) + (self.state[1] ** 2) + (self.state[2] ** 2)
        elif self.steps_beyond_terminated is None:
            # Just terminated, so incur big penalty!
            self.steps_beyond_terminated = 0
            costs = 100
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            costs = 0.0

        return np.array(self.state, dtype=np.float32), -costs, False, False, {}

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, -self.max_torque, self.max_torque)
        state = jnp.matmul(self.A, state) + jnp.matmul(self.B, u) + jnp.matmul(self.W, w)

        return state

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values '''

        # Propogate dynamics for both the lower bound and upper bound of the noise
        # (note: this works because the noise is additive)
        state_lb = self.step_base(state, u, w_lb)
        state_ub = self.step_base(state, u, w_ub)

        # Compute the mean and the epsilon (difference between mean and ub/lb)
        state_mean = (state_ub + state_lb) / 2
        epsilon = (state_ub - state_lb) / 2

        return state_mean, epsilon

    def integrate_noise(self, w_lb, w_ub):
        prob_lb, prob_ub = self.integrate_noise_triangular(w_lb, w_ub)

        return prob_lb, prob_ub

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(self.noise_dim,))

        costs = -1 + (state[0] ** 2) + (state[1] ** 2) + (state[2] ** 2)

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = False
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}
