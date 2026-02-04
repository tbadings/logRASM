from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


def angle_normalize(x):
    return x % (2 * np.pi)


class PlanarRobot(BaseEnvironment, gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):

        self.variable_names = ['x', 'y', 'ang.vel.']

        self.min_torque = np.array([-1, -1], dtype=np.float32)
        self.max_torque = np.array([1, 1], dtype=np.float32)

        self.delta = 0.2

        self.state_dim = 3
        self.plot_dim = [0, 1]

        # Entries of Jacobian; row 1
        self.dxdx = 1
        self.dxdy = 0
        self.dxdv = self.delta
        self.dxdu1 = np.pi * self.delta

        # Entries of Jacobian; row 2
        self.dydx = 0
        self.dydy = 1
        self.dydv = self.delta
        self.dydu1 = np.pi * self.delta

        # Entries of Jacobian; row 3
        self.dvdx = 0
        self.dvdy = 0
        self.dvdv = 1
        self.dvdu0 = 2 * self.delta
        
        self.dxdu0 = 2 * self.delta ** 2
        self.dydu0 = 2 * self.delta ** 2

        self.lipschitz_f_l1_A = max(np.abs(self.dxdx) + np.abs(self.dydx) + np.abs(self.dvdx),
                                    np.abs(self.dxdy) + np.abs(self.dydy) + np.abs(self.dvdy),
                                    np.abs(self.dxdv) + np.abs(self.dydv) + np.abs(self.dvdv))
        self.lipschitz_f_l1_B = max(np.abs(self.dxdu1) + np.abs(self.dydu1),
                                    np.abs(self.dxdu0) + np.abs(self.dydu0) + np.abs(self.dvdu0))
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=self.min_torque, high=self.max_torque, shape=(len(self.min_torque),), dtype=np.float32
        )

        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        self.v_min = -1
        self.v_max = 1

        # Set target set
        self.target_space = RectangularSet(low=np.array([-1, 0.6, self.v_min]),
                                           high=np.array([-0.6, 1, self.v_max]),
                                           fix_dimensions=[2], dtype=np.float32)

        self.init_space = RectangularSet(low=np.array([0.4, -0.8, -0.1]),
                                         high=np.array([0.6, -0.6, 0.1]),
                                         fix_dimensions=[2], dtype=np.float32)

        self.init_unsafe_dist = 0.4 + 0.2

        # Set observation / state space
        low = np.array([self.x_min, self.y_min, self.v_min], dtype=np.float32)
        high = np.array([self.x_max, self.y_max, self.v_max], dtype=np.float32)
        self.state_space = RectangularSet(low=low, high=high, fix_dimensions=[2], dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        if args.deterministic:
            high = np.array([0, 0], dtype=np.float32)
        else:
            high = np.array([1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = 2

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-1, -1, self.v_min]),
                           high=np.array([-0.8, 0, self.v_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([-0.1, 0.8, self.v_min]),
                           high=np.array([1, 1, self.v_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([0.8, 0, self.v_min]),
                           high=np.array([1, 0.8, self.v_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([-0.4, -0.4, self.v_min]),
                           high=np.array([0.0, 0.1, self.v_max]), fix_dimensions=[2],
                           dtype=np.float32),
        ])

        self.num_steps_until_reset = 100

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(PlanarRobot, self).__init__()

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

        u = jnp.clip(u, self.min_torque, self.max_torque)
        w = self.sample_noise_numpy()

        v = self.state[2] + self.delta * 2 * u[0]
        x = self.state[0] + self.delta * v * np.cos(np.pi * u[1]) + 0.01 * w[0]
        y = self.state[1] + self.delta * v * np.sin(np.pi * u[1]) + 0.01 * w[1]

        # Lower/upper bound state
        self.state = jnp.array([x, y, v])
        self.state = jnp.clip(self.state, self.state_space.low, self.state_space.high)
        self.last_u = u  # for rendering

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True) +
            self.state_space.not_contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 5
        elif goal_reached:
            costs = -5
        else:
            costs = -1 + np.sqrt((self.state[0] - self.target_space.center[0]) ** 2 +
                                 (self.state[1] - self.target_space.center[1]) ** 2)

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, self.min_torque, self.max_torque)

        v = state[2] + self.delta * 2 * u[0]
        x = state[0] + self.delta * v * jnp.cos(jnp.pi * u[1]) + 0.01 * w[0]
        y = state[1] + self.delta * v * jnp.sin(jnp.pi * u[1]) + 0.01 * w[1]

        # Lower bound state
        state = jnp.array([x, y, v])
        state = jnp.clip(state, self.state_space.low, self.state_space.high)

        return state

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values.
        Propagate state under lower/upper bound of the noise (note: this works because the noise is additive) '''

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

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]

        costs = 10 * jnp.sqrt(((state[0] - self.target_space.center[0]) ** 2) +
                              ((state[1] - self.target_space.center[1]) ** 2)) - 10 * goal_reached

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}
