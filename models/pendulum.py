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
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Pendulum(BaseEnvironment, gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):

        self.variable_names = ['position', 'velocity']

        self.max_torque = np.array([1])

        # Pendulum parameters
        self.delta = 0.05
        self.G = 10
        self.m = 0.15
        self.l = 0.5
        self.b = 0.1
        self.max_speed = 5

        self.state_dim = 2
        self.plot_dim = [0, 1]

        self.lipschitz_f_l1_A = max(1 + self.delta * (self.delta + 1) * 1.5 * self.G / (2 * self.l),
                                    (1 + self.delta) * (1 - self.b))  # 1.7875
        self.lipschitz_f_l1_B = 6.0 / (self.m * self.l ** 2) * self.delta * (self.delta + 1)  # 8.4
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Set observation / state space
        high = np.array([0.7, 0.7], dtype=np.float32)
        self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        if args.deterministic:
            high = np.array([0, 0], dtype=np.float32)
        else:
            high = np.array([1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = 2

        # Set target set
        self.target_space = RectangularSet(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)

        self.init_space = RectangularSet(low=np.array([-0.3, -0.3]), high=np.array([0.3, 0.3]), dtype=np.float32)

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-0.7, -0.7]), high=np.array([-0.6, 0]), dtype=np.float32),
            RectangularSet(low=np.array([0.6, 0]), high=np.array([0.7, 0.7]), dtype=np.float32)
        ])

        self.init_unsafe_dist = 0.3

        self.num_steps_until_reset = 1000

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(Pendulum, self).__init__()

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

        u = 2 * np.clip(u, -self.max_torque, self.max_torque)
        w = self.sample_noise_numpy()

        x1 = (1 - self.b) * self.state[1] + (
                -1.5 * self.G * np.sin(self.state[0] + np.pi) / (2 * self.l) +
                3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * w[0]
        x1 = np.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = self.state[0] + self.delta * x1 + 0.01 * w[1]

        # Clip state
        self.state = np.clip(np.array([x0, x1]), self.state_space.low, self.state_space.high)

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 5
        elif goal_reached:
            costs = -5
        else:
            costs = -1 + np.sqrt(angle_normalize(self.state[0]) ** 2 + 0.1 * self.state[1] ** 2)

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = 2 * jnp.clip(u, -self.max_torque, self.max_torque)

        x1 = (1 - self.b) * state[1] + (
                -1.5 * self.G * jnp.sin(state[0] + jnp.pi) / (2 * self.l) +
                3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * w[0]
        x1 = jnp.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = state[0] + self.delta * x1 + 0.01 * w[1]

        # Lower bound state
        state = jnp.clip(jnp.array([x0, x1]), self.state_space.low, self.state_space.high)

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
        noise = self.sample_noise(subkey, size=(2,))

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]
        costs = angle_normalize(state[0]) ** 2 + 0.1 * state[1] ** 2

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = False  # fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}
