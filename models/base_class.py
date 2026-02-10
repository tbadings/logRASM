from functools import partial

import jax
import numpy as np
from gymnasium import spaces
from jax import jit
from scipy.stats import triang


class BaseEnvironment:

    def __init__(self):
        # Define vectorized functions
        self.vreset = jax.jit(jax.vmap(self.reset_jax, in_axes=0, out_axes=0))
        self.vstep = jax.jit(jax.vmap(self.step_train, in_axes=0, out_axes=0))
        self.vstep_base = jax.jit(jax.vmap(self.step_base, in_axes=0, out_axes=0))
        self.vstep_noise_batch = jax.jit(jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0))

        self.state_dim = len(self.state_space.low)

        # Initialize as gym environment
        self.initialize_gym_env()

        if hasattr(self, 'lipschitz_f_l1_A'):
            print(f'- Lipschitz constant of dynamics w.r.t. state variables: {np.round(self.lipschitz_f_l1_A, 3)}')
        if hasattr(self, 'lipschitz_f_l1_B'):
            print(f'- Lipschitz constant of dynamics w.r.t. input variables: {np.round(self.lipschitz_f_l1_B, 3)}')
        print(f'- Overall Lipschitz constant of dynamics: {np.round(self.lipschitz_f_l1, 3)}')

    def initialize_gym_env(self):
        # Initialize state
        self.state = None
        self.steps_beyond_terminated = None

        # Observation space is only used in the gym version of the environment
        self.observation_space = spaces.Box(low=self.reset_space.low, high=self.reset_space.high, dtype=np.float32)

    def set_linear_lipschitz(self):
        self.lipschitz_f_l1 = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=0)))
        self.lipschitz_f_linfty = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=1)))

        self.lipschitz_f_l1_A = float(np.max(np.sum(self.A, axis=0)))
        self.lipschitz_f_linfty_A = float(np.max(np.sum(self.A, axis=1)))
        self.lipschitz_f_l1_B = float(np.max(np.sum(self.B, axis=0)))
        self.lipschitz_f_linfty_B = float(np.max(np.sum(self.B, axis=1)))

    @partial(jit, static_argnums=(0,))
    def step_noise_key(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(self.noise_dim,))

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        return state, key

    def _maybe_reset(self, state, key, steps_since_reset, done):
        return jax.lax.cond(done, self._reset, lambda key: (state, key, steps_since_reset), key)

    def _reset(self, key):
        high = self.reset_space.high
        low = self.reset_space.low

        key, subkey = jax.random.split(key)
        new_state = jax.random.uniform(subkey, minval=low,
                                       maxval=high, shape=(self.state_dim,))

        steps_since_reset = 0

        return new_state, key, steps_since_reset

    def reset(self, seed=None, options=None):
        ''' Reset function for pytorch / gymnasium environment '''

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Sample state uniformly from observation space
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.last_u = None

        return self.state, {}

    @partial(jit, static_argnums=(0,))
    def reset_jax(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset

    def integrate_noise_triangular(self, w_lb, w_ub):
        ''' Integrate noise distribution in the box [w_lb, w_ub]. '''

        # For triangular distribution, integration is simple, because we can integrate each dimension individually and
        # multiply the resulting probabilities
        probs = np.ones(len(w_lb))

        # Triangular cdf increases from loc to (loc + c*scale), and decreases from (loc+c*scale) to (loc + scale)
        # So, 0 <= c <= 1.
        loc = self.noise_space.low
        c = 0.5  # Noise distribution is zero-centered, so c=0.5 by default
        scale = self.noise_space.high - self.noise_space.low

        for d in range(self.noise_space.shape[0]):
            probs *= triang.cdf(w_ub[:, d], c, loc=loc[d], scale=scale[d]) - triang.cdf(w_lb[:, d], c, loc=loc[d],
                                                                                        scale=scale[d])

        # In this case, the noise integration is exact, but we still return an upper and lower bound
        prob_ub = probs
        prob_lb = probs

        return prob_lb, prob_ub
