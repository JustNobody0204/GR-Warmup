# Copyright 2022 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMSProp optimizer."""

import flax
import jax
import jax.numpy as jnp
import numpy as np


class RMSPropOptimizer(flax.optim.OptimizerDef):
    """RMSProp optimizer with gradient clip."""

    @flax.struct.dataclass
    class HyperParams:
        learning_rate: np.ndarray
        beta2: np.ndarray
        eps: np.ndarray
        weight_decay: np.ndarray
        centered: bool

    @flax.struct.dataclass
    class State:
        v: np.ndarray
        mg: np.ndarray

    def __init__(self,
                learning_rate = None,
                beta2 = 0.9,
                eps = 1e-8,
                weight_decay = 0.0,
                centered = False):

        hyper_params = RMSPropOptimizer.HyperParams(learning_rate = learning_rate,
                                                beta2 = beta2,
                                                eps = eps,
                                                weight_decay = weight_decay,
                                                centered = centered)
        super().__init__(hyper_params)
    
    def init_param_state(self, param):
        mg = jnp.zeros_like(param) if self.hyper_params.centered else None
        return RMSPropOptimizer.State(jnp.zeros_like(param), mg)

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        weight_decay = hyper_params.weight_decay
        new_v = hyper_params.beta2 * state.v + (
            1.0 - hyper_params.beta2) * jnp.square(grad)
        if hyper_params.centered:
            new_mg = hyper_params.beta2 * state.mg + (1.0 - hyper_params.beta2) * grad
            maybe_centered_v = new_v - jnp.square(new_mg)
        else:
            new_mg = state.mg
            maybe_centered_v = new_v
        new_param = param - hyper_params.learning_rate * grad / ( 
            jnp.sqrt(maybe_centered_v) + hyper_params.eps)
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = RMSPropOptimizer.State(new_v, new_mg)
        return new_param, new_state
