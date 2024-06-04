from typing import Callable, Optional

import jax

from optax._src import base
from optax._src import utils


def lex_wrap(fn: Callable, chained_algs: base.GradientTransformationExtraArgs) -> base.GradientTransformationExtraArgs:

  value_and_grad = utils.value_and_grad_from_state(fn)

  def init_fn(params):
    return chained_algs.init(params)
  
  def update_fn(_, state, params, l,name: Optional[str] = None):

    if name == 'scale_by_backtracking_linesearch':
      value, grad = value_and_grad(params, state=state)
      grad = -grad
      updates, state = chained_algs.update(grad, state, params, value=value, grad=grad, value_fn=fn)

    elif name == 'sgd' or name == 'adam':
      grad = jax.grad(fn)(params)
      updates, state = chained_algs.update(grad, state, params)
    else:
      raise(Exception('Current layout is temporary, only the supported solvers are provided'))

    return updates, state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
  