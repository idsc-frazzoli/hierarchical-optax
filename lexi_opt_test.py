import time
import optax
import jax
import jax.numpy as jnp
import pytest

from optax._src.base import EmptyState

# the objective function
def fn(params): return jnp.sum(params ** 2)

# the rules
def r1(params): return jnp.max(jnp.array([0, -(params[0]-5)]))
def r2(params): return jnp.max(jnp.array([0, -(params[1]-4)]))


rate_decay = optax.exponential_decay(init_value=1, transition_steps=10, decay_rate=0.8)



@pytest.mark.parametrize(
    "method_fn, method_name, itr, expected",
    [
        (optax.sgd(learning_rate=rate_decay), 'sgd', 300, 41),
        (optax.adam(learning_rate=rate_decay), 'adam', 300, 3363),
        (optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, decrease_factor=0.5,store_grad=True), 'scale_by_backtracking_linesearch', 20, 34.67),
    ]
)
def test(method_fn, method_name, itr, expected):
    solver = optax.lex(
        optax.chain(method_fn),
        rules=(r1, r2), 
        fun=fn,
        l0=10
    )

    params = jnp.array([100.,2.])

    opt_state = solver.init(params)

    l = 10
    for _ in range(itr):
        updates, opt_state = solver.update((EmptyState(),), opt_state,params,l=l, name=method_name)
        params = optax.apply_updates(params, updates)
    assert jnp.isclose(fn(params), expected, atol=1)


if __name__=="__main__":
    test(optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, decrease_factor=0.5,store_grad=True), 'scale_by_backtracking_linesearch', 20, 34.67)