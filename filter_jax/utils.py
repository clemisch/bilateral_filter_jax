import jax
from functools import partial



def multi_vmap(fun, in_axes, out_axes):
    fun_mapped = fun
    for i, o in zip(in_axes, out_axes):
        fun_mapped = jax.vmap(fun_mapped, i, o)
    return fun_mapped


@partial(jax.jit, static_argnames="fun")
def get_patch_nd(indices, img, kernel, fun):
    # TODO: can this be done more efficiently?
    kernel_axes = tuple([
        jnp.arange(n) - n//2 + i 
        for i, n in zip(indices, kernel.shape)
    ])
    kernel_indices = jnp.ix_(*kernel_axes)
    patch = img.at[tuple(kernel_indices)].get(
        # mode="drop",
        # indices_are_sorted=True,
        # unique_indices=True
    )
    val = fun(patch, kernel)

    return val
