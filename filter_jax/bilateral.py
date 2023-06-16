import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from .utils import multi_vmap


@partial(jax.jit, static_argnames="potential")
def _get_patch_3d(iz, iy, ix, img, kernel, potential):
    nz, ny, nx = kernel.shape
    indices_z = jnp.arange(nz) - nz//2 + iz
    indices_y = jnp.arange(ny) - ny//2 + iy
    indices_x = jnp.arange(nx) - nx//2 + ix

    # TODO (CS): use flags for `.get()` to improve performance
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    kernel_indices = jnp.ix_(indices_z, indices_y, indices_x)
    patch = img.at[kernel_indices].get()

    w_patch = potential(patch)
    w_tot = kernel * w_patch
    out = jnp.sum(patch * w_tot) / jnp.sum(w_tot)

    return out



@partial(jax.jit, static_argnames="potential")
def bilateral_filter(img, kernel, potential):
    assert img.ndim == 3
    assert kernel.ndim == 3

    dimz, dimy, dimx = img.shape
    indices_z = jnp.arange(0, dimz)
    indices_y = jnp.arange(0, dimy)
    indices_x = jnp.arange(0, dimx)

    # vmap y and x
    _get_slice = multi_vmap(
        _get_patch_3d,
        (
            (None, 0   , None, None, None, None),
            (None, None, 0   , None, None, None)
        ),
        (0, 1)
    )

    # map z (to avoid memory overload)
    img_out = jax.lax.map(
        lambda iz: _get_slice(iz, indices_y, indices_x, img, kernel, potential),
        indices_z,
    )

    return img_out



def _gauss(x, mu, sigma):
    return jnp.exp(-0.5 * jnp.square(x - mu) / jnp.square(sigma))


@jax.jit
def bilateral_gauss(img, kernel, sigma):
    sl_center = tuple([sh//2 for sh in kernel.shape])
    def potential(patch):
        mu = patch[sl_center]
        return _gauss(patch, mu, sigma)

    out = bilateral_filter(img, kernel, potential)

    return out
