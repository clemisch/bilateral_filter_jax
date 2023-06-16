import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from .utils import multi_vmap


@partial(jax.jit, static_argnames="potential")
def _get_patch_3d(iz, iy, ix, img1, img2, kernel, potential):
    # get patch for each image
    nz, ny, nx = kernel.shape
    indices_z = jnp.arange(nz) - nz//2 + iz
    indices_y = jnp.arange(ny) - ny//2 + iy
    indices_x = jnp.arange(nx) - nx//2 + ix

    kernel_indices = jnp.ix_(indices_z, indices_y, indices_x)

    # TODO (CS): use flags for `.get()` to improve performance
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    patch1 = img1.at[kernel_indices].get()
    patch2 = img2.at[kernel_indices].get()

    # compute value weights from image patch 1
    w_patch = potential(patch1)
    w_tot = kernel * w_patch

    # filter image patch 2
    out = jnp.sum(patch2 * w_tot) / jnp.sum(w_tot)

    return out



@partial(jax.jit, static_argnames="potential")
def bilateral_filter_cc(img1, img2, kernel, potential):
    assert img1.ndim == 3
    assert img1.shape == img2.shape
    assert kernel.ndim == 3

    dimz, dimy, dimx = img1.shape
    indices_z = jnp.arange(0, dimz)
    indices_y = jnp.arange(0, dimy)
    indices_x = jnp.arange(0, dimx)

    # vmap y and x
    _get_slice = multi_vmap(
        _get_patch_3d,
        (
            (None, 0   , None, None, None, None, None),
            (None, None, 0   , None, None, None, None)
        ),
        (0, 1)
    )

    # map z (to avoid memory overload)
    img_out = jax.lax.map(
        lambda iz: _get_slice(iz, indices_y, indices_x, img1, img2, kernel, potential),
        indices_z,
    )

    return img_out



def _gauss(x, mu, sigma):
    # TODO (CS): do we need the extra normalization term?
    return jnp.exp(-0.5 * jnp.square(x - mu) / jnp.square(sigma))


@jax.jit
def bilateral_gauss_cc(img1, img2, kernel, sigma):
    sl_center = tuple([sh//2 for sh in kernel.shape])
    def potential(patch):
        mu = patch[sl_center]
        return _gauss(patch, mu, sigma)

    out = bilateral_filter(img1, img2, kernel, potential)

    return out
