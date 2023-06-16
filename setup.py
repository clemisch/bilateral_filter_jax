from setuptools import setup

setup(
    name="filter_jax",
    version="0.1",
    description="Moving-window filters in JAX",
    author="Clemens Schmid",
    author_email="clem.schmid@tum.de",
    packages=["filter_jax"],
    install_requires=["numpy", "jax"],
    license="MIT",
)
