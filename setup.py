from setuptools import setup, find_packages

setup(
    name='bayes_opt_pen',
    version='0.0.1',
    description='Bayesian Optimization with Penalty Term',
    author='Xie Zikai',
    author_email='zikaix@liverpool.ac.uk',
    packages=find_packages(),
    install_requires=[
        "bayesian-optimization",
        "numpy",
        "scipy",
        "pandas",
        "botorch",
        "torch"
    ],
)
