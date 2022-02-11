from setuptools import setup, find_packages

setup(
    name='bayes_opt_pen',
    version='0.0.1',
    description='Bayesian Optimization with Penalty Term',
    author='Xie Zikai',
    author_email='zikaix@liverpool.ac.uk',
    packages=find_packages(),
    url='https://github.com/XieZikai/bayes_opt_pen',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License ::',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.5',
    install_requires=[
        "bayesian-optimization",
        "numpy",
        "scipy",
        "pandas",
        "botorch",
        "torch"
    ],
)
