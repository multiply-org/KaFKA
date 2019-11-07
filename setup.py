#!/usr/bin/env python

from setuptools import setup, find_packages

# requirements = [
#     'pytest',
#     'numpy',
#     'scipy',
#     'gdal',
    # 'BRDF_descriptors', # Not available for automatic installation
    # 'matplotlib'
# ]
requirements = []

__version__ = None
with open('kafka/version.py') as f:
    exec(f.read())

setup(name='KaFKA',
      version=__version__,
      description='MULTIPLY KaFKA inference engine',
      author='MULTIPLY Team',
      packages=find_packages(),
      install_requires=requirements
)
