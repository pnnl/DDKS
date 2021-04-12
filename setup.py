#!/usr/bin/env python
import os
from setuptools import setup

REQUIREMENTS = ['torch', 'numpy', 'scipy', 'pandas', 'bootstrapped', 'openimages']

setup(name='ddKS',
      version=0.1,
      description='d-Dimensional Kolmogorov-Smirnov Test',
      author='Alex Hagen, Shane Jackson, James Kahn, Jan Strube, Isabel Haide, Karl Pazdernik, Connor Hainje',
      author_email='alexander.hagen@pnnl.gov',
      url='https://github.com/pnnl/ddks',
      long_description=open('README.md').read(),
      packages=['ddks'],
      install_requires=REQUIREMENTS,
     )
