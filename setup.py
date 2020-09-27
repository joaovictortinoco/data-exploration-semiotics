#!/usr/bin/python
from setuptools import find_packages
from setuptools import setup

setup(
    name='data-exploration-semiotics',
    version='0.1',
    author='Joao Tinoco / Denis Martins',
    author_email='jvtsa@ecomp.poli.br',
    install_requires=['tensorflow==2.3.0',
                      'tensorflow-transform==0.24.0'],
    packages=find_packages(exclude=['data', 'utils']),
    description='MSc. Research Implementations',
    url='https://github.com/joaovictortinoco/data-exploration-semiotics'
)