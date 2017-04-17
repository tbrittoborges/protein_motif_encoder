#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'future',
    'pandas>=0.19.1',
    'Biopython>=1.6'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='protein_motif_encoder',
    version='0.1.0',
    description="Python tools for encoding protein motif  sequence for machine learning",
    long_description=readme + '\n\n' + history,
    author="Thiago Britto-Borges",
    author_email='tbrittborges@dundee.ac.uk',
    url='https://github.com/tbrittoborges/protein_motif_encoder',
    packages=[
        'protein_motif_encoder',
    ],
    package_dir={'protein_motif_encoder':
                 'protein_motif_encoder'},
    entry_points={
        'console_scripts': [
            'protein_motif_encoder=protein_motif_encoder.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='protein_motif_encoder',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
