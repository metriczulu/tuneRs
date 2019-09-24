# -*- coding: utf-8 -*-
"""
Created on 24 Aug 2019

@author: steph
"""

from setuptools import setup, find_packages

setup(

    name='tuneRs',
    url='https://github.com/metriczulu/tuneRs',
    author='Shane Stephenson / metriczulu',
    author_email='stephenson.shane.a@gmail.com', 
    packages=find_packages(),
    install_requires = ['numpy', 'tqdm', 'sklearn', 'scipy'],
    version='v0.6',
    license="None",
    description='Package for tuning hyperparameters with resampling methods',
    long_description_content_type='text/markdown',
    long_description=open('README.md', 'r').read(),
    download_url = 'https://github.com/metriczulu/tuneRs/archive/v0.6.tar.gz'
)
