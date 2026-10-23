
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

setup(
    name="badapt",
    py_modules=["badapt"],
    install_requires=[],
    package_dir={"": "."},
    packages=find_packages("."),
    package_data={'': []},
    include_package_data=False,
)
