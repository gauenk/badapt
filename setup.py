

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="badapt",
    packages=find_packages("."),
    ext_modules=[
        CUDAExtension(
            name="badapt._ba_cuda",
            sources=["badapt/ba/ba.cu"],
            libraries=["cuda", "cublas", "cudadevrt"],
            extra_compile_args={
                "cxx": ["-g", "-w", "-O3"],
                "nvcc": ["-w", "-O3", "--use_fast_math",
                         "--extended-lambda"],
            },
            library_dirs=["bin"],
        ),

        CppExtension('badapt._deform_cpp',
            sources=['badapt/deform/deform.cpp'],
            extra_compile_args=['-O3', '-w'],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)

