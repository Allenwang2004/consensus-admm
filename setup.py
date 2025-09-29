import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension, find_packages
import pybind11

# Get the long description from README
def get_long_description():
    this_directory = Path(__file__).parent
    readme_path = this_directory / "README.rst"
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "consensus_admm.consensus_admm_cpp", # Where the module will be located
        sources=[
            "src/python/binding.cpp", # The main binding file
            "src/cpp/consensus_admm.cpp",
        ],
        include_dirs=[
            "include",
            pybind11.get_cmake_dir() + "/../include",
        ],
        language="c++",
        cxx_std=17,
    ),
]

# Custom build_ext class to handle Eigen dependency
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Check for Eigen3
        try:
            import eigen
            eigen_include = eigen.include_path
            for ext in self.extensions:
                ext.include_dirs.append(eigen_include)
        except ImportError:
            # Try to find Eigen via pkg-config or common locations
            eigen_dirs = [
                "/usr/include/eigen3",
                "/usr/local/include/eigen3", 
                "/opt/homebrew/include/eigen3",  # macOS Homebrew
                "/opt/local/include/eigen3",     # macOS MacPorts
            ]
            
            eigen_found = False
            for eigen_dir in eigen_dirs:
                if os.path.exists(os.path.join(eigen_dir, "Eigen")):
                    for ext in self.extensions:
                        ext.include_dirs.append(eigen_dir)
                    eigen_found = True
                    break
            
            if not eigen_found:
                print("Warning: Eigen3 not found in standard locations.")
                print("Please install Eigen3 or set the include path manually.")
                print("On macOS: brew install eigen")
                print("On Ubuntu: sudo apt-get install libeigen3-dev")
        
        # Set compiler flags
        for ext in self.extensions:
            ext.extra_compile_args = ["-O3", "-std=c++17"]
            if sys.platform == "darwin":  # macOS
                ext.extra_compile_args.append("-march=native")
            elif sys.platform.startswith("linux"):
                ext.extra_compile_args.append("-march=native")
        
        super().build_extensions()

setup(
    name="consensus-admm",
    version="0.1.0",
    author="Allen Wang", 
    author_email="your.email@example.com",
    description="A high-performance Consensus ADMM optimization toolkit",
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    url="https://github.com/Allenwang2004/consensus-admm",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "matplotlib>=3.0",
            "scipy>=1.0", 
            "scikit-learn>=0.24",
        ],
    },
    zip_safe=False,
)