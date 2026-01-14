"""
Setup script for MUSiK: Multi-transducer Ultrasound Simulations in K-wave
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

# Get version from the package
def get_version():
    version_file = os.path.join("musik", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="musik",
    version=get_version(),
    author="Trevor Chan, Aparna Nair-kanneganti",
    author_email="tjchan@seas.upenn.edu",
    description="Multi-transducer Ultrasound Simulations in K-wave",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/norway99/MUSiK",
    project_urls={
        "Bug Reports": "https://github.com/norway99/MUSiK/issues",
        "Source": "https://github.com/norway99/MUSiK",
        "Documentation": "https://github.com/norway99/MUSiK/blob/main/README.md",
    },
    packages=find_packages(include=["musik", "musik.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "full": [
            "imageio>=2.33.0",
            "json5>=0.9.14",
            "meshlib>=2.4.1.114",
            "nptyping>=2.5.0",
            "open3d>=0.18.0",
            "ordered-set>=4.0.2",
            "pandas>=2.1.3",
            "pydicom>=2.4.4",
            "pyquaternion>=0.9.9",
            "PyWavelets>=1.5.0",
            "tifffile>=2023.9.26",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ultrasound simulation k-wave medical imaging acoustics",
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here if needed
        ],
    },
)
