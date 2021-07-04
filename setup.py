"""
Setup file for DeepRelNN
Refer to https://github.com/rodrigoazs/DeepRelNN
"""

from os import path

from setuptools import find_packages, setup

setup(
    name="deeprelnn",
    packages=find_packages(exclude=["test"]),
    package_dir={"deeprelnn": "deeprelnn"},
    author="Rodrigo Azevedo Santos (rodrigoazs)",
    author_email="rodrigoazvsantos@gmail.com",
    version="0.1.0",
    description="A deep learning approach for SRL.",
    include_package_data=True,
    package_data={"deepedn": []},
    download_url="https://github.com/rodrigoazs/DeepRelNN",
    license="GPL-3.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning-algorithms machine-learning statistical-learning pattern-classification artificial-intelligence",
    install_requires=["numpy", "pandas", "tensorflow"],
    extras_require={
        "tests": ["pytest"],
    },
)
