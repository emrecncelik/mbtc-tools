#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.readlines()


setup(
    name="mbtc_tools",
    version="0.0.1",
    description="Automated NLP tools for MBT-C.",
    python_requires=">=3.8",
    author="emrecncelik",
    author_email="emrecncelik@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[requirements],
    include_package_data=True,
    packages=find_packages(
        include=[
            "mbtc_tools.*",
        ]
    ),
)
