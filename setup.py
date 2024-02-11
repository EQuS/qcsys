"""
Setup
"""
import os

from setuptools import setup, find_namespace_packages


DIST_NAME = "qcsys"
PACKAGE_NAME = "qcsys"

REQUIREMENTS = ["numpy", "matplotlib", "qutip", "jax[cpu]", "flax"]

EXTRA_REQUIREMENTS = {
    "dev": [
        "jupyterlab>=3.1.0",
        "mypy",
        "pylint",
        "black",
        "mkdocs",
        "mkdocs-material",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "mkdocstrings-python",
    ],
}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()


setup(
    name=DIST_NAME,
    version=version_str,
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EQuS/qcsys",
    author="Shantanu Jha, Shoumik Chowdhury",
    author_email="shantanu.rajesh.jha@gmail.com",
    license="MIT",
    packages=find_namespace_packages(exclude=["tutorials*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    keywords="quantum computing circuits",
    python_requires=">=3.7",
    project_urls={
        "Documentation": "https://github.com/EQuS/qcsys",
        "Source Code": "https://github.com/EQuS/qcsys",
        "Tutorials": "https://github.com/EQuS/qcsys/tree/master/tutorials",
    },
    include_package_data=True,
)
