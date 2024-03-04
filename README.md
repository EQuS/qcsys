# qcsys
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Shantanu&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=EQuS&color=inactive"/>
</p>

***Documentation**: [github.com/pages/EQuS/qcsys](https://github.com/pages/EQuS/qcsys/)*
## Motivation

Built on JAX,  `qcsys` presents a scalable way to assemble and simulate systems of quantum circuits. 

## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `qcsys` into your conda environment.*

`qcsys` may soon be published on PyPI. Once it is, simply run the following code to install the package:

```bash
pip install qcsys
```
If you also want to download the dependencies needed to run optional tutorials, please use `pip install qcsys[dev,docs]` or `pip install 'qcsys[dev,docs]'` (for `zsh` users).


To check if the installation was successful, run:

```python
>>> import qcsys
```

## Building from source

To build `qcsys` from source, pip install using:

```bash
git clone https://github.com/EQuS/qcsys.git
cd qcsys
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev,docs]` or `pip install --upgrade '.[dev,docs]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `qcsys` in editable mode as follows:
```bash
git clone https://github.com/EQuS/qcsys.git
cd qcsys
pip install -e .[dev,docs]
```

Please use `pip install -e '.[dev,docs]'` if you are a `zsh` user.

## Documentation

Documentation should be viewable here: [https://github.com/pages/EQuS/qcsys/](https://github.com/pages/EQuS/qcsys/) 

### Build and view locally

To view documentation locally, plesae make sure the install the requirements under the `docs` extra, as specified above. Then, run the following:

```
mkdocs serve
```

The documentation should now be at the url provided by the above command. 

### Updating Docs

The documentation should be updated automatically when any changes are made to the `main` branch. However, updates can also be forced by running:

```
mkdocs gh-deploy --force
```
This will build your documentation and deploy it to a branch gh-pages in your repository.

## Acknowledgements

**Core Devs:** [Shantanu Jha](https://github.com/Phionx) and [Shoumik Chowdhury](https://github.com/shoumikdc).


This project was created by the Engineering Quantum Systems group at MIT.

