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