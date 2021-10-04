# PlaineDFT
[![license](https://img.shields.io/badge/license-MIT-green)](https://gitlab.com/wangenau/plainedft/-/blob/master/LICENSE)
[![language](https://img.shields.io/badge/language-Python3-blue)](https://www.python.org/)

PlaineDFT is a plane wave density functional theory (DFT) code.

It is built upon the [DFT++](https://arxiv.org/abs/cond-mat/9909130) pragmas, that aim to let programming languages and theory coincide.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.

## Documentation

To learn more about the implementation, take a look inside the [documentation](https://wangenau.gitlab.io/plainedft/).

## Installation

The package and all necessary dependencies can be installed using pip

```bash
git clone https://gitlab.com/wangenau/plainedft
cd plainedft
pip install .
```

To also install all optional dependecies to use built-in addons, use

```bash
pip install .[addons]
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://gitlab.com/wangenau/plainedft/-/blob/master/LICENSE) file for details.

