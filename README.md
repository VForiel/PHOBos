# Kernel test bench controls

This repo aim to provide a full set of tools to control all the devices on the Kernel-Nuller test bench.

## 🚀 Quickstart

Requirements:
- [Python 3.12](https://www.python.org/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)

### Option 1: Installation with Conda (recommended)

1. Create and activate the conda environment
    ```bash
    conda env create -f environment.yml
    conda activate kbench-controls
    ```

2. Start a Python instance
    ```bash
    python
    ```
    And import the kbench module
    ```python
    import kbench
    ```

### Option 2: Installation with pip

1. (Recommended) Create a virtual environment
    ```bash
    python3.12 -m venv .venv
    ```
    and activate it
    ```bash
    source .venv/bin/activate # Linux
    .venv/Scripts/activate # Windows
    ```

2. Install the Python module
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3. Start a Python instance
    ```bash
    python
    ```
    And import the kbench module
    ```python
    import kbench
    ```
    You can now use all Kbench devices according to the documentation!

## 📚 Documentation

The documentation should be available at the adress: [kbench.readthedocs.io](http://kbench.readthedocs.io).

If you want to build the doc locally, once the project is setup (according to the instructions above):

1. Go in the `docs` folder
    ```bash
    cd docs
    ```
2. Install the requirements (by preference, in a virtual environment)
    ```bash
    pip install -r requirements.txt
    ```
3. Build the doc
    ```bash
    make html # Linux
    .\make.bat html # Windows
    ```
Once the documentation is build, you can find it in the `docs/_build_` folder.