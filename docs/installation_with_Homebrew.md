# Installation guide using brew

This project requires **Python 3.12** and uses [Poetry](https://python-poetry.org/) for dependency management.  
Follow the steps below to install the required tools and set up your environment.

## Installation
* Clone the repository

  ```shell
    git clone git@github.com:IBM/agentics.git
    cd agentics
  ```

* We install Poetry using Homebrew:

```bash
brew install poetry
```

You can verify the installation with:

```bash
poetry --version
```

* Install Python 3.12 (via Homebrew)

Next, install Python **3.12**:

```bash
brew install python@3.12
```


*  Configure Poetry to use Python 3.12

Tell Poetry to use the Python 3.12 executable installed by Homebrew:

```bash
poetry env use $(brew --prefix)/opt/python@3.12/bin/python3.12
```

* Install Dependencies

Once Python and Poetry are set up, install the project dependencies:

```bash
poetry install
```

You can set up the env in the local folder by running

```bash
poetry config virtualenvs.in-project true
```




