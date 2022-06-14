# Analysis of Howell's data with pymc3

## Setup 

The `exercise.py` is intended to be modified as a
[jupyter](https://jupyter.org/) notebook. The notebook format `.ipynb`, however,
is not very convenient for configuration management, testing, and giving
feedback via the usual "pull-request" mechanism provided by Github. Thus, this
repository uses
[jupytext](https://jupytext.readthedocs.io/en/latest/install.html) to **pair** a
pure Python file with a notebook with the same name. The notebook is
automatically created when you open the Python file with jupyter, and the two
files are kept in sync. Do not add `exercise.ipynb` to the files managed by git.

To start, you need the following actions (preferibly in a virtual python environment):

```sh
pip install -r requirements.txt
jupyter notebook
```

Then you can open the `exercise.py` as a notebook in the browser.

If you are using `pipenv` is even easier.


```sh
pipenv install
pipenv run jupyter notebook
```


## Test

You can execute tests locally on the python file:


```sh
mypy exercise.py
python -m doctest exercise.py
```
