# Feature Selection Tests

## Setup

Using VirtualBox and Vagrant (recommended, especially on Windows):

1. Install VirtualBox and Vagrant on your machine.
2. Clone the repo.
3. Run `vagrant up` in the repo directory. This installs all the dependencies.
4. SSH into the VM. Run `workon feature-selection` and `python test.py`.

If you aren't using VirtualBox and Vagrant:

1. You need to have Python 2.7 with numpy and scipy installed.
2. Run `pip install -r requirements.txt` to install the 
   dependencies specified in the `requirements.txt file.
3. Run `python test.py` to make sure it's working.

The output from the `test.py` script should be like this:

```
With all features...
Average accuracy over 2 folds: 32.0%

With Chi-squared...
All features:
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)

Selected features:
petal length (cm)
petal width (cm)
Average accuracy over 2 folds: 31.3%

With random...
All features:
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)

Selected features:
sepal length (cm)
sepal width (cm)
Average accuracy over 2 folds: 23.3%
```