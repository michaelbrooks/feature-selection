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

Dataset Input Format
--------------------

Feature selection requires as input information about a collection
of labeled documents that have been converted into bag-of-words term vectors.
Information about these documents should be provided in three files together in one folder.
See the `test_data` folder for an example.

The `Vocab.csv` file is the bag of words vocabulary, a list of the terms from across the documents.
The position of the term in this list should correspond to the index in the bag of words vectors.

The `Labels.csv` file contains information about the label of each document.
It has five columns:

```
DocId, Label, IsInTrainingSet, IsInValidationSet, IsInTestSet
23, 0, True, False, False
24, 1, False, False, True
```

The `Indices.csv` file encodes what terms exist in which documents.
The rows in this file have variable length, but the first column is
the document id. The remaining columns are numbers that refer to indices
in the vocabulary.
