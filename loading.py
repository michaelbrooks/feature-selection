import path as path_py
import csv

LABEL_FIELDNAMES = ('DocId', 'Label', 'IsInTrainingSet', 'IsInValidationSet', 'IsInTestSet')


def path(str_or_path):
    if not isinstance(str_or_path, path_py.path):
        return path_py.path(str_or_path)
    return str_or_path


def load_vocab(path_to_dataset):
    """Returns a list of terms"""
    path_to_dataset = path(path_to_dataset)

    with open(path_to_dataset / 'Vocab.csv', 'rb') as vocab:
        words = vocab.readlines()
        for i, word in enumerate(words):
            words[i] = word.strip()
        return words


def load_labels(path_to_dataset):
    """
    Returns a generator of dictionaries containing 
    DocId, Label, IsInTrainingSet, IsInValidationSet, IsInTestSet.
    """
    path_to_dataset = path(path_to_dataset)

    with open(path_to_dataset / 'Labels.csv', 'rb') as labels:
        reader = csv.DictReader(labels, fieldnames=LABEL_FIELDNAMES)
        for row in reader:
            # Convert to appropriate types
            row['DocId'] = int(row['DocId'])
            row['Label'] = int(row['Label'])
            for i in 'IsInTrainingSet', 'IsInValidationSet', 'IsInTestSet':
                row[i] = True if row[i] == 'True' else False

            yield row


def load_indices(path_to_dataset):
    """Returns tuples of document ids and sparse bow indices (0-indexed)"""
    path_to_dataset = path(path_to_dataset)

    with open(path_to_dataset / 'Indices.csv', 'rb') as indices:
        reader = csv.reader(indices)
        for row in reader:
            # First col is id, rest are indices, all ints
            docId = int(row[0])
            bow = [int(idx) for idx in row[1:]]
            yield (docId, bow)


def load_values(path_to_dataset):
    """Returns tuples of document ids and sparse vow values"""
    path_to_dataset = path(path_to_dataset)

    with open(path_to_dataset / 'Values.csv', 'rb') as values:
        reader = csv.reader(values)
        for row in reader:
            # First col is id, rest are values, floats
            docId = int(row[0])
            bow = [float(val) for val in row[1:]]
            yield (docId, bow)


class Document(object):
    def __init__(self, id, label, is_train=True, is_validation=False, is_test=False):
        self.id = id
        self.label = label
        self.is_train = is_train
        self.is_validation = is_validation
        self.is_test = is_test
        self.bow_indices = None
        self.bow_values = None

    def set_bow(self, indices, values):
        """Indices and values, i.e. a sparse feature vector"""
        self.bow_indices = indices
        self.bow_values = values


def load_dataset(path_to_folder):
    """
    Returns a tuple of three sklearn dataset objects, containing
    feature_names, data, and target.
    
    The first is for training, the second is for validation, and the third is test data.
    """

    root = path(path_to_folder)

    print "Loading dataset %s" % path_to_folder

    vocab = load_vocab(root)
    labels = load_labels(root)
    indices = load_indices(root)
    values = load_values(root)

    # Combine into documents
    docs_by_id = {}
    for labeldict in labels:
        doc = Document(
            id=labeldict['DocId'],
            label=labeldict['Label'],
            is_train=labeldict['IsInTrainingSet'],
            is_validation=labeldict['IsInValidationSet'],
            is_test=labeldict['IsInTestSet'],
        )

        if doc.id in docs_by_id:
            raise ValueError("Document id %d is duplicated in labels file" % doc.id)

        if doc.label not in (0, 1):
            raise ValueError("Document %d has a non-binary label %s" % (doc.id, doc.label))

        docs_by_id[doc.id] = doc

    # Add the indices and values
    for indices_row, values_row in zip(indices, values):
        indices_id, bow_indices = indices_row
        values_id, bow_values = values_row

        if indices_id != values_id:
            raise ValueError("Documents %d in indices and %d in values are mismatched rows" % (indices_id, values_id))

        id = indices_id

        if len(bow_indices) != len(bow_values):
            raise ValueError("Document %d has different indices and values lengths" % id)

        if id not in docs_by_id:
            raise ValueError("Document %d from indices file not in labels file" % id)

        if max(bow_indices) >= len(vocab):
            raise ValueError(
                "Document %d has max term index of %d and vocab only has %d words" % (id, max(bow_indices), len(vocab)))

        if min(bow_indices) < 0:
            raise ValueError("Document %d has min term index %d, below 0" % (id, min(bow_indices)))

        doc = docs_by_id[id]
        doc.set_bow(bow_indices, bow_values)

    documents = docs_by_id.values()
    print "  Vocab has %d words." % len(vocab)
    print "  Dataset contains %d documents" % len(documents)

    training = [d for d in documents if d.is_train]
    validation = [d for d in documents if d.is_validation]
    test = [d for d in documents if d.is_test]

    print "  Training data contains %d documents" % len(training)
    print "  Validation data contains %d documents" % len(validation)
    print "  Test data contains %d documents" % len(test)

    training = convert_to_sklearn(training, vocab)
    validation = convert_to_sklearn(validation, vocab)
    test = convert_to_sklearn(test, vocab)

    return training, validation, test


def unsparse(indices, values, length):
    import numpy as np

    result = np.zeros(length)
    for i, idx in enumerate(indices):
        result[idx] = values[i]

    return result


def convert_to_sklearn(documents, vocab):
    """Convert a document collection to an sklearn dataset"""
    from sklearn.datasets.base import Bunch
    import numpy as np

    n_samples = len(documents)
    n_features = len(vocab)

    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=np.int)

    for i, document in enumerate(documents):
        bowvector = unsparse(document.bow_indices, document.bow_values, n_features)
        data[i] = np.asarray(bowvector, dtype=np.float)
        target[i] = np.asarray(document.label, dtype=np.int)

    return Bunch(data=data,
                 target=target,
                 feature_names=vocab)