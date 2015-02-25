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
        return vocab.readlines()


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
    """Returns tuples of document ids and bow vectors (0-indexed)"""
    path_to_dataset = path(path_to_dataset)

    with open(path_to_dataset / 'Indices.csv', 'rb') as indices:
        reader = csv.reader(indices)
        for row in reader:
            # First col is id, rest are indices, all ints
            docId = int(row[0])
            bow = [int(idx) - 1 for idx in row[1:]]
            yield (docId, bow)


class Document(object):
    def __init__(self, id, label, is_train=True, is_validation=False, is_test=False):
        self.id = id
        self.label = label
        self.is_train = is_train
        self.is_validation = is_validation
        self.is_test = is_test
        self.bow = None

    def set_bow(self, bow):
        self.bow = bow

def load_dataset(path_to_folder):
    """Returns a tuple of a collection of `Document` objects and a vocab list."""
    
    root = path(path_to_folder)

    print "Loading dataset %s" % path_to_folder

    vocab = load_vocab(root)
    labels = load_labels(root)
    indices = load_indices(root)

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

    # Add the indices
    for id, bow in indices:
        if id not in docs_by_id:
            raise ValueError("Document %d from indices file not in labels file" % id)

        if max(bow) >= len(vocab):
            raise ValueError(
                "Document %d has max term index of %d and vocab only has %d words" % (id, max(bow), len(vocab)))

        if min(bow) < 0:
            raise ValueError("Document %d has min term index %d, below 0" % (id, min(bow)))

        doc = docs_by_id[id]
        doc.set_bow(bow)


    print "Vocab has %d words." % len(vocab)
    print "Dataset contains %d documents" % len(docs_by_id)
    
    return docs_by_id.values(), vocab


def convert_to_sklearn(documents, vocab):
    """Convert a document collection to an sklearn dataset"""
    