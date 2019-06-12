import itertools as it


class Batch:

    def __init__(self, builder, **kwargs):
        self._builder = builder
        self._kwargs = kwargs
        self._algo = None

    def fit_batch(self, data):
        if self._algo:
            self._kwargs['init'] = self._algo
        self._algo = self._builder(**self._kwargs)
        self._algo.fit(data)


    def predict(self, data):
        return self._algo.predict(data)

    @property
    def centroids(self):
        return self._algo.centroids


def chunker(iterable, n):
    """Get chunks of fixed size from an iterable
    :param iterable: the iterable
    :param n: chunk size
    :return: a generator of chunks
    """
    gen = iter(iterable)
    while True:
        current = list(it.islice(gen, n))
        if current:
            yield current
        else:
            return