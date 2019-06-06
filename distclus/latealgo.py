from threading import Lock

import numpy as np


class LateAlgo:
    """Wrapper to a late initialized algorithm"""

    def __init__(self, builder):
        self._builder = builder
        self._buffer = []
        self._algo = None
        self._latestart = False
        self._mu = Lock()

    def push(self, data):
        if self._algo:
            self._algo.push(data)
        else:
            self._try_initialize(data)

    def run(self, rasync=False):
        if self._algo:
            self._algo.run(rasync)
        elif rasync:
            self._latestart = True
        else:
            self._raise_unitialized()

    @property
    def centroids(self):
        if self._algo:
            return self._algo.centroids
        self._raise_unitialized()

    def predict(self, data):
        if self._algo:
            return self._algo.predict(data)
        self._raise_unitialized()

    def predict_online(self, data):
        if self._algo:
            return self._algo.predict_online(data)
        self._raise_unitialized()

    def close(self):
        self._algo.close()

    def _try_initialize(self, data):
            self._mu.acquire()
            self._buffer = [*self._buffer, *data]
            if self._algo is None:
                self._initialize()
            self._mu.release()

    def _initialize(self):
        algo = self._builder(np.array(self._buffer))
        if algo:
            if self._latestart:
                algo.run(True)
            self._algo = algo
            self._buffer.clear()

    def _raise_unitialized(self):
        raise ValueError('algorithm has not been initialized')