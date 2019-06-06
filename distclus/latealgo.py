from threading import Lock

import numpy as np


class LateAlgo:

    def __init__(self, builder):
        self._builder = builder
        self._buffer = []
        self._algo = None
        self._latestart = False
        self._mu = Lock()

    def push(self, data):
        self._buffer = [*self._buffer, *data]
        self._try_initialize()

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

    def _try_initialize(self):
        if self._algo is None:
            self._mu.acquire()
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