from threading import Lock
from contextlib import contextmanager

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
        """
        Push the data to the wrapped algorithm if initialized
        otherwise try to initialize
        :param data: train data
        """
        if self._algo:
            self._algo.push(data)
        else:
            self._try_initialize(data)

    def run(self, rasync=False):
        """
        Run the wrapped algorithm if initialized 
        otherwise delay the run after initialization if rasync=True
        or raise an error if rasync=False
        :param rasync: if True run in asynchronous mode
        otherwise run in synchronous mode (default)
        """
        if self._algo:
            self._algo.run(rasync)
        elif rasync:
            self._latestart = True
        else:
            self._raise_unitialized()

    @property
    def centroids(self):
        """
        Get the centroids from wrapped algorithm if initialized
        otherwise raise an error
        """
        if self._algo:
            return self._algo.centroids
        self._raise_unitialized()

    def predict(self, data):
        """
        Predict the labels from wrapped algorithm if initialized
        otherwise raise an error
        :param data: input data
        :return: labels
        """
        if self._algo:
            return self._algo.predict(data)
        self._raise_unitialized()

    def predict_online(self, data):
        """
        Get centroids and predict the labels from wrapped algorithm
        if initialized otherwise raise an error
        :param data: input data
        :return: centroids and output labels
        """
        if self._algo:
            return self._algo.predict_online(data)
        self._raise_unitialized()

    def close(self):
        """
        Stop the wrapped algorithm and release resources
        """
        self._algo.close()

    def _try_initialize(self, data):
        self._mu.acquire()
        if self._algo:
            self._algo.push(data)
        else:
            self._initialize(data)
        self._mu.release()

    def _initialize(self, data):
        self._buffer = [*self._buffer, *data]
        algo = self._builder(np.array(self._buffer))
        if algo:
            if self._latestart:
                algo.run(True)
            self._algo = algo
            self._buffer.clear()

    def _raise_unitialized(self):
        raise ValueError('algorithm has not been initialized')

    def __enter__(self):
        self.run(rasync=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
