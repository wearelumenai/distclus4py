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
        """
        Push the data to the wrapped algorithm if initialized
        otherwise try to initialize
        :param data: train data
        """
        if self._algo:
            self._algo.push(data)
        else:
            self._try_initialize(data)

    def _run(self, play=True):
        if self._algo:
            return getattr(self._algo, 'play' if play else 'batch')()
        elif play:
            self._latestart = True
        else:
            self._raise_unitialized()

    def play(self):
        return self._run()

    def batch(self):
        return self._run(False)

    def run(self, rasync=False):
        """
        @deprecated
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

    @property
    def cluster_centers_(self):
        return self.centroids

    @property
    def status(self):
        if self._algo:
            return self._algo.status
        self._raise_unitialized()

    def predict(self, data):
        """
        Predict the centroids and labels from wrapped algorithm if initialized
        otherwise raise an error
        :param data: input data
        :return: centroids and labels
        """
        if self._algo:
            return self._algo.predict(data)

        self._raise_unitialized()

    def stop(self):
        """
        Stop the wrapped algorithm and release resources
        """
        if self._algo:
            return self._algo.stop()
        self._raise_unitialized()

    def wait(self):
        if self._algo:
            return self._algo.wait()
        self._raise_unitialized()

    def pause(self):
        if self._algo:
            return self._algo.pause()
        self._raise_unitialized()

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
                algo.play()

            self._algo = algo
            self._buffer.clear()

    def _raise_unitialized(self):
        raise ValueError('algorithm has not been initialized')

    def __enter__(self):
        self.play()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
