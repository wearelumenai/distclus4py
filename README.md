# Distclus4py

> Multi-threaded online distance based clustering library

# Introduction

This library implements the concepts and theoretical results described in the article https://hal.inria.fr/hal-01264233.
It is based on a Go library (https://github.com/wearelumenai/distclus) compiled in native format and binded with CFFI.

# Installation

```
$ make build
```

# Test

```
$ make test
```

# Static learning

Static learning means that when the algorithm is run, all train data are already known
(this is the common way to do machine learning).
 
The algorithms in this library implements the following interface, compliant with [scikit-learn](https://scikit-learn.org) :
 - ```fit(data)```: Compute the clustering using *data* as training samples.
 - ```predict(data)```: Predict the closest cluster each sample in *data* belongs to.
 - ```cluster_centers_``` : once the model is fitted, the centers of the clusters.
 
# Online learning

Online learning occurs when new data are used to update the model after the algorithm is started.

The algorithm provided by this library implements a specific interface, dedicated to online learning :
 - ```run(rasync=True)```

# Algorithms
 
## MCMC
 
The library offers 3 clustering algorithms :
 - MCMC
 - KMeans
 - Streaming
 
 ```python
class distclus.MCMC(
    space='vectors', par=True, init='kmeanspp',
    init_k=8, max_k=16, mcmc_iter=100, frame_size=None, 
    b=1., amp=1., dim=None, nu=3., norm=2,
    seed=None, data=None, inner_space=None, window=None
)
```

Parameter name | values | default | description *
-------------- | ------ | ------- | -------------
```space``` | *'vectors', 'cosinus','series'* | *'vectors'* | how distance and barycenters are computed
```par``` | *boolean* | *True* | indicates if computation is done in parallel
```init``` | *'kmeanspp', 'random', 'given'* | *'kmeanspp'* | the way initial centers are chosen
```init_k``` | *int* | *8* | the number of initial centers
```max_k``` | *int* | *16* | the maximum number of center
```mcmc_iter``` | *int* | *100* | the number of mcmc iteration (only used if *```par=False```*)
```frame_size``` | *int* | *None* | the number of data used for computation: <br> - None means all data, <br> - N > 0 means the N last pushed values
```b``` | *float* | *1.* | the value of the *b* parameter (used for the acceptation computation)
```amp``` | *float* | *1.* | the value of the *b* parameter (used for the acceptation computation)
```dim``` | *int* | *None* | the size of the data points (used by the student distribution)
```nu``` | *float* | *3.* | the size of the data points (used by the student distribution)
```norm``` | *float* | *2.* | the power of the p-norm used to compute the loss generalized mean
```seed``` | *int* | *None* | the seed of the pseudo-random number generator. If None the seed is computed from epoch.
```data``` | *ndarray* | *None* | data to be pushed at algorithm construction time (optional)
```inner_space``` | *'vectors', 'cosinus'* | *None* | inner space when *```space='series'```*
```window``` | *int* | *None* | size of window for *```space='series'```*

<sup>* for more information on parameter values please refer to the article https://hal.inria.fr/hal-01264233</sup>

The following example create the algorithm, fit train data then predict sample data :
```python
>>> import numpy as np
>>> from distclus import MCMC
>>> 
>>> train = np.array([[0., 3.], [15., 5.], [0., 5.], [0., 8.], [15., 1.], [15., 6.]])
>>> test = np.array([[1., 4.], [13., 2.]])
>>>
>>> algo = MCMC(init_k=2, b=10., amp=.05, dim=2)
>>> algo.fit(train)
>>> algo.centroids
array([[15.        ,  4.        ],
       [ 0.        ,  5.33333333]])
       
>>> labels = algo.predict(train)
>>> print(labels)
[0 1 0 0 1 1]

>>> predictions = algo.predict(test)
>>> print(predictions)
[0 1]
```

## Streaming

```python
class distclus.Streaming(
    space='vectors', buffer_size=0,
    b=.95, lambd=3.,
    seed=None, data=None, inner_space=0, window=10
)
```

Parameter name | values | default | description
-------------- | ------ | ------- | -----------
```space``` | *'vectors', 'cosinus','series'* | *'vectors'* | how distance and barycenters are computed
```buffer_size``` | *int* | *100* | the size of the buffer used to store samples before being consumed by the algorithm *
```b``` | *float* | *.95* | the value of the *b* parameter
```labmd``` | *float* | *3.* | the value of the *lambda* parameter
```seed``` | *int* | *None* | the seed of the pseudo-random number generator. If None the seed is computed from epoch.
```data``` | *ndarray* | *None* | data to be pushed at algorithm construction time (optional)
```inner_space``` | *'vectors', 'cosinus'* | *None* | inner space when *```space='series'```*
```window``` | *int* | *None* | size of window for *```space='series'```*

<sup>* this algorithm is designed for online usage (see below). For static usage, ```buffer_size``` must be set to the number of samples.</sup>

The following example create the algorithm, fit train data then predict sample data :
```python
import numpy as np
from distclus import Streaming

train = np.array([[0., 3.], [15., 5.], [0., 5.], [0., 8.], [15., 1.], [15., 6.]])
test = np.array([[1., 4.], [13., 2.]])

algo = Streaming()
algo.fit(train)
algo.centroids
array([[15.        ,  4.        ],
       [ 0.        ,  5.33333333]])
       
>>> labels = algo.predict(train)
>>> print(labels)
[0 1 0 0 1 1]

>>> predictions = algo.predict(test)
>>> print(predictions)
[0 1]
```

