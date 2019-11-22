# Distclus4py

> Multi-threaded online distance based clustering library

# Introduction

This library implements the concepts and theoretical results described in the article https://hal.inria.fr/hal-01264233.
It is based on a Go library (https://github.com/wearelumenai/distclus) compiled in native format and binded with CFFI.

# Requirements
* A Go 1.11 proper environment must be installed and configured before proceeding to the library installation. Refer to https://golang.org/doc/install.
* Although the Makefile should do it, it is better to install previously and separately the [distclus](https://github.com/wearelumenai/distclus) library for now.

# Installation

The repo should be clone inside your GO environnment. One standard location for it is to put it inside `~/go/src`

[//]: <> (This will build the go library and copy it in the python `distclus` package.)

```
$ cd ~/go/src/
$ git clone https://github.com/wearelumenai/distclus4py
$ cd distclus4py
$ make build
```

# Static learning

Static learning means that when the algorithm is run, all train data are already known
(this is the common way to do machine learning).
 
The algorithms in this library implements the following interface, compliant with [scikit-learn](https://scikit-learn.org) :
 - ```fit(data)```: Compute the clustering using *data* as training samples.
 - ```predict(data)```: Predict the closest cluster each sample in *data* belongs to.
 - ```cluster_centers_``` : Once the model is fitted, the centers of the clusters.
 
Example of usage : create the algorithm, fit train data then predict sample data.
```python
import numpy as np
from distclus import MCMC

train = np.array([[0., 3.], [15., 5.], [0., 5.], [0., 8.], [15., 1.], [15., 6.]])
test = np.array([[1., 4.], [13., 2.]])

algo = MCMC(init_k=2, b=10., amp=.05, dim=2)
algo.fit(train)
print(algo.centroids)
labels = algo.predict(train)
print(labels)

predictions = algo.predict(test)
print(predictions)
```
output :
```
[[ 0.          5.33333333]
 [15.          4.        ]]
       
[0 1 0 0 1 1]

[0 1]
```
# Online learning

Online learning occurs when new data are used to update the model after the algorithm is started.

The algorithm provided by this library implements a specific interface, dedicated to online learning :
 - ```push(data)``` : Push data to feed the algorithm and update the model
 - ```run(rasync=True)``` : Start the algorithm in asynchronous (background) mode
 - ```predict_online(data)``` : Get a tuple current centroids and labels of the closest one for the given data
 - ```close()``` : Stop the background algorithm and release resources
 - ```centroids``` : Get the current centroids
 
 Before starting the algorithm (with the ```run``` method),
 it must have been fed with enough data to initialize properly (with the ```push```).
 This behavior can be enhanced thanks to the ```LateAlgo``` decorator (see below).
 
Example of usage : create the algorithm, push enough train data to initialize, run the algorithm in background mode,
push more data then predict test data
```python
import time
import numpy as np
from distclus import MCMC

data = np.array([[[0., 3.], [15., 5.]], [[0., 5.], [0., 8.]], [[15., 1.], [15., 6.]], [[1., 4.], [13., 2.]]])

algo = MCMC(init_k=2, b=10., amp=.05, dim=2)
# algorithm initialization needs data
algo.push(data[0])
algo.run()
for i, chunk in enumerate(data):
    time.sleep(.5) # wait for the algorithm to converge
    # simulate new data arrival
    centroids, labels = algo.predict_online(chunk)
    algo.push(chunk)
    print("chunk", i)
    print(centroids)
    print(labels)

algo.close()
```
output :
```
chunk 2 4
[[ 0.  3.]
 [15.  5.]]
[0 0]
chunk 4 6
[[ 0.          5.33333333]
 [15.          5.        ]]
[1 1]
chunk 6 8
[[ 0.          5.33333333]
 [15.          4.        ]]
[0 1]
```

The ```run``` method returns a context manager that can be used to properly close the algorithm :

```python
with algo.run():
    for chunk in data:
        algo.push(chunk)
        # ...
```

# Algorithms
 
The library offers 3 clustering algorithms :
 - [MCMC](#mcmc)
 - [Streaming](#streaming)
 - [KMeans](#kmeans)
 
## MCMC
 
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
```mcmc_iter``` | *int* | *100* | the number of mcmc iteration
```frame_size``` | *int* | *None* | the number of data used for computation: <br> - None means all data, <br> - N > 0 means the N last pushed values
```amp``` | *float* | *1.* | The weight for the term related to the data in the acceptation. Increasing it results in a larger number of clusters.
```dim``` | *int* | *None* | the size of the data points (used by the student distribution)
```nu``` | *float* | *3.* | the size of the data points (used by the student distribution)
```norm``` | *float* | *2.* | the power of the p-norm used to compute the loss generalized mean
```seed``` | *int* | *None* | the seed of the pseudo-random number generator. If None the seed is computed from epoch.
```data``` | *ndarray* | *None* | data to be pushed at algorithm construction time (optional)
```inner_space``` | *'vectors', 'cosinus'* | *None* | inner space when *```space='series'```*
```window``` | *int* | *None* | size of window for *```space='series'```*

<sup>* for more information on parameter values please refer to the article https://hal.inria.fr/hal-01264233</sup>


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

## KMeans

```python

class distclus.KMeans(
    space='vectors', par=True, init='kmeanspp',
    k=16, nb_iter=100, frame_size=None,
    seed=None, data=None, inner_space=None, window=None
)
```

Parameter name | values | default | description
-------------- | ------ | ------- | -----------
```space``` | *'vectors', 'cosinus','series'* | *'vectors'* | how distance and barycenters are computed
```par``` | *boolean* | *True* | indicates if computation is done in parallel
```init``` | *'kmeanspp', 'random', 'given'* | *'kmeanspp'* | the way initial centers are chosen
```k``` | *int* | *8* | the number of clusters
```nb_iter``` | *int* | *100* | the number of iteration
```frame_size``` | *int* | *None* | the number of data used for computation: <br> - None means all data, <br> - N > 0 means the N last pushed values
```seed``` | *int* | *None* | the seed of the pseudo-random number generator. If None the seed is computed from epoch.
```data``` | *ndarray* | *None* | data to be pushed at algorithm construction time (optional)
```inner_space``` | *'vectors', 'cosinus'* | *None* | inner space when *```space='series'```*
```window``` | *int* | *None* | size of window for *```space='series'```*

# Advanced usage

The library offers two decorators :
 - ```LateAlgo``` : differ the initialization of an algorithm until data arrives
 - ```Batch``` : use an algorithm by running subsequent mini-batches
 
## LateAlgo
```LateAlgo``` is useful when data knowledge is needed to initialize the algorithm.
The following example builds a MCMC algorithm when ```init_k``` data are known
to initialize the algorithm, it also deduce the ```dim``` from the data. 

```python
import numpy as np
from distclus import LateAlgo, MCMC

def LateMCMC(init_k, **kwargs):

    def builder(data):
        if len(data) >= init_k:
            kw = {**kwargs, 'init_k': init_k, 'dim': len(data[0])}
            return MCMC(**kw, data=data)

    return LateAlgo(builder)
    

data = np.array([[0., 3.], [15., 5.], [0., 5.], [0., 8.], [15., 1.], [15., 6.], [1., 4.], [13., 2.]])
algo = LateMCMC(init_k=2, mcmc_iter=20, seed=166348259467)
algo.run()
algo.push(data) # the algorithm will initialize properly
# ...
algo.close()
```

```LateAlgo``` has also a context manager :
```python
with LateMCMC(init_k=2, mcmc_iter=20, seed=166348259467) as algo:
    algo.push(data)
    # ...
```

## Batch
```Batch``` allows to implement the "mini-batch" pattern. Each time data is pushed, the algorithm runs few iterations,
starting from the previous configuration.

```python
import numpy as np
from distclus import Batch, MCMC

data = np.array([[[0., 3.], [15., 5.], [0., 5.], [0., 8.]], [[15., 1.], [15., 6.], [1., 4.], [13., 2.]]])

algo = Batch(MCMC, init_k=2, b=500, amp=0.1, mcmc_iter=10)
with algo.run():
    for chunk in data:
        algo.push(chunk)
        centroids, labels = algo.predict_online(chunk)
        # ...
```

```Batch``` constructor also accepts an optional parameter ```frame_size``` that indicates the size of data to be pushed 
to the underlying algorithm. To achieve this, data are memoized between ```push``` calls in order to complete new data
with historical data if necessary.

```python
algo = Batch(MCMC, frame_size=100, init_k=2, b=500, amp=0.1, mcmc_iter=10)
```
