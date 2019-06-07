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

# Basic usage

The library offers 3 clustering algorithms :
 - MCMC
 - KMeans
 - Streaming
 
All three algorithms implements the following interface, compliant with [scikit-learn](https://scikit-learn.org) :
 - ```fit(X)```: Compute the clustering.
 - ```predict(X)```: Predict the closest cluster each sample in X belongs to.
 
 ## MCMC
 
 ```python
class distclus.MCMC(
    space='vectors', par=True, init='kmeanspp',
    init_k=8, max_k=16, mcmc_iter=100, frame_size=None, 
    b=1., amp=1., dim=None, nu=3.,
    norm=2, seed=None,
    data=None, inner_space=None, window=None
)
```

Parameter name | values | default | description*
-------------- | ------ | ------- | -----------
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

<sup>*for more information on parameter values please refer to the article https://hal.inria.fr/hal-01264233</sup>