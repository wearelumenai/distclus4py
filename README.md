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
 
 
