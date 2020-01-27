package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/kmeans"
	"time"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// KMeans builds and registers a kmeans algorithm
//export KMeans
func KMeans(
	space C.space, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t,
	par C.int, init C.initializer, initDescr C.int, seed C.long,
	k C.int, iter C.int, framesize C.int, iterFreq C.float, dataPerIter C.int,
	timeout C.int, numCPU C.int,
	innerSpace C.space, window C.int,
) (descr C.int, errMsg *C.char) {
	defer handlePanic(0, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var implConf = kmeansConf(
		par, k, iter, framesize, seed, iterFreq, dataPerIter, timeout, numCPU,
	)
	var implSpace = getSpace(space, window, innerSpace)
	var implInit = initializer(init, initDescr)
	var algo = kmeans.NewAlgo(implConf, implSpace, elemts, implInit)
	descr = C.int(RegisterAlgorithm(algo, implSpace))
	return
}

func kmeansConf(
	par C.int,
	k C.int, iter C.int, framesize C.int, seed C.long, iterFreq C.float, dataPerIter C.int,
	timeout C.int, numCPU C.int,
) kmeans.Conf {
	return kmeans.Conf{
		Par: par != 0, K: (int)(k), FrameSize: (int)(framesize),
		RGen:   rand.New(rand.NewSource((uint64)(seed))),
		NumCPU: (int)(numCPU),
		Conf: core.Conf{
			Iter:        (int)(iter),
			IterFreq:    (float64)(iterFreq),
			DataPerIter: (int)(dataPerIter),
			Timeout:     (time.Duration)(timeout),
		},
	}
}
