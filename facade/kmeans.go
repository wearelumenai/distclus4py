package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/cosinus"
	"distclus/kmeans"
	"distclus/series"
	"distclus/vectors"
	"fmt"
	"golang.org/x/exp/rand"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// KMEANS algorithm
//export KMEANS
func KMEANS(
	space C.space,
	par C.int,
	initializer C.initializer,
	seed C.long,
	data *C.double, l1 C.size_t, l2 C.size_t,
	k C.int, iter C.int, framesize C.int,
	innerSpace C.space, window C.int,
) (descr C.int, errMsg *C.char) {
	defer handlePanic(0, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2)
	var implConf = kmeansConf(par, k, iter, framesize, seed)
	var implSpace = getSpace(space, window, innerSpace)
	var implInit = Initializer(initializer)
	var algo = kmeans.NewAlgo(implConf, implSpace, elemts, implInit)
	descr = C.int(RegisterAlgorithm(algo))
	return
}

func kmeansConf(
	par C.int,
	k C.int, iter C.int, framesize C.int, seed C.long,
) kmeans.Conf {
	return kmeans.Conf{
		Par: par != 0, K: (int)(k), FrameSize: (int)(framesize), Iter: (int)(iter),
		RGen: rand.New(rand.NewSource((uint64)(seed))),
	}
}

func getSpace(spaceName C.space, window C.int, innerSpace C.space) core.Space {
	switch spaceName {
	case C.S_SERIES:
		var conf = series.Conf{
			InnerSpace: getSpace(innerSpace, 0, 0),
			Window:     (int)(window),
		}
		return series.NewSpace(conf)
	case C.S_VECTORS:
		var conf = vectors.Conf{}
		return vectors.NewSpace(conf)
	case C.S_COSINUS:
		var conf = cosinus.Conf{}
		return cosinus.NewSpace(conf)
	default:
		panic(fmt.Sprintf("unknown space %v", spaceName))
	}
}
