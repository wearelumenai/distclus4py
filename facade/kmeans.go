package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/series"

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
	seed C.long, k C.int, iter C.int, framesize C.int,
	innerSpace C.space, window C.int,
) C.int {
	var conf = core.Conf{
		ImplConf:  kmeansConf(par, k, iter, framesize, seed),
		SpaceConf: spaceConf(space, window, innerSpace),
	}
	return CreateOC(C.O_KMEANS, space, conf, initializer)
}

func kmeansConf(
	par C.int,
	k C.int, iter C.int, framesize C.int, seed C.long,
) kmeans.Conf {

	return kmeans.Conf{
		Par: (par != 0), K: (int)(k), FrameSize: (int)(framesize), Iter: (int)(iter),
		RGen: rand.New(rand.NewSource((uint64)(seed))),
	}
}

func spaceConf(space C.space, window C.int, innerSpace C.space) core.SpaceConf {
	if space == C.S_SERIES {
		return series.NewSpace(series.Conf{"real", (int)(window)})
	}
	return nil
}
