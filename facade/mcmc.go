package main

//#include "bind.h"
import "C"
import (
	"distclus/mcmc"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// MCMC algorithm
//export MCMC
func MCMC(
	space C.space, par C.int, initializer C.initializer, seed C.long,
	data *C.double, l1 C.size_t, l2 C.size_t,
	dim C.size_t, initK C.int, maxK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	initIter C.int,
	innerSpace C.space, window C.int,
) C.int {
	var implConf = mcmcConf(par, dim, initK, maxK, mcmcIter, framesize, b, amp, norm, nu, initIter, seed)
	var spaceConf = spaceConf(space, window, innerSpace)
	return CreateOC(implConf, spaceConf, initializer, data, l1, l2)
}

func mcmcConf(par C.int,
	l2 C.size_t,
	initK C.int, maxK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	initIter C.int, seed C.long) mcmc.Conf {

	var rgen *rand.Rand
	if seed != 0 {
		rgen = rand.New(rand.NewSource((uint64)(seed)))
	}

	return mcmc.Conf{
		Par: par != 0,
		Dim: (int)(l2), FrameSize: (int)(framesize), B: (float64)(b), Amp: (float64)(amp),
		Norm: (float64)(norm), Nu: (float64)(nu), InitK: (int)(initK), MaxK: (int)(maxK), McmcIter: (int)(mcmcIter),
		InitIter: (int)(initIter),
		RGen:     rgen,
	}
}
