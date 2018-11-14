package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/mcmc"
	"distclus/real"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

//export MCMC
func MCMC(dim C.size_t,
	initK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	initIter C.int, initializer C.initializer, seed C.long) C.int {

	var mcmcConf = MakeConfig(dim, initK, mcmcIter, framesize, b, amp, norm, nu, initIter, seed)
	var distrib = mcmc.NewMultivT(mcmc.MultivTConf{mcmcConf})
	var algo = mcmc.NewParMCMC(mcmcConf, distrib, Initializer(initializer), []core.Elemt{})
	return C.int(RegisterAlgorithm(algo))
}

func MakeConfig(l2 C.size_t,
	initK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	initIter C.int, seed C.long) mcmc.MCMCConf {

	return mcmc.MCMCConf{
		AlgorithmConf: core.AlgorithmConf{Space: real.RealSpace{}},
		Dim:           (int)(l2), FrameSize: (int)(framesize), B: (float64)(b), Amp: (float64)(amp),
		Norm: (float64)(norm), Nu: (float64)(nu), InitK: (int)(initK), McmcIter: (int)(mcmcIter),
		InitIter: (int)(initIter),
		RGen:     rand.New(rand.NewSource((uint64)(seed))),
	}
}
