package main

//#include "bind.h"
import "C"
import (
	"time"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/mcmc"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// MCMC builds and registers a mcmc algorithm
//export MCMC
func MCMC(
	space C.space, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t,
	par C.int, init C.initializer, initDescr C.int, seed C.long,
	dim C.size_t, initK C.int, maxK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	iterFreq C.float, dataPerIter C.int, timeout C.int, numCPU C.int,
	innerSpace C.space, window C.int,
) (descr C.int, errMsg *C.char) {
	defer handlePanic(0, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var implConf = mcmcConf(
		par, initK, maxK, mcmcIter, framesize, b, amp, norm, seed,
		iterFreq, dataPerIter, timeout, numCPU,
	)
	var implSpace = getSpace(space, window, innerSpace)
	var implInit = initializer(init, initDescr)
	var distrib = buildDistrib(implSpace, dim, nu, space)
	var algo = mcmc.NewAlgo(implConf, implSpace, elemts, implInit, distrib)
	descr = C.int(RegisterAlgorithm(algo, implSpace))
	return
}

func buildDistrib(implSpace core.Space, dim C.size_t, nu C.double, space C.space) (distrib mcmc.Distrib) {
	switch {
	case space == C.S_SERIES:
		distrib = mcmc.NewDirac()
	case (int)(dim) == 0:
		var init = func(data core.Elemt) mcmc.Distrib {
			var c = mcmc.MultivTConf{
				Dim: implSpace.Dim([]core.Elemt{data}),
				Nu:  (float64)(nu),
			}
			return mcmc.NewMultivT(c)
		}
		distrib = mcmc.NewLateDistrib(init)
	default:
		var c = mcmc.MultivTConf{
			Dim: (int)(dim),
			Nu:  (float64)(nu),
		}
		distrib = mcmc.NewMultivT(c)
	}
	return
}

func mcmcConf(par C.int,
	initK C.int, maxK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, seed C.long,
	iterFreq C.float, dataPerIter C.int,
	timeout C.int, numCPU C.int,
) mcmc.Conf {

	var rgen *rand.Rand
	if seed != 0 {
		rgen = rand.New(rand.NewSource((uint64)(seed)))
	}

	return mcmc.Conf{
		Par:       par != 0,
		FrameSize: (int)(framesize), B: (float64)(b), Amp: (float64)(amp),
		Norm: (float64)(norm), InitK: (int)(initK), MaxK: (int)(maxK),
		RGen:   rgen,
		NumCPU: (int)(numCPU),
		CtrlConf: core.CtrlConf{
			Iter:        (int)(mcmcIter),
			IterFreq:    (float64)(iterFreq),
			DataPerIter: (int)(dataPerIter),
			Timeout:     (time.Duration)(timeout),
		},
	}
}
