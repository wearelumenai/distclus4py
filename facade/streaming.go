package main

//#include "bind.h"
import "C"
import (
	"time"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/streaming"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a Streaming algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real Streaming instance is stored in a global table and accessed with a descriptor.

// Streaming builds and registers a streaming algorithm
//export Streaming
func Streaming(
	space C.space, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t,
	seed C.long, bufsize C.int,
	mu C.double, sigma C.double,
	outRatio C.double, outAfter C.int,
	iter C.int, iterFreq C.float, dataPerIter C.int,
	timeout C.int,
	innerSpace C.space, window C.int,
) (descr C.int, errMsg *C.char) {
	defer handlePanic(0, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var implConf = streamConf(
		bufsize, mu, sigma, outRatio, outAfter, seed, iter, iterFreq, dataPerIter, timeout,
	)
	var implSpace = getSpace(space, window, innerSpace)
	var algo = streaming.NewAlgo(implConf, implSpace, elemts)
	descr = C.int(RegisterAlgorithm(algo, implSpace))
	return
}

func streamConf(
	bufsize C.int, mu C.double, sigma C.double, outRatio C.double, outAfter C.int, seed C.long,
	iter C.int, iterFreq C.float, dataPerIter C.int, timeout C.int,
) streaming.Conf {

	var rgen *rand.Rand
	if seed != 0 {
		rgen = rand.New(rand.NewSource((uint64)(seed)))
	}

	return streaming.Conf{
		BufferSize: int(bufsize),
		Mu:         float64(mu),
		Sigma:      float64(sigma),
		OutRatio:   float64(outRatio),
		OutAfter:   int(outAfter),
		RGen:       rgen,
		CtrlConf: core.CtrlConf{
			Iter:        (int)(iter),
			IterFreq:    (float64)(iterFreq),
			DataPerIter: (int)(dataPerIter),
			Timeout:     (time.Duration)(timeout),
		},
	}
}
