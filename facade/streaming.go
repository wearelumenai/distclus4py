package main

//#include "bind.h"
import "C"
import (
	"distclus/streaming"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a Streaming algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real Streaming instance is stored in a global table and accessed with a descriptor.

// Streaming algorithm
//export Streaming
func Streaming(
	space C.space, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t,
	seed C.long, bufsize C.int,
	b C.double, lambda C.double,
	innerSpace C.space, window C.int,
) (descr C.int, errMsg *C.char) {
	defer handlePanic(0, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var implConf = streamConf(bufsize, b, lambda, seed)
	var implSpace = getSpace(space, window, innerSpace)
	var algo = streaming.NewAlgo(implConf, implSpace, elemts)
	descr = C.int(RegisterAlgorithm(algo))
	return
}

func streamConf(bufsize C.int, b C.double, lambda C.double, seed C.long) streaming.Conf {

	var rgen *rand.Rand
	if seed != 0 {
		rgen = rand.New(rand.NewSource((uint64)(seed)))
	}

	return streaming.Conf{
		BufferSize: int(bufsize),
		B:          float64(b),
		Lambda:     float64(lambda),
		RGen:       rgen,
	}
}
