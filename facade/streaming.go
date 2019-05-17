package main

//#include "bind.h"
import "C"
import (
	"distclus/streaming"

	"golang.org/x/exp/rand"
)

// These functions act as a facade on a STREAMING algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real STREAMING instance is stored in a global table and accessed with a descriptor.

// STREAMING algorithm
//export STREAMING
func STREAMING(
	space C.space, par C.int, init C.initializer, seed C.long,
	data *C.double, l1 C.size_t, l2 C.size_t,
	bufsize C.int,
	b C.double, lambda C.double,
	innerSpace C.space, window C.int,
) (C.int, *C.char) {
	var implConf = streamConf(par, bufsize, b, lambda, seed)
	var spaceConf = spaceConf(space, window, innerSpace)
	return CreateOC(implConf, spaceConf, C.I_GIVEN, data, l1, l2)
}

func streamConf(par C.int, bufsize C.int, b C.double, lambda C.double, seed C.long) streaming.Conf {

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
