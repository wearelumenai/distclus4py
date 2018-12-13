package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/factory"
)

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// Push push an element in a specific algorithm
//export Push
func Push(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t) {
	var elemts = ArrayToRealElemts(data, l1, l2)
	var algo = GetAlgorithm((int)(descr))
	for i := range elemts {
		algo.Push(elemts[i])
	}
}

// Run executes a specific algorithm
//export Run
func Run(descr C.int, async C.int) {
	var algo = GetAlgorithm((int)(descr))
	algo.Run((int)(async) != 0)
}

// Predict predicts an element in a specific algorithm
//export Predict
func Predict(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t) (*C.long, C.size_t) {
	var elemts = ArrayToRealElemts(data, l1, l2)
	var algo = GetAlgorithm((int)(descr))

	var predictions = make([]int, len(elemts))
	for i := range elemts {
		var _, label, err = algo.Predict(elemts[i])

		if err != nil {
			panic(err)
		}

		predictions[i] = label
	}

	return IntsToArray(predictions)
}

// RealCentroids returns specific on centroids
//export RealCentroids
func RealCentroids(descr C.int) (*C.double, C.size_t, C.size_t) {
	var algo = GetAlgorithm((int)(descr))
	var centroids, err = algo.Centroids()

	if err != nil {
		panic(err)
	}

	return RealElemtsToArray(centroids)
}

// Close terminates an oc execution
//export Close
func Close(descr C.int) {
	var algo = GetAlgorithm((int)(descr))
	algo.Close()
}

// Free terminates an oc execution and unregister it from global registry
//export Free
func Free(descr C.int) {
	Close(descr)
	UnregisterAlgorithm((int)(descr))
}

// CreateOC creates an OC according to configurable parameters
func CreateOC(name C.oc, space C.space, conf core.Conf, initializer C.initializer) C.int {
	var ocName = OC(name)
	var spaceName = Space(space)
	var oc = factory.CreateOC(ocName, spaceName, conf, []core.Elemt{}, Initializer(initializer))
	return C.int(RegisterAlgorithm(oc))
}

// SetConf switches algorithm configuration
func SetConf(descr C.int, conf core.Conf) error {
	var algo = GetAlgorithm((int)(descr))
	return algo.SetConf(conf)
}

// SetSpace switches algorithm space
func SetSpace(descr C.int, space C.space, conf core.SpaceConf) error {
	var algo = GetAlgorithm((int)(descr))
	var spaceImpl = factory.CreateSpace(Space(space), conf)
	return algo.SetSpace(spaceImpl)
}

// Reset resets the algorithm
//export Reset
func Reset(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t,
	space C.space, par C.int, seed C.long,
	dim C.size_t, initK C.int, maxK C.int, mcmcIter C.int, framesize C.int,
	b C.double, amp C.double, norm C.double, nu C.double,
	initIter C.int,
	innerSpace C.space, window C.int,
) {
	var elemts = ArrayToRealElemts(data, l1, l2)
	var algo = GetAlgorithm((int)(descr))
	var conf = core.Conf{
		ImplConf:  mcmcConf(par, dim, initK, maxK, mcmcIter, framesize, b, amp, norm, nu, initIter, seed),
		SpaceConf: spaceConf(space, window, innerSpace),
	}
	var err = algo.Reset(conf, elemts)
	if err != nil {
		panic(err)
	}
}
