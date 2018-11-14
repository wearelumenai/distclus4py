package main

//#include "bind.h"
import "C"

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

//export Push
func Push(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t) {
	var elemts = ArrayToRealElemts(data, l1, l2)
	var algo = GetAlgorithm((int)(descr))
	for i := range elemts {
		algo.Push(elemts[i])
	}
}

//export Run
func Run(descr C.int, async C.int) {
	var algo = GetAlgorithm((int)(descr))
	algo.Run((int)(async) != 0)
}

//export Predict
func Predict(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t, push C.int) (*C.long, C.size_t) {
	var elemts = ArrayToRealElemts(data, l1, l2)
	var algo = GetAlgorithm((int)(descr))

	var predictions = make([]int, len(elemts))
	for i := range elemts {
		var _, label, err = algo.Predict(elemts[i], (int)(push) != 0)

		if err != nil {
			panic(err)
		}

		predictions[i] = label
	}

	return IntsToArray(predictions)
}

//export RealCentroids
func RealCentroids(descr C.int) (*C.double, C.size_t, C.size_t) {
	var algo = GetAlgorithm((int)(descr))
	var centroids, err = algo.Centroids()

	if err != nil {
		panic(err)
	}

	return RealElemtsToArray(centroids)
}

//export Close
func Close(descr C.int) {
	var algo = GetAlgorithm((int)(descr))
	algo.Close()
}

//export Free
func Free(descr C.int) {
	Close(descr)
	UnregisterAlgorithm((int)(descr))
}
