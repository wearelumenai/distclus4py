package main

//#include "bind.h"
import "C"

// These functions act as a facade on a MCMC algorithm instance.
// The facade works with C input and output parameters that are bound to Go types inside the functions.
// The real MCMC instance is stored in a global table and accessed with a descriptor.

// Push push an element in a specific algorithm
//export Push
func Push(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var algo = GetAlgorithm((AlgorithmDescr)(descr))
	for i := range elemts {
		var err = algo.Push(elemts[i])
		if err != nil {
			errMsg = setError((AlgorithmDescr)(descr), err.Error())
			return
		}
	}
	return
}

// Run executes a specific algorithm
//export Run
func Run(descr C.int, async C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Run((AlgorithmDescr)(async) != 0)
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Predict predicts an element in a specific algorithm
//export Predict
func Predict(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t) (labels *C.long, n1 C.size_t, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var algo = GetAlgorithm((AlgorithmDescr)(descr))

	var predictions = make([]int, len(elemts))
	for i := range elemts {
		var _, label, err = algo.Predict(elemts[i])

		if err != nil {
			errMsg = setError((AlgorithmDescr)(descr), err.Error())
			return
		}

		predictions[i] = label
	}

	labels, n1 = IntsToArray(predictions)
	return
}

// Centroids returns specific on centroids
//export Centroids
func Centroids(descr C.int) (data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo = GetAlgorithm((AlgorithmDescr)(descr))
	var centroids, err = algo.Centroids()

	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	} else {
		data, l1, l2, l3 = RealElemtsToArray(centroids)
	}

	return
}

// Iterations returns number of iterations per execution
//export RuntimeFigure
func RuntimeFigure(descr C.int, figure C.figure) (value C.double, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo = GetAlgorithm((AlgorithmDescr)(descr))
	var figures, err = algo.RuntimeFigures()

	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	} else {
		value = (C.double)(figures[Figure(figure)])
	}

	return
}

// Close terminates an oc execution
//export Close
func Close(descr C.int) {
	var algo = GetAlgorithm((AlgorithmDescr)(descr))
	_ = algo.Close()
}

// Free terminates an oc execution and unregister it from global registry
//export Free
func Free(descr C.int) {
	Close(descr)
	UnregisterAlgorithm((AlgorithmDescr)(descr))
}

func handlePanic(descr C.int, msg **C.char) {
	if r := recover(); r != nil {
		switch v := r.(type) {
		case error:
			*msg = setError((AlgorithmDescr)(descr), v.Error())
		case string:
			*msg = setError((AlgorithmDescr)(descr), v)
		default:
			*msg = setError((AlgorithmDescr)(descr), "unexpected error")
		}
	}
}
