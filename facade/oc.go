// Package main is intended to export a native library that provides a facade
// to the distclus (https://github.com/wearelumenai/distclus) library.
// It is used by the Python module https://github.com/wearelumenai/distclus4py/tree/master/distclus
// to proxy the algorithms offered by distclus.
package main

//#include "bind.h"
import "C"
import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

// Combine combines two elements with respective weight
//export Combine
func Combine(
	descr C.int,
	data1 *C.double, l11 C.size_t, l21 C.size_t, l31 C.size_t, weight1 C.int,
	data2 *C.double, l12 C.size_t, l22 C.size_t, l32 C.size_t, weight2 C.int,
) (combined *C.double, c1 C.size_t, c2 C.size_t, c3 C.size_t, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemt1 = ArrayToRealElemt(data1, l11, l21, l31)
	var elemt2 = ArrayToRealElemt(data2, l12, l22, l32)
	var _, space = GetAlgorithm((AlgorithmDescr)(descr))
	var combine = space.Combine(elemt1, int(weight1), elemt2, int(weight2))
	combined, c1, c2, c3 = realElemtToArray(combine)
	return
}

// Dist get space distance between two elemnts
//export Dist
func Dist(
	descr C.int,
	data1 *C.double, l11 C.size_t, l21 C.size_t, l31 C.size_t,
	data2 *C.double, l12 C.size_t, l22 C.size_t, l32 C.size_t,
) (dist C.double, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemt1 = ArrayToRealElemt(data1, l11, l21, l31)
	var elemt2 = ArrayToRealElemt(data2, l12, l22, l32)
	var _, space = GetAlgorithm((AlgorithmDescr)(descr))
	dist = (C.double)(space.Dist(elemt1, elemt2))
	return
}

// Push pushes an array of element to the algorithm corresponding to the given descriptor
//export Push
func Push(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	for i := range elemts {
		var err = algo.Push(elemts[i])
		if err != nil {
			errMsg = setError((AlgorithmDescr)(descr), err.Error())
			return
		}
	}
	return
}

// Play runs the algorithm corresponding to the given descriptor
//export Play
func Play(descr C.int, iter C.int, duration C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Play(int(iter), time.Duration(duration*1e9))
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Predict returns the centroids and labels for the input data
// from the algorithm corresponding to the given descriptor
//export Predict
func Predict(descr C.int, data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t) (labels *C.long, n1 C.size_t, centers *C.double, c1 C.size_t, c2 C.size_t, c3 C.size_t, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var elemts = ArrayToRealElemts(data, l1, l2, l3)
	var algo, space = GetAlgorithm((AlgorithmDescr)(descr))
	var centroids, err = algo.Centroids()

	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	} else {
		var predictions, _ = centroids.ParMapLabel(elemts, space, runtime.NumCPU())
		labels, n1 = intsToArray(predictions)
		centers, c1, c2, c3 = realElemtsToArray(centroids)
	}
	return
}

// Centroids returns the centroids
// from the algorithm corresponding to the given descriptor
//export Centroids
func Centroids(descr C.int) (data *C.double, l1 C.size_t, l2 C.size_t, l3 C.size_t, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var centroids, err = algo.Centroids()

	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	} else {
		data, l1, l2, l3 = realElemtsToArray(centroids)
	}

	return
}

// RuntimeFigure returns runtime figures
// from the algorithm corresponding to the given descriptor
//export RuntimeFigure
func RuntimeFigure(descr C.int, fig C.figure) (value C.double, errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var rfigures, err = algo.RuntimeFigures()

	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	} else {
		value = (C.double)(rfigures[figure(fig)])
	}

	return
}

// Stop terminates the algorithm corresponding to the given descriptor
//export Stop
func Stop(descr C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Stop()
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Wait waits the algorithm corresponding to the given descriptor
//export Wait
func Wait(descr C.int, iter C.int, duration C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Wait(int(iter), time.Duration(duration*1e9))
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Pause pauses the algorithm corresponding to the given descriptor
//export Pause
func Pause(descr C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Pause()
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Init initialises the algorithm corresponding to the given descriptor
//export Init
func Init(descr C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Init()
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Batch batches the algorithm corresponding to the given descriptor
//export Batch
func Batch(descr C.int, iter C.int, duration C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Batch(int(iter), time.Duration(duration*1e9))
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Close batches the algorithm corresponding to the given descriptor
//export Close
func Close(descr C.int) (errMsg *C.char) {
	defer handlePanic(descr, &errMsg)
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	var err = algo.Close()
	if err != nil {
		errMsg = setError((AlgorithmDescr)(descr), err.Error())
	}
	return
}

// Status return the status of the algorithm corresponding to the given descriptor
//export Status
func Status(descr C.int) *C.char {
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	return C.CString(algo.Status().String())
}

// Alive true iif the algorithm corresponding to the given descriptor is running
//export Alive
func Alive(descr C.int) C.int {
	var algo, _ = GetAlgorithm((AlgorithmDescr)(descr))
	if algo.Alive() {
		return C.int(1)
	}
	return C.int(0)
}

// Free terminates the algorithm corresponding to the given descriptor
// and free allocated resources
//export Free
func Free(descr C.int) {
	Stop(descr)
	UnregisterAlgorithm((AlgorithmDescr)(descr))
}

func handlePanic(descr C.int, msg **C.char) {
	var f string
	for i := 1; ; i++ {
		var _, fn, line, ok = runtime.Caller(i)
		if strings.Contains(fn, "distclus") || !ok {
			f = fmt.Sprintf("%s:%d %%v", fn, line)
			break
		}
	}
	if r := recover(); r != nil {
		switch v := r.(type) {
		case error:
			*msg = setError((AlgorithmDescr)(descr), fmt.Sprintf(f, v.Error()))
		case string:
			sprintf := fmt.Sprintf(f, v)
			*msg = setError((AlgorithmDescr)(descr), sprintf)
		default:
			*msg = setError((AlgorithmDescr)(descr), fmt.Sprintf(f, "unexpected error"))
		}
	}
}
