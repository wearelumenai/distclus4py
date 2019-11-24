package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/figures"
	"distclus/kmeans"
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

// initializer returns a specific OC initializer
func initializer(i C.initializer, initDescr C.int) (fi core.Initializer) {
	switch i {
	case C.I_RANDOM:
		fi = kmeans.RandInitializer
	case C.I_GIVEN:
		fi = kmeans.GivenInitializer
	case C.I_KMEANSPP:
		fi = kmeans.PPInitializer
	case C.I_OC:
		fi = descrInitializer(initDescr)
	}
	return
}

func descrInitializer(initDescr C.int) core.Initializer {
	var algo, _ = GetAlgorithm((int)(initDescr))
	var centroids, err = algo.Centroids()
	if err != nil {
		panic(err)
	}
	return centroids.Initializer
}

// figure converts a C figure enum to a figure constant
func figure(figure C.figure) (name figures.Key) {
	switch figure {
	case C.F_ITERATIONS:
		name = figures.Iterations
	case C.F_ACCEPTATIONS:
		name = figures.Acceptations
	case C.F_MAX_DISTANCE:
		name = figures.MaxDistance
	}
	return
}

// intsToArray convert integer Go slice to a C long array.
// It returns the pointer to the C array and its size.
func intsToArray(arr []int) (*C.long, C.size_t) {
	var l = len(arr)
	var mem = (*C.long)(C.calloc(C.size_t(l), C.sizeof_long))
	var sh = reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(mem)),
		Len:  l,
		Cap:  l,
	}
	var slice = *(*[]int)(unsafe.Pointer(&sh))
	copy(slice, arr)
	return mem, C.size_t(l)
}

// arrayToInts convert a C long array with the given size to element Go slice
func arrayToInts(arr *C.long, l C.size_t) []int {
	var ls = (int)(l)
	var data = make([]int, ls)
	var sh = reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  ls,
		Cap:  ls,
	}
	var slice = *(*[]int)(unsafe.Pointer(&sh))
	copy(data, slice)
	return data
}

// realElemtsToArray converts core.Elemt Go Slice to a C double array.
// Element type must be []float64 or [][]float64.
// It returns a pointer to the C array in row major layout
// and the size of its 3 dimensions, if element type os []float64 then l3 = 0.
func realElemtsToArray(elemts []core.Elemt) (arr *C.double, l1, l2, l3 C.size_t) {
	var n1 = len(elemts)
	if n1 > 0 {
		var n2, n3 int
		var slice []float64
		switch e := elemts[0].(type) {
		case []float64:
			n2 = len(e)
			slice, arr = makeArray(n1 * n2)
			copyFrom2D(slice, elemts, n2)
		case [][]float64:
			n2, n3 = len(e), len(e[0])
			slice, arr = makeArray(n1 * n2 * n3)
			copyFrom3D(slice, elemts, n2, n3)
		default:
			panic(errUnauthorizedType)
		}
		l1, l2, l3 = C.size_t(n1), C.size_t(n2), C.size_t(n3)
	}
	return
}

// errUnauthorizedType describes an error when using
// an unauthorized type in slice to array conversion
var errUnauthorizedType = errors.New("unauthorized element type")

func copyFrom2D(slice []float64, elemts []core.Elemt, n2 int) {
	for i := range elemts {
		var e = elemts[i].([]float64)
		copy(slice[i*n2:(i+1)*n2], e)
	}
}

func copyFrom3D(slice []float64, elemts []core.Elemt, n2 int, n3 int) {
	for i := range elemts {
		var e = elemts[i].([][]float64)
		for j := range e {
			copy(slice[i*n2*n3+j*n3:i*n2*n3+(j+1)*n3], e[j])
		}
	}
}

func makeArray(n int) ([]float64, *C.double) {
	var arr = (*C.double)(C.calloc(C.size_t(n), C.sizeof_double))
	var sh = reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  n,
		Cap:  n,
	}
	var slice = *(*[]float64)(unsafe.Pointer(&sh))
	return slice, arr
}

// ArrayToRealElemts convert a C double array with the given 3 dimensions size
// to element Go slice. If l3 = 0 element type is []float64 otherwise [][]float64.
func ArrayToRealElemts(arr *C.double, l1, l2, l3 C.size_t) []core.Elemt {
	var n1, n2, n3 = (int)(l1), (int)(l2), (int)(l3)
	var n int
	var data = make([]core.Elemt, n1)
	if n1 > 0 {
		if n3 > 0 {
			n = n1 * n2 * n3
			var slice = makeSlice(arr, n)
			copyTo3D(data, slice, n2, n3)
		} else {
			n = n1 * n2
			var slice = makeSlice(arr, n)
			copyTo2D(data, slice, n2)
		}
	}
	return data
}

func copyTo2D(data []core.Elemt, slice []float64, n2 int) {
	for i := range data {
		var sl = make([]float64, n2)
		copy(sl, slice[i*n2:(i+1)*n2])
		data[i] = sl
	}
}

func copyTo3D(data []core.Elemt, slice []float64, n2 int, n3 int) {
	for i := range data {
		var sl = make([][]float64, n2)
		for j := range sl {
			sl[j] = make([]float64, n3)
			copy(sl[j], slice[i*n2*n3+j*n3:i*n2*n3+(j+1)*n3])
		}
		data[i] = sl
	}
}

func makeSlice(arr *C.double, n int) []float64 {
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  n,
		Cap:  n,
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	return slice
}

// FreeRealArray is a convenient function to free a C double array allocated by the facade
//export FreeRealArray
func FreeRealArray(arr *C.double) {
	C.free(unsafe.Pointer(arr))
}

// FreeIntArray is a convenient function to free a C long array allocated by the facade
//export FreeIntArray
func FreeIntArray(arr *C.long) {
	C.free(unsafe.Pointer(arr))
}
