package main

//#include "bind.h"
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/cosinus"
	"github.com/wearelumenai/distclus/dtw"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"
	"github.com/wearelumenai/distclus/streaming"

	"github.com/pkg/errors"
)

// initializer returns a specific OC initializer
func initializer(i C.initializer, initDescr C.int) (fi core.Initializer) {
	switch i {
	case C.I_RANDOM:
		fi = kmeans.RandInitializer
	case C.I_GIVEN:
		fi = kmeans.GivenInitializer
	case C.I_KMEANS_PP:
		fi = kmeans.PPInitializer
	case C.I_OC:
		fi = descrInitializer(initDescr)
	}
	return
}

func descrInitializer(initDescr C.int) core.Initializer {
	var algo, _ = GetAlgorithm((int)(initDescr))
	var centroids = algo.Centroids()
	return centroids.Initializer
}

// figure converts a C figure enum to a figure constant
func figure(figure C.figure) (name string) {
	switch figure {
	case C.F_ITERATIONS:
		name = core.Iterations
	case C.F_PUSHED_DATA:
		name = core.PushedData
	case C.F_DURATION:
		name = core.Duration
	case C.F_LAST_DATA_TIME:
		name = core.LastDataTime
	case C.F_ACCEPTATIONS:
		name = mcmc.Acceptations
	case C.F_LAMBDA:
		name = mcmc.Lambda
	case C.F_RHO:
		name = mcmc.Rho
	case C.F_RGIBBS:
		name = mcmc.RGibbs
	case C.F_TIME:
		name = mcmc.Time
	case C.F_MAX_DISTANCE:
		name = streaming.MaxDistance
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
		case float64:
			slice, arr = makeArray(n1)
			for i, value := range elemts {
				slice[i] = value.(float64)
			}
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

// realElemtToArray converts core.Elemt Go Slice to a C double array.
// Element type must be []float64 or [][]float64.
// It returns a pointer to the C array in row major layout
// and the size of its 3 dimensions, if element type os []float64 then l3 = 0.
func realElemtToArray(elemt core.Elemt) (arr *C.double, l1, l2, l3 C.size_t) {
	var slice []float64
	var n1, n2, n3 int
	switch elemt.(type) {
	case []float64:
		var data = elemt.([]float64)
		n1 = len(data)
		slice, arr = makeArray(n1)
		for i, value := range data {
			slice[i] = value
		}
	case [][]float64:
		var data = elemt.([][]float64)
		n1, n2 = len(data), len(data[0])
		slice, arr = makeArray(n1 * n2)
		for i, d := range data {
			copy(slice[i*n2:(i+1)*n2], d)
		}
	case [][][]float64:
		var data = elemt.([][][]float64)
		n1, n2, n3 = len(data), len(data[0]), len(data[0][0])
		slice, arr = makeArray(n1 * n2 * n3)
		for i, d := range data {
			for j, v := range d {
				copy(slice[i*n2*n3+j*n3:i*n2*n3+(j+1)*n3], v)
			}
		}
	default:
		panic(errUnauthorizedType)
	}
	l1, l2, l3 = C.size_t(n1), C.size_t(n2), C.size_t(n3)
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
		} else if n2 > 0 {
			n = n1 * n2
			var slice = makeSlice(arr, n)
			copyTo2D(data, slice, n2)
		} else {
			var slice = makeSlice(arr, n1)
			for i, value := range slice {
				data[i] = value
			}
		}
	}
	return data
}

// ArrayToRealElemt convert a C double array with the given 3 dimensions size
// to element Go slice. If l3 = 0 element type is []float64 otherwise [][]float64.
func ArrayToRealElemt(arr *C.double, l1, l2, l3 C.size_t) (elemt core.Elemt) {
	var n1, n2, n3 = (int)(l1), (int)(l2), (int)(l3)
	var n int
	if n1 > 0 {
		if n3 > 0 {
			n = n1 * n2 * n3
			var slice = makeSlice(arr, n)
			var data = make([][][]float64, n1)
			for i := range data {
				var sl = make([][]float64, n2)
				for j := range sl {
					sl[j] = make([]float64, n3)
					copy(sl[j], slice[i*n2*n3+j*n3:i*n2*n3+(j+1)*n3])
				}
				data[i] = sl
			}
			elemt = data
		} else if n2 > 0 {
			n = n1 * n2
			var slice = makeSlice(arr, n)
			var data = make([][]float64, n1)
			for i := range data {
				var sl = make([]float64, n2)
				copy(sl, slice[i*n2:(i+1)*n2])
				data[i] = sl
			}
			elemt = data
		} else {
			var data = make([]float64, n1)
			var slice = makeSlice(arr, n1)
			for i, value := range slice {
				data[i] = value
			}
			elemt = data
		}
	}
	return
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

func getSpace(spaceName C.space, window C.int, innerSpace C.space) core.Space {
	switch spaceName {
	case C.S_SERIES:
		var conf = dtw.Conf{
			InnerSpace: getSpace(innerSpace, 0, 0).(dtw.PointSpace),
			Window:     (int)(window),
		}
		return dtw.NewSpace(conf)
	case C.S_EUCLID:
		return euclid.NewSpace()
	case C.S_COSINUS:
		return cosinus.NewSpace()
	default:
		panic(fmt.Sprintf("unknown space %v", spaceName))
	}
}
