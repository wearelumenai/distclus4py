package main

//#include "bind.h"
import "C"
import (
	"distclus/core"
	"distclus/kmeans"
	"reflect"
	"unsafe"
)

// These functions bind C types to Go types and conversely.

// Initializer returns a specific OC initializer
func Initializer(i C.initializer) (fi core.Initializer) {
	switch i {
	case C.I_RANDOM:
		fi = kmeans.RandInitializer
	case C.I_GIVEN:
		fi = kmeans.GivenInitializer
	case C.I_KMEANSPP:
		fi = kmeans.PPInitializer
	}
	return
}

// Space retuns a specific OC space name
func Space(s C.space) (fs string) {
	switch s {
	case C.S_VECTORS:
		fs = "vectors"
	case C.S_COSINUS:
		fs = "cosinus"
	case C.S_SERIES:
		fs = "series"
	}
	return
}

// OC returns a specific oc name
func OC(o C.oc) (fo string) {
	switch o {
	case C.O_KMEANS:
		fo = "kmeans"
	case C.O_MCMC:
		fo = "mcmc"
	case C.O_KNN:
		fo = "knn"
	case C.O_STREAMING:
		fo = "streaming"
	}
	return
}

// Initializer returns a specific OC initializer
func Figure(figure C.figure) (name string) {
	switch figure {
	case C.F_ITERATIONS:
		name = "iterations"
	}
	return
}

// IntsToArray convert integers to an array
func IntsToArray(arr []int) (*C.long, C.size_t) {
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

// ArrayToInts convert an array to integers
func ArrayToInts(arr *C.long, l C.size_t) []int {
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

// RealElemtsToArray convert real elements to an array
func RealElemtsToArray(elemts []core.Elemt) (arr *C.double, l1, l2, l3 C.size_t) {
	var n1 = len(elemts)
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
	}
	l1, l2, l3 = C.size_t(n1), C.size_t(n2), C.size_t(n3)
	return
}

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

// ArrayToRealElemts free an array of real elements
func ArrayToRealElemts(arr *C.double, l1, l2, l3 C.size_t) []core.Elemt {
	var n1, n2, n3 = (int)(l1), (int)(l2), (int)(l3)
	var n int
	var data = make([]core.Elemt, n1)
	if n3 > 0 {
		n = n1 * n2 * n3
		var slice = makeSlice(arr, n)
		copyTo3D(data, slice, n2, n3)
	} else {
		n = n1 * n2
		var slice = makeSlice(arr, n)
		copyTo2D(data, slice, n2)
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
		for j := range data {
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

// FreeRealArray free an array of reals
//export FreeRealArray
func FreeRealArray(arr *C.double) {
	C.free(unsafe.Pointer(arr))
}

// FreeIntArray free an array of integers
//export FreeIntArray
func FreeIntArray(arr *C.long) {
	C.free(unsafe.Pointer(arr))
}
