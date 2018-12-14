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
func RealElemtsToArray(arr []core.Elemt) (*C.double, C.size_t, C.size_t) {
	l1 := len(arr)
	l2 := len(arr[0].([]float64))
	l := l1 * l2
	mem := (*C.double)(C.calloc(C.size_t(l), C.sizeof_double))
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(mem)),
		Len:  l,
		Cap:  l,
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	for i := range arr {
		copy(slice[i*l2:(i+1)*l2], arr[i].([]float64))
	}
	return mem, C.size_t(l1), C.size_t(l2)
}

// ArrayToRealElemts free an array of real elements
func ArrayToRealElemts(arr *C.double, l1 C.size_t, l2 C.size_t) []core.Elemt {
	l1s := (int)(l1)
	l2s := (int)(l2)
	ls := l1s * l2s
	data := make([]core.Elemt, l1s)
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  ls,
		Cap:  ls,
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	for i := range data {
		data[i] = make([]float64, l2s)
		copy(data[i].([]float64), slice[i*l2s:(i+1)*l2s])
	}
	return data
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
