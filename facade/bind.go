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

func Initializer(i C.initializer) (fi core.Initializer) {
	switch i {
	case C.I_RANDOM:
		fi = kmeans.RandInitializer
	case C.I_GIVEN:
		fi = kmeans.GivenInitializer
	case C.I_KMEANSPP:
		fi = kmeans.KMeansPPInitializer
	}
	return fi
}

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

func ArrayToInts(arr *C.long, l C.size_t) []int {
	var l_ = (int)(l)
	var data = make([]int, l_)
	var sh = reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  l_,
		Cap:  l_,
	}
	var slice = *(*[]int)(unsafe.Pointer(&sh))
	copy(data, slice)
	return data
}

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

func ArrayToRealElemts(arr *C.double, l1 C.size_t, l2 C.size_t) []core.Elemt {
	l1_ := (int)(l1)
	l2_ := (int)(l2)
	l_ := l1_ * l2_
	data := make([]core.Elemt, l1_)
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  l_,
		Cap:  l_,
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	for i := range data {
		data[i] = make([]float64, l2_)
		copy(data[i].([]float64), slice[i*l2_:(i+1)*l2_])
	}
	return data
}

//export FreeRealArray
func FreeRealArray(arr *C.double) {
	C.free(unsafe.Pointer(arr))
}

//export FreeIntArray
func FreeIntArray(arr *C.long) {
	C.free(unsafe.Pointer(arr))
}
