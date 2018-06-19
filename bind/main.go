package main

// #include <stdlib.h>
// #include <stdint.h>
// #include <string.h>
// typedef enum {I_RANDOM, I_GIVIEN, I_KMEANSPP} initializer;
import "C"
import (
	"unsafe"
	"reflect"
	"distclus/algo"
	"distclus/core"
)

func InitConvert(i C.initializer) (fi algo.Initializer) {
	switch i {
	case C.I_RANDOM:
		fi = algo.RandInitializer
	case C.I_GIVIEN:
		fi = algo.GivenInitializer
	case C.I_KMEANSPP:
		fi = algo.KmeansPPInitializer
	}
	return fi
}

func SliceArr1D(arr []float64) (*C.double, C.size_t) {
	l := len(arr)
	mem := (*C.double)(C.calloc(C.size_t(l), C.sizeof_double))
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(mem)),
		Len:  (int)(l),
		Cap:  (int)(l),
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	for i := range slice {
		slice[i] = arr[i]
	}
	return mem, C.size_t(l)
}

func SliceArr2D(arr []core.Elemt) (*C.double, C.size_t, C.size_t) {
	l1 := len(arr)
	var arr0 = arr[0].([]float64)
	l2 := len(arr0)
	l := l1 * l2
	mem := (*C.double)(C.calloc(C.size_t(l), C.sizeof_double))
	sh := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(mem)),
		Len:  (int)(l),
		Cap:  (int)(l),
	}
	slice := *(*[]float64)(unsafe.Pointer(&sh))
	for i := range arr {
		for j, val := range arr[i].([]float64) {
			slice[i*l2+j] = val
		}
	}
	return mem, C.size_t(l1), C.size_t(l2)
}

func ArrSlice1D(arr *C.double, l C.size_t) ([]float64) {
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(arr)),
		Len:  (int)(l),
		Cap:  (int)(l),
	}
	return *(*[]float64)(unsafe.Pointer(&hdr))
}

func ArrSlice2D(arr *C.double, l1 C.size_t, l2 C.size_t) ([][]float64) {
	l1_ := (int)(l1)
	l2_ := (int)(l2)
	s := make([][]float64, l1_)
	for i := range s {
		e := unsafe.Pointer(uintptr(unsafe.Pointer(arr)) + uintptr(i*l2_*C.sizeof_double))
		sh := reflect.SliceHeader{
			Data: uintptr(e),
			Len:  l2_,
			Cap:  l2_,
		}
		s[i] = *(*[]float64)(unsafe.Pointer(&sh))
	}
	return s
}

//export Kmeans
func Kmeans(data *C.double, l1, l2 C.size_t, k C.int, iter C.int, initializer C.initializer) (*C.double, C.size_t, C.size_t) {
	var conf = algo.KMeansConf{Iter: (int)(iter), K: (int)(k), Space: core.RealSpace{}}
	var km = algo.NewKMeans(conf, InitConvert(initializer))
	for _, elemt := range ArrSlice2D(data, l1, l2) {
		km.Push(elemt)
	}
	km.Run(false)
	var res, err = km.Centroids()
	if err != nil {
		panic(err)
	}
	return SliceArr2D(res)
}

//func TestGo(arr *C.double, l1 C.size_t, l2 C.size_t) (*C.double, C.size_t, C.size_t) {
//	s := ArrSlice2D(arr, l1, l2)
//	for i := range s {
//		for j, v := range s[i] {
//			println(i, j, v)
//		}
//	}
//	return SliceArr2D(s)
//	//s := ArrSlice1D(arr, l1)
//	//for i, v := range s {
//	//	println(i, v)
//	//}
//	//return SliceArr1D(s)
//}

func main() {
}
