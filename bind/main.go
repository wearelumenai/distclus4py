package main

// #include <stdlib.h>
// #include <stdint.h>
// #include <string.h>
// typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP} initializer;
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
	case C.I_GIVEN:
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
	var km = algo.NewKMeans(conf, InitConvert(initializer), []core.Elemt{})
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

//export MCMC
func MCMC(data *C.double, l1, l2 C.size_t, framesize, initK, mcmcIter, initIter C.int, b, amp, norm, nu C.double,
	initializer C.initializer) (*C.double, C.size_t, C.size_t) {
	var data_ = ArrSlice2D(data, l1, l2)
	var mcmcConf = algo.MCMCConf{
		Dim:      len(data_[0]), FrameSize: (int)(framesize), B: (float64)(b), Amp: (float64)(amp),
		Norm:     (float64)(norm), Nu: (float64)(nu), InitK: (int)(initK), McmcIter: (int)(mcmcIter),
		InitIter: (int)(initIter), Space: core.RealSpace{},
	}
	var distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})
	var mcmc = algo.NewMCMC(mcmcConf, distrib, InitConvert(initializer), []core.Elemt{})
	for _, elemt := range data_ {
		mcmc.Push(elemt)
	}
	mcmc.Run(false)
	var res, err = mcmc.Centroids()
	if err != nil {
		panic(err)
	}
	return SliceArr2D(res)
}

//export FreeArrPtr
func FreeArrPtr(arr *C.double){
	C.free(unsafe.Pointer(arr))
}

func main() {
}
