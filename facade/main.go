package main

//#include <stdlib.h>
//#include <string.h>
import "C"
import (
	"sync"
	"unsafe"

	"github.com/wearelumenai/distclus/core"
)

// These functions maintain a global table for allocated algorithms.
// Algorithm may be registered, unregistered or accessed.
// The table is a map accessed with a descriptor

// AlgorithmDescr is the algorithm description type
type AlgorithmDescr = int

const errorMsgSize = 100

type container struct {
	algo    core.OnlineClust
	space   core.Space
	lastErr *C.char
}

var lock = &sync.Mutex{}
var sequence = 1
var table = map[int]container{
	0: newContainer(nil, nil),
}

// RegisterAlgorithm registers an algorithm and retuns a descriptor
func RegisterAlgorithm(algo core.OnlineClust, space core.Space) AlgorithmDescr {
	lock.Lock()
	defer lock.Unlock()

	var descr = sequence
	sequence++
	table[descr] = newContainer(algo, space)
	return descr
}

func newContainer(algo core.OnlineClust, space core.Space) container {
	return container{
		algo:    algo,
		space:   space,
		lastErr: (*C.char)(C.calloc(errorMsgSize, C.sizeof_char)),
	}
}

// UnregisterAlgorithm unregisters an OC
func UnregisterAlgorithm(descr AlgorithmDescr) {
	lock.Lock()
	defer lock.Unlock()

	var container = table[descr]
	C.free(unsafe.Pointer(container.lastErr))
	delete(table, descr)
}

// GetAlgorithm returns an OC
func GetAlgorithm(descr AlgorithmDescr) (core.OnlineClust, core.Space) {
	lock.Lock()
	defer lock.Unlock()
	var cont = table[descr]
	return cont.algo, cont.space
}

func setError(descr AlgorithmDescr, errMsg string) *C.char {
	lock.Lock()
	defer lock.Unlock()
	var container = table[descr]
	var cerr = C.CString(errMsg)
	defer C.free(unsafe.Pointer(cerr))
	C.strcpy(container.lastErr, cerr)
	return container.lastErr
}

func main() {
}
