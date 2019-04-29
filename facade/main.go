package main

//#include <stdlib.h>
//#include <string.h>
import "C"
import (
	"distclus/core"
	"sync"
	"unsafe"
)

// These functions maintain a global table for allocated algorithms.
// Algorithm may be registered, unregistered or accessed.
// The table is a map accessed with a descriptor

// AlgorithmDescr is the algorithm description type
type AlgorithmDescr = int

const errorMsgSize = 100

type container struct {
	algo    core.OnlineClust
	lastErr *C.char
}

var lock = &sync.Mutex{}
var sequence = 1
var table = map[int]container{
	0: newContainer(nil),
}

// RegisterAlgorithm registers an algorithm and retuns a descriptor
func RegisterAlgorithm(algo core.OnlineClust) AlgorithmDescr {
	lock.Lock()
	defer lock.Unlock()

	var descr = sequence
	sequence++
	table[descr] = newContainer(algo)
	return descr
}

func newContainer(algo core.OnlineClust) container {
	return container{
		algo:    algo,
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
func GetAlgorithm(descr AlgorithmDescr) core.OnlineClust {
	lock.Lock()
	defer lock.Unlock()
	return table[descr].algo
}

func setError(descr AlgorithmDescr, errMsg string) *C.char {
	lock.Lock()
	defer lock.Unlock()
	var container = table[descr]
	var cerr = C.CString(errMsg)
	defer C.free(unsafe.Pointer(cerr))
	C.strcpy(container.lastErr, cerr, errorMsgSize)
	return container.lastErr
}

func main() {
}
