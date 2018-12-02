package main

import (
	"distclus/core"
	"sync"
)

// These functions maintain a global table for allocated algorithms.
// Algorithm may be registered, unregistered or accessed.
// The table is a map accessed with a descriptor

// AlgorithmDescr is the algorithm description type
type AlgorithmDescr = int

var lock = &sync.Mutex{}
var sequence = 1
var table = make(map[int]core.OnlineClust)

// RegisterAlgorithm registers an algorithm and retuns a descriptor
func RegisterAlgorithm(algo core.OnlineClust) AlgorithmDescr {
	lock.Lock()
	defer lock.Unlock()

	var descr = sequence
	sequence++
	table[descr] = algo
	return descr
}

// UnregisterAlgorithm unregisters an OC
func UnregisterAlgorithm(descr AlgorithmDescr) {
	lock.Lock()
	defer lock.Unlock()

	delete(table, descr)
}

// GetAlgorithm returns an OC
func GetAlgorithm(descr AlgorithmDescr) core.OnlineClust {
	lock.Lock()
	defer lock.Unlock()

	return table[descr]
}

func main() {
}
