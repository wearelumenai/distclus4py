package main

import (
	"distclus/core"
	"sync"
)

// These functions maintain a global table for allocated algorithms.
// Algorithm may be registered, unregistered or accessed.
// The table is a map accessed with a descriptor

type AlgoritmDescr = int

var lock = &sync.Mutex{}
var sequence = 1
var table = make(map[int]core.OnlineClust)

func RegisterAlgorithm(algo core.OnlineClust) AlgoritmDescr {
	lock.Lock()
	defer lock.Unlock()

	var descr = sequence
	sequence += 1
	table[descr] = algo
	return descr
}

func UnregisterAlgorithm(descr AlgoritmDescr) {
	lock.Lock()
	defer lock.Unlock()

	delete(table, descr)
}

func GetAlgorithm(descr AlgoritmDescr) core.OnlineClust {
	lock.Lock()
	defer lock.Unlock()

	return table[descr]
}

func main() {
}
