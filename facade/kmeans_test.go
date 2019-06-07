package main

import (
	"testing"
)

func TestKMEANS(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var descr, msg = KMeans(
		0, arr, l1, l2, l3,
		0, 2, 6305689164243,
		2, 2, 3,
		0, 0,
	)

	if msg != nil {
		t.Error("message should be NULL")
		return
	}

	var algo, _ = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Free(descr)
}
