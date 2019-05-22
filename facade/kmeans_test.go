package main

import (
	"testing"
)

func TestKMEANS(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2, _ = RealElemtsToArray(elemts)
	var descr, msg = KMeans(
		0, arr, l1, l2,
		0, 2, 6305689164243,
		2, 2, 3,
		0, 0,
	)

	if msg != nil {
		t.Error("message should be NULL")
		return
	}

	var algo = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Free(descr)
}
