package main

import (
	"testing"
)

func TestMCMC(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var descr, msg = MCMC(
		0, 0, 2, 6305689164243,
		arr, l1, l2,
		2, 2, 3, 30, 10000,
		100.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	if msg != nil {
		t.Error("message should be NULL")
	}

	var algo = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Free(descr)
}
