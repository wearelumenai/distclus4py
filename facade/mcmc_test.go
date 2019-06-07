package main

import (
	"testing"
)

func TestMCMC(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var descr, msg = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 6305689164243,
		2, 2, 3, 30, 10000,
		100.0, 1.0, 2.0, 1.0,
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

func TestErr(t *testing.T) {
	var _, d = MCMC(
		0, nil, 0, 0, 0,
		1, 2, 654126513379,
		0, 2, 16, 5, 0,
		1.0, 0.1, 2.0, 3.0,
		0, 10,
	)
	if d != nil {
		t.Error("unexpected error")
	}
}
