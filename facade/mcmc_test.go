package main

import (
	"testing"
)

func TestMCMC(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var descr = MCMC(
		0, 0, 2, 6305689164243,
		arr, l1, l2,
		2, 2, 3, 30, 10000,
		100.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)
	var algo = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Free(descr)
}

func TestMCMCPushRunCentroidsPredict(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var descr = MCMC(
		0, 0, 2, 6305689164243,
		arr, l1, l2,
		2, 2, 3, 100000000, 10000,
		1.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	elemts = makeElements()
	arr, l1, l2 = RealElemtsToArray(elemts)

	Run(descr, 1)
	Push(descr, arr, l1, l2)
	Close(descr)

	var centroids, c1, c2 = RealCentroids(descr)

	if c1 != 2 {
		t.Error("Expected 2 got", c1)
	}

	if c2 != l2 {
		t.Error("Expected", l2, "got", c2)
	}

	assertCentroids(ArrayToRealElemts(centroids, c1, c2), t)

	var labels, l = Predict(descr, arr, l1, l2)

	if l != l1 {
		t.Error("Expected", l1, "got", l)
	}

	assertLabels(ArrayToInts(labels, l), t)

	Free(descr)
}
