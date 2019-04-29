package main

import (
	"strings"
	"testing"
)

func TestCreateError(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var _, msg = MCMC(
		0, 0, 2, 6305689164243,
		arr, l1, l2,
		2, 0, 3, 30, 10000,
		100.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	if m := goString(msg); !strings.HasPrefix(m, "Illegal") {
		t.Error("an error message was expected")
	}
}

func TestRun(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var descr, _ = MCMC(
		0, 0, 2, 6305689164243,
		arr, l1, l2,
		2, 2, 3, 100000000, 10000,
		1.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	var _, _, _, msgErr = RealCentroids(descr)
	if m := goString(msgErr); m != "clustering not started" {
		t.Error("expected error", m)
	}

	elemts = makeElements()
	arr, l1, l2 = RealElemtsToArray(elemts)

	var msgPush = Push(descr, arr, l1, l2)
	if msgPush != nil {
		t.Error("unexpected error")
	}

	var msgRun = Run(descr, 1)
	if msgRun != nil {
		t.Error("unexpected error")
	}

	Close(descr)

	var centroids, c1, c2, msgCentroids = RealCentroids(descr)
	if msgCentroids != nil {
		t.Error("unexpected error")
	}

	if c1 != 2 {
		t.Error("Expected 2 got", c1)
	}

	if c2 != l2 {
		t.Error("Expected", l2, "got", c2)
	}

	assertCentroids(ArrayToRealElemts(centroids, c1, c2), t)

	var labels, l, msgPredict = Predict(descr, arr, l1, l2)
	if msgPredict != nil {
		t.Error("unexpected error")
	}

	if l != l1 {
		t.Error("Expected", l1, "got", l)
	}

	assertLabels(ArrayToInts(labels, l), t)

	Free(descr)
}
