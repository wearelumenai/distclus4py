package main

import (
	"distclus/core"
	"strings"
	"testing"
)

func TestCreateError(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = RealElemtsToArray(elemts)
	var _, msg = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 6305689164243,
		2, 0, 3, 30, 10000,
		100.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	if m := goString(msg); !strings.HasPrefix(m, "Illegal") {
		t.Error("an error message was expected")
	}
}

func TestRunVectors(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = RealElemtsToArray(elemts)
	var descr, _ = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 6305689164243,
		2, 2, 3, 100000000, 10000,
		1.0, 1.0, 2.0, 1.0,
		1,
		0, 0,
	)

	assertAlgo(t, (int)(descr), makeVectors())
	Free(descr)
}

func TestRunSeries(t *testing.T) {
	var arr, l1, l2, l3 = RealElemtsToArray(make([]core.Elemt, 0))
	var descr, _ = MCMC(
		2, arr, l1, l2, l3,
		0, 2, 6305689164243,
		2, 2, 3, 100000000, 10000,
		1.0, 1.0, 2.0, 1.0,
		1,
		1, 3,
	)
	assertAlgo(t, (int)(descr), makeSeries())
	Free(descr)
}

func assertAlgo(t *testing.T, d int, elemts []core.Elemt) {
	var descr = cInt(d)
	var _, _, _, _, msgErr = Centroids(descr)
	if m := goString(msgErr); m != "clustering not started" {
		t.Error("expected error", m)
	}
	var arr, l1, l2, l3 = RealElemtsToArray(elemts)
	var msgPush = Push(descr, arr, l1, l2, l3)
	if msgPush != nil {
		t.Error("unexpected error")
	}
	var msgRun = Run(descr, 1)
	if msgRun != nil {
		t.Error("unexpected error")
	}
	Close(descr)
	var centroids, c1, c2, c3, msgCentroids = Centroids(descr)
	if msgCentroids != nil {
		t.Error("unexpected error")
	}
	if c1 != 2 {
		t.Error("Expected 2 got", c1)
	}
	if c2 != l2 {
		t.Error("Expected", l2, "got", c2)
	}
	assertCentroids(ArrayToRealElemts(centroids, c1, c2, c3), t)
	var labels, l, _, _, _, _, msgPredict = Predict(descr, arr, l1, l2, l3)
	if msgPredict != nil {
		t.Error("unexpected error")
	}
	if l != l1 {
		t.Error("Expected", l1, "got", l)
	}
	assertLabels(ArrayToInts(labels, l), t)
}
