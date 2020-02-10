package main

import (
	"distclus/core"
	"math"
	"strings"
	"testing"
	"time"
)

func TestCreateError(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var _, msg = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 0, 6305689164243,
		2, 0, 3, 30, 1000,
		10.0, 1.0, 2.0, 1.0,
		0, 0,
		0, 0, 0, 0,
	)

	if m := goString(msg); !strings.Contains(m, "Illegal") {
		t.Error("an error message was expected")
	}
}

func TestRunVectors(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var descr, _ = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 0, 6305689164243,
		2, 2, 3, 1000, 10000,
		10.0, 1.0, 2.0, 1.0,
		0, 0,
		0, 0, 0, 0,
	)

	assertAlgo(t, (int)(descr), makeVectors())
	Free(descr)
}

func TestInitFromDescr(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var descr0, _ = MCMC(
		0, arr, l1, l2, l3,
		0, 2, 0, 6305689164243,
		2, 2, 3, 1000, 10000,
		10.0, 1.0, 2.0, 1.0,
		0, 0,
		0, 0, 0, 0,
	)
	_ = Push(descr0, arr, l1, l2, l3)
	_ = Batch(descr0, 0, 0)
	var centroids0, c01, c02, c03, _ = Centroids(descr0)
	var descr1, _ = MCMC(
		0, arr, l1, l2, l3,
		0, 3, descr0, 6305689164243,
		2, 2, 3, 20, 10000,
		10.0, 1.0, 2.0, 1.0,
		0, 0,
		0, 0, 0, 0,
	)
	Free(descr0)
	_ = Batch(descr1, 0, 0)
	var centroids1, c11, c12, c13, _ = Centroids(descr1)

	var elemts0 = ArrayToRealElemts(centroids0, c01, c02, c03)
	var elemts1 = ArrayToRealElemts(centroids1, c11, c12, c13)

	var precision = math.Pow(10, 14)

	for i, elemt := range elemts0 {
		var values = elemt.([]float64)
		for j := range values {
			var zero = int(values[j] * precision)
			var one = int(elemts1[i].([]float64)[j] * precision)
			if zero != one {
				t.Error("error in initialization from descriptor")
			}
		}
	}
}

func TestRunSeries(t *testing.T) {
	var arr, l1, l2, l3 = realElemtsToArray(make([]core.Elemt, 0))
	var descr, _ = MCMC(
		2, arr, l1, l2, l3,
		0, 2, 0, 6305689164243,
		2, 2, 3, 1000, 10000,
		10.0, 1.0, 2.0, 1.0,
		0, 0,
		0, 0, 0, 0,
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
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var msgPush = Push(descr, arr, l1, l2, l3)
	if msgPush != nil {
		t.Error("unexpected error")
	}
	var msgRun = Play(descr, 0, 0)
	if msgRun != nil {
		t.Error("unexpected error")
	}
	time.Sleep(500 * time.Millisecond)
	Stop(descr)
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
	assertLabels(arrayToInts(labels, l), t)
	var iters, msgFig = RuntimeFigure(descr, 0)
	if msgFig != nil {
		t.Error("unexpected error")
	}
	if iters < 50 {
		t.Error("Expected more iterations got", iters)
	}
}
