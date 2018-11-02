package main

import (
	"distclus/core"
	"golang.org/x/exp/rand"
	"testing"
)

func TestMCMC(t *testing.T) {
	var descr = MCMC(2, 2, 30, 0, 100.0, 1.0, 2.0, 1.0, 1, 2, 6305689164243)
	var algo = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	FreeMCMC(descr)
}

func TestMCMCPushRunCentroidsPredict(t *testing.T) {
	var descr = MCMC(2, 2, 30, 0, 100.0, 1.0, 2.0, 1.0, 1, 2, 6305689164243)

	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)

	MCMCPush(descr, arr, l1, l2)
	MCMCRun(descr, 0)
	MCMCClose(descr)

	var _, c1, c2 = MCMCRealCentroids(descr)

	if c1 != 2 {
		t.Error("Expected 2 got", c1)
	}

	if c2 != l2 {
		t.Error("Expected", l2, "got", c2)
	}

	var labels, l = MCMCPredict(descr, arr, l1, l2, 0)

	if l != l1 {
		t.Error("Expected", l1, "got", l)
	}

	assertLabels(ArrayToInts(labels, l), t)

	FreeMCMC(descr)
}

func makeElements() []core.Elemt {
	var rgen = rand.New(rand.NewSource(6305689164243))
	var elemts = make([]core.Elemt, 20)
	for i := range elemts {
		var elemt = make([]float64, 2)

		var shift = 2.0
		if i >= 10 {
			shift = 30.0
		}

		for j := range elemt {
			elemt[j] = rgen.Float64() + shift
		}

		elemts[i] = elemt
	}
	return elemts
}

func assertLabels(labels []int, t *testing.T) {
	var label0, label10 int
	for i, label := range labels {
		switch {
		case i == 0:
			label0 = label
		case i < 10:
			if label != label0 {
				t.Error("Expected", label0, "got", label)
			}
		case i == 10:
			label10 = label
			if label10 == label0 {
				t.Error("Expected !=", label0, "got", label)
			}
		default:
			if label != label10 {
				t.Error("Expected", label10, "got", label)
			}
		}
	}
}
