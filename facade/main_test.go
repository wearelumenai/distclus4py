package main

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"
)

func TestRegisterAlgorithm(t *testing.T) {
	var descr0 = makeAlgorithm()
	var descr1 = makeAlgorithm()

	if descr1 <= descr0 {
		t.Error("Expected greate than descr0 got", descr1)
	}

	UnregisterAlgorithm(descr0)
	UnregisterAlgorithm(descr1)
}

func TestGetAlgorithm(t *testing.T) {
	var descr0 = makeAlgorithm()
	var descr1 = makeAlgorithm()

	var algo0, _ = GetAlgorithm(descr0)

	if algo0 == nil {
		t.Error("Expected pointer got nil")
	}

	var algo1, _ = GetAlgorithm(descr1)

	if algo1 == nil {
		t.Error("Expected pointer got nil")
	}

	if algo0 == algo1 {
		t.Error("Expect distinct pointers got same")
	}

	UnregisterAlgorithm(descr0)
	UnregisterAlgorithm(descr1)
}

func TestUnregisterAlgorithm(t *testing.T) {
	var descr0 = makeAlgorithm()
	UnregisterAlgorithm(descr0)

	var algo0, _ = GetAlgorithm(descr0)

	if algo0 != nil {
		t.Error("Expected nil got pointer")
	}
}

func TestSetError(t *testing.T) {
	var descr0 = makeAlgorithm()
	var message4Test = "just 4 testing"
	var lastErr = setError(descr0, message4Test)
	if goString(lastErr) != message4Test {
		t.Error("expected error message")
	}
	UnregisterAlgorithm(descr0)
}

func makeAlgorithm() AlgorithmDescr {
	var elemts = makeVectors()
	var implConf = mcmc.Conf{InitK: 2}
	var init = func(elemt core.Elemt) mcmc.Distrib {
		var tConf = mcmc.MultivTConf{
			Dim: 2,
		}
		return mcmc.NewMultivT(tConf)
	}
	var distrib = mcmc.NewLateDistrib(init)
	var space = euclid.Space{}
	var oc = mcmc.NewAlgo(implConf, space, elemts, kmeans.PPInitializer, distrib)
	return RegisterAlgorithm(oc, space)
}
