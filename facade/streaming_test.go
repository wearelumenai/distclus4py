package main

import (
	"testing"
	"time"
)

func TestStreaming(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = realElemtsToArray(elemts)
	var descr, msg = Streaming(
		0, nil, 0, 0, 0,
		6305689164243, 50,
		.5, .1,
		2., 5,
		0, 0,
		0, 0, 0, 0,
	)

	if msg != nil {
		t.Error("message should be NULL")
		return
	}

	var algo, _ = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Push(descr, arr, 1, l2, l3)
	Play(descr)
	Push(descr, arr, l1, l2, l3)
	time.Sleep(time.Second)
	var maxDistance = RuntimeFigure(descr, 2)
	if maxDistance < .1 {
		t.Error("max distance should be grater than .1")
	}
	Free(descr)
}
