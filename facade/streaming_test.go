package main

import (
	"testing"
)

func TestStreaming(t *testing.T) {
	var elemts = makeVectors()
	var arr, l1, l2, l3 = RealElemtsToArray(elemts)
	var descr, msg = Streaming(
		0, arr, l1, l2, l3,
		6305689164243, 50,
		.95, 3,
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
