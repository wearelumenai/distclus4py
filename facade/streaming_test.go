package main

import (
	"testing"
)

func TestStreaming(t *testing.T) {
	var elemts = makeElements()
	var arr, l1, l2 = RealElemtsToArray(elemts)
	var descr, msg = STREAMING(
		0, 0, 0, 6305689164243,
		arr, l1, l2,
		50, .95, 3,
		0, 0,
	)

	if msg != nil {
		t.Error("message should be NULL")
		return
	}

	var algo = GetAlgorithm((int)(descr))

	if algo == nil {
		t.Error("Expected pointer got nil")
	}

	Free(descr)
}
