package main

import (
	"reflect"
	"testing"
	"unsafe"

	"github.com/wearelumenai/distclus/kmeans"

	"github.com/wearelumenai/distclus/core"
)

func TestIntsToArray(t *testing.T) {
	var arr, l1 = intsToArray([]int{1, 1, 0, 0, 0, 1})

	if l1 != 6 {
		t.Error("Expected 6 got", l1)
	}

	if *arr != 1 {
		t.Error("Expected 1 got", *arr)
	}

	FreeIntArray(arr)
}

func TestArrayToInts(t *testing.T) {
	expected := []int{1, 1, 0, 0, 0, 1}
	var arr, l1 = intsToArray(expected)
	var values = arrayToInts(arr, l1)

	if !reflect.DeepEqual(expected, values) {
		t.Error("Expected", expected, "got", values)
	}

	FreeIntArray(arr)
}

func TestRealElemtsToArray2D(t *testing.T) {
	var elemts = []core.Elemt{
		[]float64{.1, .2},
		[]float64{.3, .4},
		[]float64{.5, .6},
		[]float64{.7, .8},
	}
	var arr, l1, l2, l3 = realElemtsToArray(elemts)

	if l1 != 4 {
		t.Error("Expected 3 got", l1)
	}

	if l2 != 2 {
		t.Error("Expected 2 got", l2)
	}

	if l3 != 0 {
		t.Error("Expected 0 got", l3)
	}

	assertArray(t, unsafe.Pointer(arr))

	FreeRealArray(arr)
}

func TestRealElemtsToArray3D(t *testing.T) {
	var elemts = []core.Elemt{
		[][]float64{
			{.1, .2},
			{.3, .4},
		},
		[][]float64{
			{.5, .6},
			{.7, .8},
		},
	}
	var arr, l1, l2, l3 = realElemtsToArray(elemts)

	if l1 != 2 {
		t.Error("Expected 3 got", l1)
	}

	if l2 != 2 {
		t.Error("Expected 2 got", l2)
	}

	if l3 != 2 {
		t.Error("Expected 0 got", l3)
	}

	assertArray(t, unsafe.Pointer(arr))

	FreeRealArray(arr)
}

func assertArray(t *testing.T, arr unsafe.Pointer) {
	var p = uintptr(arr)
	check(t, .1, p)
	p = incr(p)
	check(t, .2, p)
	p = incr(p)
	check(t, .3, p)
	p = incr(p)
	check(t, .4, p)
	p = incr(p)
	check(t, .5, p)
	p = incr(p)
	check(t, .6, p)
	p = incr(p)
	check(t, .7, p)
	p = incr(p)
	check(t, .8, p)
}

func TestArrayToRealElemts2D(t *testing.T) {
	var expected = []core.Elemt{
		[]float64{.1, .2},
		[]float64{.3, .4},
		[]float64{.5, .6},
		[]float64{.7, .8},
	}
	var arr, l1, l2, l3 = realElemtsToArray(expected)
	var elemts = ArrayToRealElemts(arr, l1, l2, l3)

	if !reflect.DeepEqual(expected, elemts) {
		t.Error("Expected", expected, "got", elemts)
	}

	FreeRealArray(arr)
}

func TestArrayToRealElemts3D(t *testing.T) {
	var expected = []core.Elemt{
		[][]float64{
			{.1, .2},
			{.3, .4},
		},
		[][]float64{
			{.5, .6},
			{.7, .8},
		},
	}
	var arr, l1, l2, l3 = realElemtsToArray(expected)
	var elemts = ArrayToRealElemts(arr, l1, l2, l3)

	if !reflect.DeepEqual(expected, elemts) {
		t.Error("Expected", expected, "got", elemts)
	}

	FreeRealArray(arr)
}

func TestInitConvert(t *testing.T) {
	assertInitializer(t, initializer(0, 0), kmeans.RandInitializer)
	assertInitializer(t, initializer(1, 0), kmeans.GivenInitializer)
	assertInitializer(t, initializer(2, 0), kmeans.PPInitializer)
}

func assertInitializer(t *testing.T, expected core.Initializer, actual core.Initializer) {
	var vexpected = reflect.ValueOf(expected)
	var vactual = reflect.ValueOf(actual)
	if vactual != vexpected {
		t.Error("Expected", vexpected, "got", vactual)
	}
}

func check(t *testing.T, expected float64, pactual uintptr) {
	if star(pactual) != expected {
		t.Error("Expected .1 got", star(pactual))
	}
}

func incr(p uintptr) uintptr {
	return p + unsafe.Sizeof(0.0)
}

func star(p uintptr) float64 {
	return *(*float64)(unsafe.Pointer(p))
}
