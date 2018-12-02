package main

import (
	"distclus/core"
	"distclus/kmeans"
	"reflect"
	"testing"
	"unsafe"
)

func TestIntsToArray(t *testing.T) {
	var arr, l1 = IntsToArray([]int{1, 1, 0, 0, 0, 1})

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
	var arr, l1 = IntsToArray(expected)
	var values = ArrayToInts(arr, l1)

	if !reflect.DeepEqual(expected, values) {
		t.Error("Expected", expected, "got", values)
	}

	FreeIntArray(arr)
}

func TestRealElemtsToArray(t *testing.T) {
	var arr, l1, l2 = RealElemtsToArray([]core.Elemt{[]float64{.1, .2}, []float64{.3, .4}, []float64{.5, .6}})

	if l1 != 3 {
		t.Error("Expected 3 got", l1)
	}

	if l2 != 2 {
		t.Error("Expected 2 got", l2)
	}

	var p = uintptr(unsafe.Pointer(arr))
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

	FreeRealArray(arr)
}

func TestArrayToRealElemts(t *testing.T) {
	expected := []core.Elemt{[]float64{.1, .2}, []float64{.3, .4}, []float64{.5, .6}}
	var arr, l1, l2 = RealElemtsToArray(expected)
	var elemts = ArrayToRealElemts(arr, l1, l2)

	if !reflect.DeepEqual(expected, elemts) {
		t.Error("Expected", expected, "got", elemts)
	}

	FreeRealArray(arr)
}

func TestInitConvert(t *testing.T) {
	assertInitializer(t, Initializer(0), kmeans.RandInitializer)
	assertInitializer(t, Initializer(1), kmeans.GivenInitializer)
	assertInitializer(t, Initializer(2), kmeans.PPInitializer)
}

func assertInitializer(t *testing.T, expected core.Initializer, actual core.Initializer) {
	var vexpected = reflect.ValueOf(expected)
	var vactual = reflect.ValueOf(actual)
	if vactual != vexpected {
		t.Error("Expected", vexpected, "got", vactual)
	}
}

func TestOC(t *testing.T) {
	assertOC(t, OC(0), "kmeans")
	assertOC(t, OC(1), "mcmc")
	assertOC(t, OC(2), "knn")
	assertOC(t, OC(3), "streaming")
}

func assertOC(t *testing.T, actual string, expected string) {
	if actual != expected {
		t.Error("Expected", expected, "got", actual)
	}
}

func TestSpace(t *testing.T) {
	assertSpace(t, Space(0), "real")
	assertSpace(t, Space(1), "complex")
	assertSpace(t, Space(2), "series")
}

func assertSpace(t *testing.T, actual string, expected string) {
	if actual != expected {
		t.Error("Expected", expected, "got", actual)
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
