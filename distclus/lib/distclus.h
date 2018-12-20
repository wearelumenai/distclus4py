/* Code generated by cmd/cgo; DO NOT EDIT. */

/* package distclus4py/facade */


#line 1 "cgo-builtin-prolog"

#include <stddef.h> /* for ptrdiff_t below */

#ifndef GO_CGO_EXPORT_PROLOGUE_H
#define GO_CGO_EXPORT_PROLOGUE_H

typedef struct { const char *p; ptrdiff_t n; } _GoString_;

#endif

/* Start of preamble from import "C" comments.  */


#line 3 "bind.go"
#include "bind.h"

#line 1 "cgo-generated-wrapper"

#line 3 "kmeans.go"
#include "bind.h"

#line 1 "cgo-generated-wrapper"

#line 3 "mcmc.go"
#include "bind.h"

#line 1 "cgo-generated-wrapper"

#line 3 "oc.go"
#include "bind.h"

#line 1 "cgo-generated-wrapper"


/* End of preamble from import "C" comments.  */


/* Start of boilerplate cgo prologue.  */
#line 1 "cgo-gcc-export-header-prolog"

#ifndef GO_CGO_PROLOGUE_H
#define GO_CGO_PROLOGUE_H

typedef signed char GoInt8;
typedef unsigned char GoUint8;
typedef short GoInt16;
typedef unsigned short GoUint16;
typedef int GoInt32;
typedef unsigned int GoUint32;
typedef long long GoInt64;
typedef unsigned long long GoUint64;
typedef GoInt64 GoInt;
typedef GoUint64 GoUint;
typedef __SIZE_TYPE__ GoUintptr;
typedef float GoFloat32;
typedef double GoFloat64;
typedef float _Complex GoComplex64;
typedef double _Complex GoComplex128;

/*
  static assertion to make sure the file is being used on architecture
  at least with matching size of GoInt.
*/
typedef char _check_for_64_bit_pointer_matching_GoInt[sizeof(void*)==64/8 ? 1:-1];

typedef _GoString_ GoString;
typedef void *GoMap;
typedef void *GoChan;
typedef struct { void *t; void *v; } GoInterface;
typedef struct { void *data; GoInt len; GoInt cap; } GoSlice;

#endif

/* End of boilerplate cgo prologue.  */

#ifdef __cplusplus
extern "C" {
#endif


// FreeRealArray free an array of reals

extern void FreeRealArray(double* p0);

// FreeIntArray free an array of integers

extern void FreeIntArray(long int* p0);

// KMEANS algorithm

extern int KMEANS(space p0, int p1, initializer p2, long int p3, double* p4, size_t p5, size_t p6, int p7, int p8, int p9, space p10, int p11);

// MCMC algorithm

extern int MCMC(space p0, int p1, initializer p2, long int p3, double* p4, size_t p5, size_t p6, size_t p7, int p8, int p9, int p10, int p11, double p12, double p13, double p14, double p15, int p16, space p17, int p18);

// Push push an element in a specific algorithm

extern void Push(int p0, double* p1, size_t p2, size_t p3);

// Run executes a specific algorithm

extern int Run(int p0, int p1);

/* Return type for Predict */
struct Predict_return {
	long int* r0;
	size_t r1;
};

// Predict predicts an element in a specific algorithm

extern struct Predict_return Predict(int p0, double* p1, size_t p2, size_t p3);

/* Return type for RealCentroids */
struct RealCentroids_return {
	double* r0;
	size_t r1;
	size_t r2;
};

// RealCentroids returns specific on centroids

extern struct RealCentroids_return RealCentroids(int p0);

// Close terminates an oc execution

extern void Close(int p0);

// Free terminates an oc execution and unregister it from global registry

extern void Free(int p0);

#ifdef __cplusplus
}
#endif
