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

#line 7 "oc.go"
#include "bind.h"

#line 1 "cgo-generated-wrapper"

#line 3 "streaming.go"
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


// FreeRealArray is a convenient function to free a C double array allocated by the facade

extern void FreeRealArray(double* p0);

// FreeIntArray is a convenient function to free a C long array allocated by the facade

extern void FreeIntArray(long int* p0);

/* Return type for KMeans */
struct KMeans_return {
	int r0; /* descr */
	char* r1; /* errMsg */
};

// KMeans builds and registers a kmeans algorithm

extern struct KMeans_return KMeans(space p0, double* p1, size_t p2, size_t p3, size_t p4, int p5, initializer p6, long int p7, int p8, int p9, int p10, space p11, int p12);

/* Return type for MCMC */
struct MCMC_return {
	int r0; /* descr */
	char* r1; /* errMsg */
};

// MCMC builds and registers a mcmc algorithm

extern struct MCMC_return MCMC(space p0, double* p1, size_t p2, size_t p3, size_t p4, int p5, initializer p6, long int p7, size_t p8, int p9, int p10, int p11, int p12, double p13, double p14, double p15, double p16, int p17, space p18, int p19);

// Push pushes an array of element to the algorithm corresponding to the given descriptor

extern char* Push(int p0, double* p1, size_t p2, size_t p3, size_t p4);

// Run runs the algorithm corresponding to the given descriptor

extern char* Run(int p0, int p1);

/* Return type for Predict */
struct Predict_return {
	long int* r0; /* labels */
	size_t r1; /* n1 */
	double* r2; /* centers */
	size_t r3; /* c1 */
	size_t r4; /* c2 */
	size_t r5; /* c3 */
	char* r6; /* errMsg */
};

// Predict returns the centroids and labels for the input data
// from the algorithm corresponding to the given descriptor

extern struct Predict_return Predict(int p0, double* p1, size_t p2, size_t p3, size_t p4);

/* Return type for Centroids */
struct Centroids_return {
	double* r0; /* data */
	size_t r1; /* l1 */
	size_t r2; /* l2 */
	size_t r3; /* l3 */
	char* r4; /* errMsg */
};

// Centroids returns the centroids
// from the algorithm corresponding to the given descriptor

extern struct Centroids_return Centroids(int p0);

/* Return type for RuntimeFigure */
struct RuntimeFigure_return {
	double r0; /* value */
	char* r1; /* errMsg */
};

// RuntimeFigure returns runtime figures
// from the algorithm corresponding to the given descriptor

extern struct RuntimeFigure_return RuntimeFigure(int p0, figure p1);

// Close terminates the algorithm corresponding to the given descriptor

extern void Close(int p0);

// Free terminates the algorithm corresponding to the given descriptor
// and free allocated resources

extern void Free(int p0);

/* Return type for Streaming */
struct Streaming_return {
	int r0; /* descr */
	char* r1; /* errMsg */
};

// Streaming builds and registers a streaming algorithm

extern struct Streaming_return Streaming(space p0, double* p1, size_t p2, size_t p3, size_t p4, long int p5, int p6, double p7, double p8, space p9, int p10);

#ifdef __cplusplus
}
#endif
