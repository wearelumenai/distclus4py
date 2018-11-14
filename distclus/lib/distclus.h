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

#line 3 "mcmc.go"
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


extern void FreeRealArray(double* p0);

extern void FreeIntArray(long int* p0);

extern int MCMC(size_t p0, int p1, int p2, int p3, double p4, double p5, double p6, double p7, int p8, initializer p9, long int p10);

extern void Push(int p0, double* p1, size_t p2, size_t p3);

extern void Run(int p0, int p1);

/* Return type for Predict */
struct Predict_return {
	long int* r0;
	size_t r1;
};

extern struct Predict_return Predict(int p0, double* p1, size_t p2, size_t p3, int p4);

/* Return type for RealCentroids */
struct RealCentroids_return {
	double* r0;
	size_t r1;
	size_t r2;
};

extern struct RealCentroids_return RealCentroids(int p0);

extern void Close(int p0);

extern void Free(int p0);

#ifdef __cplusplus
}
#endif
