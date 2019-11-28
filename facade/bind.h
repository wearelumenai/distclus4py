#ifndef __BIND_H_
#define __BIND_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP, I_OC} initializer;
typedef enum {S_VECTORS, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {F_ITERATIONS, F_ACCEPTATIONS, F_MAX_DISTANCE, F_PUSHED_DATA} figure;

#endif
