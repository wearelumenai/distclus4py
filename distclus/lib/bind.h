#ifndef __BIND_H_
#define __BIND_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef enum {I_RANDOM, I_GIVEN, I_KMEANS_PP, I_OC} initializer;
typedef enum {S_EUCLID, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {
  F_ITERATIONS, F_PUSHED_DATA, F_DURATION, F_LAST_DATA_TIME,
  F_ACCEPTATIONS, F_LAMBDA, F_RHO, F_RGIBBS, F_TIME,
  F_MAX_DISTANCE
} figure;

#endif
