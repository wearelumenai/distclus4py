#ifndef __BIND_H_
#define __BIND_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef enum {I_RANDOM, I_GIVEN, I_KMEANS_PP, I_OC} initializer;
typedef enum {S_EUCLID, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {
  F_ITERATIONS, F_ACCEPTATIONS, F_MAX_DISTANCE, F_PUSHED_DATA,
  F_LAST_ITERATIONS, F_DURATION, F_LAST_DURATION, F_LAST_DATA_TIME,
  F_LAMBDA, F_RHO, F_RGIBBS, F_TIME
} figure;

#endif
