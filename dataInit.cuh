#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//Hybrid LCG-Tausworthe PRNG
//From GPU GEMS 3, Chapter 37
//Authors: Lee Howes and David Thomas 
#define TAUS_STEP_1         ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2         ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3         ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP            (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS         (z1 ^ z2 ^ z3 ^ z4)

//Initialize the input to a sequence of descending integers.
__global__ void InitDescending(uint32_t* sort, uint32_t size)
{
	for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x)
		sort[i] = size - i;
}

//Initialize the input to random integers. Because this is higher entropy than the descending sequence, and
//becuase we do not implement short circuit evaluation, this tends to be significantly faster
__global__ void InitRandom(uint32_t* sort, uint32_t size, uint32_t seed)
{
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	uint32_t z1 = (idx << 2) * seed;
	uint32_t z2 = ((idx << 2) + 1) * seed;
	uint32_t z3 = ((idx << 2) + 2) * seed;
	uint32_t z4 = ((idx << 2) + 3) * seed;

	for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x)
	{
		z1 = TAUS_STEP_1;
		z2 = TAUS_STEP_2;
		z3 = TAUS_STEP_3;
		z4 = LCG_STEP;
        
		sort[i] = HYBRID_TAUS;
	}
}
