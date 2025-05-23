#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.cuh"
// 算法参数
constexpr uint32_t N = 1000000;   // 数据总量
constexpr uint32_t BITS_PER_PASS = 8;   // 每次处理8位
constexpr uint32_t RADIX  =1 << BITS_PER_PASS; // 基数大小（256）
constexpr uint32_t RADIX_MASK       = RADIX-1;     //Mask of digit bins, to extract digits
constexpr uint32_t globalHistThreads   =128;    // 线程块里的线程个数
constexpr uint32_t PASS =  32 / BITS_PER_PASS;  // 基排序
constexpr uint32_t globalHistItemsPerThread =  16;    //每个线程处理的uint32_t 个数
constexpr uint32_t globalHistPartitionSize = globalHistThreads * globalHistItemsPerThread;
constexpr uint32_t globalHistThreadblocks = (N + globalHistPartitionSize - 1) / globalHistPartitionSize;
constexpr uint32_t globalHistOffset = (globalHistThreadblocks+1) * RADIX;
//constexpr uint32_t BINS_PER_THREAD = (RADIX + globalHistThreads - 1) / globalHistThreads; //每个线程

constexpr uint32_t BIN_KEYS_PER_THREAD  = 21;                                      //Keys per thread in k_DigitBinning
constexpr uint32_t binningThreads = 384;			//2080 super seems to really like 512 
constexpr uint32_t BIN_WARPS = binningThreads>>5;                                      //Warps per threadblock in k_DigitBinning

constexpr uint32_t BIN_PART_SIZE = binningThreads * BIN_KEYS_PER_THREAD;
constexpr uint32_t BIN_HISTS_SIZE = BIN_WARPS * RADIX;                         //Total size of warp histograms in shared memory in k_DigitBinning
constexpr uint32_t BIN_SUB_PART_SIZE = BIN_KEYS_PER_THREAD * LANE_COUNT;     //Subpartition tile size of a single warp in k_DigitBinning
constexpr uint32_t partitionSize = binningThreads*BIN_KEYS_PER_THREAD;
constexpr uint32_t binningThreadblocks = (N + partitionSize - 1) / partitionSize;
const uint32_t BIN_WHSize = (BIN_PART_SIZE > BIN_HISTS_SIZE) ? BIN_PART_SIZE : (BIN_HISTS_SIZE + 1);

#define Host_SUB_PART_START (WARP_INDEX * LANE_COUNT * globalHistItemsPerThread)
#define Host_PART_START (blockIdx.x*globalHistThreads* globalHistItemsPerThread)

#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile


//for the chained scan with decoupled lookback
constexpr uint32_t FLAG_NOT_READY = 0;                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
constexpr uint32_t FLAG_REDUCTION = 1;                                       //Flag value indicating reduction of a partition tile is ready
constexpr uint32_t FLAG_INCLUSIVE = 2;                                       //Flag value indicating inclusive sum of a partition tile is ready
constexpr uint32_t FLAG_MASK = 3;                                       //Mask used to retrieve flag values

__global__ void mem_init(uint32_t* values, 
    uint32_t* global_histogram, 
    uint32_t* index, 
    uint32_t values_size, 
    uint32_t passes, 
    uint32_t global_size);

__global__ void global_histogram_kernel_atomic(
    const uint32_t* keys,
    uint32_t* global_histogram,
    uint32_t num_elements,
    uint32_t begin_bit,
    uint32_t end_bit);

__global__ void global_histogram_kernel(
    const uint32_t* d_keys,
    uint32_t* global_histogram,
    uint32_t num_elements,
    uint32_t begin_bit,
    uint32_t end_bit);


__global__ void exclusive_sum_kernel(uint32_t* global_histogram);

__global__ void digit_binning_kernel(const uint32_t* keys_in,
    uint32_t* keys_out,
    const uint32_t* values_in,
    uint32_t* values_out,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift,
    uint32_t pass);

    


