#include "oneSweep.cuh"

__global__ void mem_init(uint32_t* values, 
    uint32_t* global_histogram, 
    uint32_t* index,
    uint32_t values_size,
    uint32_t passes,
    uint32_t global_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < values_size)
        values[idx] = idx;
    if (idx < passes)
        index[idx] = 0;
    if (idx < global_size)
        global_histogram[idx] = 0;
}

__global__ void global_histogram_kernel_atomic(
    const uint32_t* keys,
    uint32_t* global_histogram,
    uint32_t num_elements,
    uint32_t begin_bit,
    uint32_t end_bit) {
    
    int idx = threadIdx.x + blockIdx.x* blockDim.x;
    if (idx < num_elements) {
        uint32_t key = keys[idx];
        uint32_t digit = (key >> begin_bit) & (RADIX - 1);
        atomicAdd(&global_histogram[digit], 1);
    }
}

// 核函数：计算直方图
__global__ void global_histogram_kernel(
    const uint32_t* d_keys,
    uint32_t* global_histogram,
    uint32_t num_elements,
    uint32_t begin_bit,
    uint32_t end_bit)
{
    __shared__ uint32_t shared_hist[PASS][RADIX];

    // 初始化共享内存
#pragma unroll
    for (uint32_t bin = threadIdx.x; bin < RADIX; bin += globalHistThreads)
    {
#pragma unroll
        for (int pass = 0; pass < PASS; ++pass)
        {
            shared_hist[pass][bin] = 0;
        }
    }


    __syncthreads();

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t keys[globalHistItemsPerThread];

    for (uint32_t i = 0, t = getLaneId() + Host_SUB_PART_START + Host_PART_START; i < globalHistItemsPerThread && t< num_elements; ++i, t += LANE_COUNT)
    {
        keys[i] = d_keys[t];

    }

    // 统计直方图
#pragma unroll
    for (uint32_t current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += BITS_PER_PASS, ++pass)
    {
#pragma unroll
        for (uint32_t u = 0; u < globalHistItemsPerThread; u += 1)
        {
            uint32_t idx = getLaneId() + Host_SUB_PART_START + Host_PART_START + u*LANE_COUNT;
            if (idx < num_elements) {
                uint32_t digit = (keys[u] >> current_bit) & RADIX_MASK;
                atomicAdd(&shared_hist[pass][digit], 1);
            }
        }
    }

    __syncthreads();

    // 合并到全局直方图
#pragma unroll
    for (uint32_t bin = threadIdx.x; bin < RADIX; bin += globalHistThreads)
    {
#pragma unroll
        for (uint32_t pass = 0; pass < PASS; ++pass)
        {
            atomicAdd(&global_histogram[pass * globalHistOffset + bin], shared_hist[pass][bin]);
        }
    }
}

// 核函数：计算前缀和  RADIX = blockDim.x
__global__ void exclusive_sum_kernel(uint32_t* global_histogram) {
    __shared__ uint32_t s_scan[RADIX];

    s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(global_histogram[threadIdx.x + blockIdx.x * globalHistOffset]);
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_scan[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_scan[threadIdx.x << LANE_LOG]);
    __syncthreads();

    global_histogram[threadIdx.x + blockIdx.x * globalHistOffset] = 
        (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
}

__global__ void digit_binning_kernel(const uint32_t* keys_in,
    uint32_t* keys_out,
    const uint32_t* values_in,
    uint32_t* values_out,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift,
    uint32_t pass) {
    __shared__ uint32_t s_warpHistograms[BIN_WHSize];
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << BITS_PER_PASS];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)  //unnecessary work for last partion but still a win to avoid another barrier
        s_warpHistograms[i] = 0;

    //atomically assign partition tiles
    if (threadIdx.x == 0)
        s_warpHistograms[BIN_WHSize - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
    __syncthreads();
    const uint32_t partitionIndex = s_warpHistograms[BIN_WHSize - 1];

    //To handle input sizes not perfect multiples of the partition tile size
    //load keys
    uint32_t keys[BIN_KEYS_PER_THREAD];
    //uint32_t values[BIN_KEYS_PER_THREAD];

#pragma unroll
    for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT)
    {
        keys[i] = keys_in[t];
    }

    uint32_t offsets[BIN_KEYS_PER_THREAD];

    //WLMS
#pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    {
        if (getLaneId() + BIN_SUB_PART_START + BIN_PART_START + i * LANE_COUNT < size) {
            /*
            //CUB version "match any"
            unsigned warpFlags;
            #pragma unroll
            for (int k = 0; k < BITS_PER_PASS; ++k)
            {
                uint32_t mask;
                uint32_t current_bit = 1 << k + radixShift;
                asm("{\n"
                    "    .reg .pred p;\n"
                    "    and.b32 %0, %1, %2;"
                    "    setp.ne.u32 p, %0, 0;\n"
                    "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
                    "    @!p not.b32 %0, %0;\n"
                    "}\n" : "=r"(mask) : "r"(keys[i]), "r"(current_bit));
                warpFlags = (k == 0) ? mask : warpFlags & mask;
            }
            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
            */
            const uint32_t mask = __activemask();
            unsigned warpFlags = mask;
            
#pragma unroll
            for (int k = 0; k < BITS_PER_PASS; ++k)
            {
                const bool t2 = keys[i] >> k + radixShift & 1;
                warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(mask, t2);
            }

            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());


            //An alternative, but slightly slower version.

            //offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
            //__syncwarp(mask);
            //if (bits == 0)
            //    s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
            //__syncwarp(mask);
            
            
            uint32_t preIncrementVal;
            if (bits == 0)
                preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));
            __syncwarp(mask);
            offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
        }

    }
    //printf("%u\n", threadIdx.x);
    __syncthreads();

    //exclusive prefix sum up the warp histograms
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = s_warpHistograms[threadIdx.x];
        for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
        {
            reduction += s_warpHistograms[i];
            s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }


        atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX + pass * globalHistOffset],
            FLAG_REDUCTION | reduction << 2);

        //begin the exclusive prefix sum across the reductions
        s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    }
    __syncthreads();



    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_localHistogram[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
    __syncthreads();

    if (threadIdx.x < RADIX && getLaneId())
        s_localHistogram[threadIdx.x] += __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
    __syncthreads();

    //update offsets
    if (WARP_INDEX)
    {
#pragma unroll 
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
        }
    }
    else
    {
#pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
    }
    __syncthreads();

    //scatter keys into shared memory
#pragma unroll
    for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT)
        s_warpHistograms[offsets[i]] = keys[i];

    //split the warps into single thread cooperative groups and lookback
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = 0;
        for (uint32_t k = partitionIndex; k >= 0; )
        {
            const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX + pass * globalHistOffset];

            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
            {
                reduction += flagPayload >> 2;
                atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX + pass * globalHistOffset], 1 | (reduction << 2));
                s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
                break;
            }

            if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
            {
                reduction += flagPayload >> 2;
                k--;
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = threadIdx.x, u = 0; i < BIN_PART_SIZE && (i + BIN_PART_START < size); i += blockDim.x,u++) 
        keys_out[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    
    uint32_t digits[BIN_KEYS_PER_THREAD];
#pragma unroll
    for (int u = 0; u < BIN_KEYS_PER_THREAD; ++u)
    {
        int idx = threadIdx.x + u * binningThreads;
        digits[u] = s_warpHistograms[idx] >> radixShift & RADIX_MASK;
    }
  
    uint32_t values[BIN_KEYS_PER_THREAD];
#pragma unroll
    #pragma unroll   
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT)
        {
            values[i] = values_in[t];
        }
    __syncthreads();

    //scatter values into shared memory
#pragma unroll
    for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT) {
        s_warpHistograms[offsets[i]] = values[i];
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < BIN_KEYS_PER_THREAD; ++u) {
        int idx = threadIdx.x + u * binningThreads;
        uint32_t value = s_warpHistograms[idx];
        uint32_t global_idx = idx + s_localHistogram[digits[u]];
        if (idx + BIN_PART_START < size)
            values_out[global_idx] = value;
    }

    //slower version   ,caused by registers and local memory
    //scatter keys into global memory
//#pragma unroll
//    for (uint32_t i = threadIdx.x, u = 0; i < BIN_PART_SIZE && (i + BIN_PART_START < size); i += blockDim.x,u++) {
//        uint32_t idx = s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i;
//        keys_out[idx] = s_warpHistograms[i];
//        keys[u] = idx;
//    }
//    __syncthreads();
//    uint32_t values[BIN_KEYS_PER_THREAD];
//#pragma unroll   
//    for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT)
//    {
//        values[i] = values_in[t];
//    }
//
//    //scatter values into shared memory
//#pragma unroll
//    for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD && t < size; ++i, t += LANE_COUNT) {
//        s_warpHistograms[offsets[i]] = values[i];
//    }
//    __syncthreads();
//
//    //scatter values into global memory
//
//    for (uint32_t i = threadIdx.x, u = 0; i < BIN_PART_SIZE && (i + BIN_PART_START < size); i += blockDim.x, u++) {
//        uint32_t idx = keys[u];
//        values_out[idx] = s_warpHistograms[i];
//    }
    //__syncthreads();
    //if (threadIdx.x == 0) {
    //    for (int i = 0;i < N;i++) {
    //        printf("%u\n", values_out[i]);
    //    }
    //}
    
}