#include "dataInit.cuh"
#include "oneSweep.cuh"
#include <iostream>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <numeric>

#define TIMES 20
#define sort_type 0
// 验证函数（主机端）
bool verify_result(const uint32_t* sorted_keys,
    const uint32_t* sorted_values,
    int n)
{
    for (int i = 1; i < n; ++i) {
        if (sorted_keys[i - 1] > sorted_keys[i]) return false;
    }
    return true;
}


int main() {

    int device_id = 0;  // 默认设备 ID
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("Device Name: %s\n", prop.name);
    printf("Number of SMs (Multi-Processors): %d\n", prop.multiProcessorCount);

    // 主机内存分配
    uint32_t* h_keys = new uint32_t[N];
    uint32_t* h_values = new uint32_t[N];
    uint32_t* h_sorted_keys = new uint32_t[N];
    uint32_t* h_sorted_values = new uint32_t[N];
    // 生成随机数据
    std::iota(h_values, h_values + N, 0);

    // 设备内存分配
    uint32_t* d_keys[2], * d_values[2];

    cudaMalloc(&d_keys[0], N * sizeof(uint32_t));
    cudaMalloc(&d_keys[1], N * sizeof(uint32_t));
    cudaMalloc(&d_values[0], N * sizeof(uint32_t));
    cudaMalloc(&d_values[1], N * sizeof(uint32_t));

    // 拷贝初始数据到设备
    //cudaMemcpy(d_keys[0], h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values[0], h_values, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t* index;
    uint32_t* d_histogram;

    //for cub radix sort and cub reduce
    size_t temp_storage_bytes{ 0 };
    std::shared_ptr<thrust::device_vector<float>> cub_temp_storge{ nullptr }; //temp_storge_size
    
    switch (sort_type) {
    case 0:
        cudaMalloc(&index, PASS * sizeof(uint32_t));
        cudaMalloc(&d_histogram, PASS * globalHistOffset * sizeof(uint32_t));
        break;
    case 1:
        cub_temp_storge = std::make_shared<thrust::device_vector<float>>(N * 4);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys[0], d_keys[1], d_values[0], d_values[1], N);
        if (temp_storage_bytes > 0 && cub_temp_storge->size() < temp_storage_bytes) 
            cub_temp_storge->resize(temp_storage_bytes);        
        break;
    }
    

    // 拷贝初始数据到设备
    //cudaMemcpy(d_keys[0], h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values[0], h_values, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (int time = 1;time < TIMES + 1;time++) {

        InitRandom << <256, 1024 >> > (d_keys[0], N, time);
        //InitDescending << <256, 1024 >> > (d_keys[0], N);

        switch (sort_type)
        {
        case 0:
            // 重置直方图
            //cudaMemset(d_histogram, 0, PASS * globalHistOffset * sizeof(uint32_t));
            //cudaMemset(index, 0, PASS * sizeof(uint32_t));
            //cudaMemcpy(d_values[0], h_values, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
            mem_init << <(N + 255) / 256, 256 >> > (d_values[0], d_histogram, index,
                N, PASS, PASS * globalHistOffset);

            // 步骤1：计算直方图
            global_histogram_kernel << <globalHistThreadblocks, globalHistThreads >> > (
                d_keys[0], d_histogram, N, 0u, 32u);

            exclusive_sum_kernel << <PASS, RADIX >> > (d_histogram);

            // 基数排序主循环
            for (int bit = 0, pass = 0; bit < 32; bit += BITS_PER_PASS, pass++) {
                int input_idx = (bit / BITS_PER_PASS) % 2;
                int output_idx = 1 - input_idx;

                digit_binning_kernel << <binningThreadblocks, binningThreads >> > (
                    d_keys[input_idx],
                    d_keys[output_idx],
                    d_values[input_idx],
                    d_values[output_idx],
                    d_histogram,
                    index,
                    N,
                    bit,
                    pass);
            }

            break;
        case 1:
            cub::DeviceRadixSort::SortPairs(cub_temp_storge->data().get(), temp_storage_bytes, d_keys[0], d_keys[1], d_values[0], d_values[1], N);
            break;
        }
        

    }


    // 拷贝结果回主机
    int final_idx;
    switch (sort_type){
        case 0: 
            final_idx = (32 / BITS_PER_PASS) % 2;
            break;
        case 1: final_idx = 1;
            break;
    }
    cudaMemcpy(h_sorted_keys, d_keys[final_idx], N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sorted_values, d_values[final_idx], N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0;i < 10 && i < N;i++) {
        printf("%u %u\n", h_sorted_keys[i], h_sorted_values[i]);
    }

    // 验证结果
    bool success = verify_result(h_sorted_keys, h_sorted_values, N);

    // 输出结果
    std::cout << "===== 手动实现性能 =====" << std::endl;
    std::cout << "验证结果: " << (success ? "成功" : "失败") << std::endl;

    return 0;
}