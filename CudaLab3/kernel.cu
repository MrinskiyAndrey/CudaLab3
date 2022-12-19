
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "wb.h"


int main(int argc, char* argv[]) {
    wbArg_t args;
    float* hostInput1 = nullptr;
    float* hostInput2 = nullptr;
    float* hostOutput = nullptr;
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;


    int inputLength;

    args = wbArg_read(argc, argv); /* чтение входных аргументов */

    // Импорт входных данных на хост
    wbTime_start(Generic, "Importing data to host");
    hostInput1 =(float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =(float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    wbTime_stop(Generic, "Importing data to host");

    // Объявление и выделение памяти под выходные данные
    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    

    // Объявление и выделение памяти под входные и выходные данные  на устройства через thrust
    wbTime_start(GPU, "Doing GPU memory allocation");
   
    cudaMalloc(&deviceInput1, inputLength * sizeof(float));
    cudaMalloc(&deviceInput2, inputLength * sizeof(float));
    cudaMalloc(&deviceOutput, inputLength * sizeof(float));

    thrust::device_ptr<float> d_in1(deviceInput1);
    thrust::device_ptr<float> d_in2(deviceInput2);
    thrust::device_ptr<float> d_out(deviceOutput);

    wbTime_stop(GPU, "Doing GPU memory allocation");

    // Копирование на устройство
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    // Выполнение операции сложения векторов
    wbTime_start(Compute, "Doing the computation on the GPU");
    thrust::transform(d_in1, d_in1 + inputLength * sizeof(float), d_in2, d_out, thrust::plus<float>());
    /////////////////////////////////////////////////////////

    // Копирование данных обратно на хост
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    return 0;
}

