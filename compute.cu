#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void compute_Pairwise_Accelerations(vector3 *hPos, double *mass, vector3 *accels, int numEntities) 
{
    // Shared memory for the portion of the accels array in each threadblock
    __shared__ double s_mass[16];
    __shared__ vector3 s_hPos[16];


    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEntities && threadIdx.x == 0) {
        s_mass[threadIdx.y] = mass[i];
        s_hPos[threadIdx.y][0] = hPos[i][0];
        s_hPos[threadIdx.y][1] = hPos[i][1];
        s_hPos[threadIdx.y][2] = hPos[i][2];
    }

    __syncthreads();

    if (i < numEntities && j < numEntities) {
        if (i == j) {
            FILL_VECTOR(accels[i * numEntities + j], 0, 0, 0);
        } else {
            vector3 distance;
            for (int k = 0; k < 3; k++) {
                distance[k] = s_hPos[threadIdx.y][k] - hPos[j][k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            FILL_VECTOR(accels[i * numEntities + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
        }
    }
}

__global__ void sum_accelerations(vector3 *accels, vector3 *accel_sum, int numEntities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEntities) {
        vector3 sum = {0, 0, 0};
        for (int j = 0; j < numEntities; j++) {
            for (int k = 0; k < 3; k++) {
                sum[k] += accels[i * numEntities + j][k];
            }
        }
        accel_sum[i][0] = sum[0];
        accel_sum[i][1] = sum[1];
        accel_sum[i][2] = sum[2];
    }
}

__global__ void update_Velocity_and_Position(vector3 *hPos, vector3 *hVel, vector3 *accel_sum, int numEntities, double interval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEntities) {
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[i][k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }
}

void compute() {
    int numEntities = NUMENTITIES;
    size_t sizePosVel = sizeof(vector3) * numEntities;
    size_t sizeMass = sizeof(double) * numEntities;
    size_t sizeAccels = sizeof(vector3) * numEntities * numEntities;

    // Allocate device memory
    double *d_mass;
    vector3 *d_hPos, *d_hVel, *d_accels, *d_accel_sum;
    cudaMalloc((void **)&d_mass, sizeMass);
    cudaMalloc((void **)&d_hPos, sizePosVel);
    cudaMalloc((void **)&d_hVel, sizePosVel);
    cudaMalloc((void **)&d_accels, sizeAccels);
    cudaMalloc((void **)&d_accel_sum, sizePosVel);

    // Copy data from host to device
    cudaMemcpy(d_mass, mass, sizeMass, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hPos, hPos, sizePosVel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizePosVel, cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 blockDim(16, 16);
    dim3 gridDim((numEntities + 15) / 16, (numEntities + 15) / 16);
    compute_Pairwise_Accelerations<<<gridDim, blockDim>>>(d_hPos, d_mass, d_accels, numEntities);
    cudaDeviceSynchronize();

    dim3 blockDim2(256);
    dim3 gridDim2((numEntities + 255) / 256);
    sum_accelerations<<<gridDim2, blockDim2>>>(d_accels, d_accel_sum, numEntities);
    cudaDeviceSynchronize();

    update_Velocity_and_Position<<<gridDim2, blockDim2>>>(d_hPos, d_hVel, d_accel_sum, numEntities, INTERVAL);
    cudaDeviceSynchronize();

    // Copy results back to the host
    cudaMemcpy(hPos, d_hPos, sizePosVel, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizePosVel, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mass);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_accels);
    cudaFree(d_accel_sum);
}
