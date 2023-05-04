#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
	
//first compute the pairwise accelerations.  Effect is on the first argument.
__global__ void compute_Pairwise_Accelerations(vector3 *hPos, double *mass, vector3 *accels, int numEntities) {

	__shared__ double share_mass[16];
	__shared__ vector3 share_hPos[16];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities && threadIdx.x == 0) {
		share_mass[threadIdx.y] = mass[i];
		share_hPos[threadIdx.y][0] = hPos[i][0];
		share_hPos[threadIdx.y][1] = hPos[i][1];
		share_hPos[threadIdx.y][2] = hPos[i][2];
	}

	__syncthreads();

	if (i < numEntities && j < numEntities) {
		if (i == j) {
			FILL_VECTOR(accels[i * numEntities + j], 0, 0, 0);
		} else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[i][k] - hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * numEntities + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	} 
}

//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
__global__ void update_velocity_and_position(vector3* hPos, vector3* hVel, vector3* accels, int numEntities, double interval) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities) {
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[k] * interval;
			hPos[i][k] = hVel[i][k] * interval;
		}
	}
}

__global__ void sum(vector3* accels, vector3* sum_accels, int numEntities) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities) {
		vector3 sum={0, 0, 0};
		for (int j = 0; j < numEntities; j++){
			for (int k = 0;k < 3; k++) {
				accel_sum[k] += accels[i * numEntities + j][k];
			}
		}
		sun_accels[i][0] = sum[0];
		sun_accels[i][1] = sum[1];
		sun_accels[i][2] = sum[2];
	}
}


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	vector3 *device_hPos, *device_hVel, *device_accels;
	double *device_mass;

	cudaMalloc((void**)&device_hPos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_hVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_mass, sizeof(double)*NUMENTITIES);
	cudaMalloc((void**)&device_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);

	cudaMemcpy(device_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

	compute_Pairwise_Accelerations<<<gridDim, blockDim>>>(device_hPos, device_mass, device_accels, NUMENTITIES);

	cudaDeviceSynchronize();

	sum_and_update_velocity_and_position<<<gridDim.x, blockDim.x>>>(device_hPos, device_hVel, device_accels, NUMENTITIES, INTERVAL);

	cudaMemcpy(hPos, device_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, device_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(device_hPos);
	cudaFree(device_hVel);
	cudaFree(device_mass);
	cudaFree(device_accels);
}
