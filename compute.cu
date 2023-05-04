#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
	
//first compute the pairwise accelerations.  Effect is on the first argument.
__global__ void compute_Pairwise_Accelerations(vector3* hPos, double* mass, vector3* accels, int numEntities) {

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k;

	if (i < numEntities && j < numEntities) {
		if (i == j) {
			FILL_VECTOR(accels[i * numEntities + j][j], 0, 0, 0);
		} else {
			vector3 distance;
			for (k=0; k < 3; k++) {
				distance[k] = hPos[i][k]  -hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * numEntities + j][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	} 
}

//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
__global__ void sum_and_update_velocity_and_position(vector3* hPos, vector3* hVel, vector3* accels, int numEntities, double interval) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;

	if (i < numEntities) {
		vector3 accel_sum={0, 0, 0};
		for (j = 0; j < numEntities; j++){
			for (k = 0;k < 3; k++) {
				accel_sum[k] += accels[i * numEntities + j][k];
			}
		}
	}
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (k = 0; k < 3; k++){
		hVel[i][k] += accel_sum[k] * interval;
		hPos[i][k] = hVel[i][k] * interval;
	}
}


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	vector3* device_hPos, device_hVel, device_accels;
	double* device_mass;

	cudaMalloc((void**)&device_hPos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_hVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_mass, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);

	cudaMemcpy(device_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mass, mass, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim(NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y;

	compute_Pairwise_Accelerations <<< gridDim, vlockDim >> (device_hPos, device_mass, device_accels, NUMENTITIES);

	cudaDeviceSynchronize();

	sum_and_update_velocity_and_position <<< gridDim.x, vlockDim.x >> (device_hPos, device_hVel, device_accels, NUMENTITIES, INTERVAL);

	cudaMemcpy(hPos, device_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, device_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	free(device_hPos);
	free(device_hVel);
	free(device_accels);
	free(device_mass);
}
