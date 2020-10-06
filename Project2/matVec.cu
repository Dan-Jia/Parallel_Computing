#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DTYPE float

__global__ void kernelAx(DTYPE *a, DTYPE *x, DTYPE *y, int size) {
  int rowId = threadIdx.x + blockDim.x * blockIdx.x;

  for (int j = 0; j < size; ++j) {
    y[rowId] += a[rowId * size + j] * x[j];
  }
}

__global__ void kernelATx(DTYPE *a, DTYPE *x, DTYPE *y, int size) {
  int rowId = threadIdx.x + blockDim.x * blockIdx.x;

  for (int j = 0; j < size; ++j) {
    y[rowId] += a[rowId + size * j] * x[j];
  }
}

__global__ void kernelAx_SM(DTYPE *a, DTYPE *x, DTYPE *temp, int size,
                            int row) {
  __shared__ DTYPE sm[512];
  int colId =
      threadIdx.x + blockDim.x * blockIdx.x;  // Globale Adresse des threads
  int smId = threadIdx.x;  // lokale Adresse des threads(innerhalb eines Blocks)
  sm[smId] = a[row * size + colId] * x[colId];
  __syncthreads();

  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (smId < k) {
      sm[smId] += sm[smId + k];
    }
    __syncthreads();
  }

  if (smId == 0) {
    temp[blockIdx.x] = sm[0];
  }
}

__global__ void kernelAx_SM_Reduction(DTYPE *y, DTYPE *temp, int row) {
  // extern __shared__ DTYPE sm[];
  __shared__ DTYPE sm[32];
  int colId =
      threadIdx.x + blockDim.x * blockIdx.x;  // Globale Adresse des threads
  int smId = threadIdx.x;  // lokale Adresse des threads(innerhalb eines Blocks)
  sm[smId] = temp[colId];

  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (smId < k) {
      sm[smId] += sm[smId + k];
    }
    __syncthreads();
  }

  if (smId == 0) {
    y[row] = sm[0];
  }
}

__global__ void kernelAx_SM_atomic(DTYPE *y, DTYPE *temp, int row) {
  int colId =
      threadIdx.x + blockDim.x * blockIdx.x;  // Globale Adresse des threads

  atomicAdd(&y[row], temp[colId]);
}

__global__ void kernelAx_SM_DP(DTYPE *a, DTYPE *x, DTYPE *y, DTYPE *temp,
                               int size, int blocksPerGrid, dim3 grid,
                               dim3 threads) {
  int tId = threadIdx.x + blockDim.x * blockIdx.x;

  kernelAx_SM<<<grid, threads>>>(a, x, temp, size, tId);
  kernelAx_SM_atomic<<<1, blocksPerGrid>>>(y, temp, tId);
}

// A mit Werten füllen (hier einfach 1en)
void fillA(DTYPE *a, int size) {
  for (int i = 0; i < size * size; i++) a[i] = 1.0;
}

// X mit Werten füllen
void fillX(DTYPE *x, int size) {
  for (int i = 0; i < size; i++) x[i] = (DTYPE)(i + 1);
}

void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size) {
  for (int i = 0; i < size; ++i) {
    y[i] = 0.0f;
    for (int j = 0; j < size; ++j) {
      y[i] += a[i * size + j] * x[j];
    }
  }
}

void hostATx(DTYPE *a, DTYPE *x, DTYPE *y, int size) {
  for (int i = 0; i < size; ++i) {
    y[i] = 0.0f;
    for (int j = 0; j < size; ++j) {
      y[i] += a[j * size + i] * x[j];
    }
  }
}

bool checkResult(DTYPE *yh_Ax, DTYPE *yh_Atx, DTYPE *yd_Ax, DTYPE *yd_ATx,
                 DTYPE *yd_Ax_SM_Reduction, DTYPE *yd_Ax_SM_atomic,
                 DTYPE *yd_Ax_SM_DP, int size) {
  bool res = true;
  for (int i = 0; i < size; i++) {
    res &= (yh_Ax[i] == yd_Ax[i]);
    if (i < 10)
      printf("%f %f %f %f %f %f %f\n", yh_Ax[i], yh_Atx[i], yd_Ax[i], yd_ATx[i],
             yd_Ax_SM_Reduction[i], yd_Ax_SM_atomic[i], yd_Ax_SM_DP[i]);
  }
  return res;
}

/*
   Main Routine:
   Input: i,[threads]
   Berechnet A*x=y auf der GPU wobei A eine Größe von R^{n x n} hat, mit
   n=1024*i
*/
int main(int argc, char **argv) {
  int i = 1;
  int t = 512;
  if (argc > 1) {
    i = atoi(argv[1]);
    if (argc > 2) t = atoi(argv[2]);
  } else {
    printf("Usage: %s i [threads] \n", argv[0]);
    return -1;
  }
  int size = 1024 * i;
  int blocksPerGrid = size / t;
  // Datenfelder anlegen für Host
  DTYPE *a_host, *x_host, *yh_Ax_host, *yh_ATx_host, *yd_Ax_host, *yd_ATx_host,
      *temp_host, *yd_Ax_SM_Reduction_host, *yd_Ax_SM_atomic_host,
      *yd_Ax_SM_DP_host;
  // und Device
  DTYPE *a_dev, *y_dev, *x_dev, *temp_dev;
  // Events für die Zeitmessung
  cudaEvent_t start, stop;
  // Zeiten:
  // htd: Host->Device Memcpy von A und x
  float htd_time = 0.0;
  // dth: Device->Host Memcpy von y
  float dth_time = 0.0;
  // kernelA, kernelAT
  float kernelA_time = 0.0;
  float kernelAT_time = 0.0;
  float kernelAx_SM_Reduction_time = 0.0;
  float kernelAx_SM_atomic_time = 0.0;
  float kernelAx_SM_DP_time = 0.0;

  // Host Speicher anlegen und A und x füllen
  a_host = (DTYPE *)malloc(size * size * sizeof(DTYPE));
  x_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yh_Ax_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yh_ATx_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yd_Ax_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yd_ATx_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  temp_host = (DTYPE *)malloc(blocksPerGrid * sizeof(DTYPE));
  yd_Ax_SM_Reduction_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yd_Ax_SM_atomic_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yd_Ax_SM_DP_host = (DTYPE *)malloc(size * sizeof(DTYPE));

  fillA(a_host, size);
  fillX(x_host, size);
  // CUDA Events erstellen
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // CUDA Speicher anlegen für alle Arrays (a_dev,x_dev,y_dev)
  cudaMalloc((void **)&a_dev, size * size * sizeof(DTYPE));
  cudaMalloc((void **)&y_dev, size * sizeof(DTYPE));
  cudaMalloc((void **)&x_dev, size * sizeof(DTYPE));
  cudaMalloc((void **)&temp_dev, blocksPerGrid * sizeof(DTYPE));

  // Host->Device Memcpy von A und x + Zeitmessung
  cudaEventRecord(start, 0);
  cudaMemcpy(a_dev, a_host, size * size * sizeof(DTYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&htd_time, start, stop);
  cudaMemset(y_dev, 0, size * sizeof(DTYPE));

  // Konfiguration der CUDA Kernels
  dim3 threads(t);
  dim3 grid(size / threads.x);

  // CacheConfig
  cudaFuncSetCacheConfig(kernelAx, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernelATx, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernelAx_SM, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernelAx_SM_Reduction, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernelAx_SM_atomic, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernelAx_SM_DP, cudaFuncCachePreferL1);

  // kernelAx ausführen und Zeit messen
  cudaEventRecord(start);
  kernelAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelA_time, start, stop);

  // //Device->Host Memcpy für y_dev -> yd_host
  cudaEventRecord(start);
  cudaMemcpy(yd_Ax_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dth_time, start, stop);

  // kernelATx ausführen und Zeit messen
  cudaMemset(y_dev, 0, size * sizeof(DTYPE));
  cudaEventRecord(start);
  kernelATx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelAT_time, start, stop);

  // //Device->Host Memcpy für y_dev -> yd_host
  cudaEventRecord(start);
  cudaMemcpy(yd_ATx_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dth_time, start, stop);

  printf("GPU timing in ms: h->d: %f kernelAx: %f kernelATx: %f d->h: %f\n",
         htd_time, kernelA_time, kernelAT_time, dth_time);
  // checkResult(yd_Ax_host, yd_ATx_host, size);

  // kernelAx_SM_Reduction ausführen und Zeit messen
  cudaMemset(y_dev, 0, size * sizeof(DTYPE));
  cudaMemset(temp_dev, 0, blocksPerGrid * sizeof(DTYPE));
  cudaEventRecord(start);
  for (int row = 0; row < size; ++row) {
    kernelAx_SM<<<grid, threads>>>(a_dev, x_dev, temp_dev, size, row);
    cudaMemcpy(temp_host, temp_dev, blocksPerGrid * sizeof(DTYPE),
               cudaMemcpyDeviceToHost);

    kernelAx_SM_Reduction<<<1, blocksPerGrid>>>(y_dev, temp_dev, row);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelAx_SM_Reduction_time, start, stop);

  // //Device->Host Memcpy für y_dev -> yd_host
  cudaEventRecord(start);
  cudaMemcpy(yd_Ax_SM_Reduction_host, y_dev, size * sizeof(DTYPE),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dth_time, start, stop);

  printf("GPU timing in ms: h->d: %f kernelAx_SM_Reduction: %f d->h: %f\n",
         htd_time, kernelAx_SM_Reduction_time, dth_time);

  // kernelAx_SM_atomic ausführen und Zeit messen
  cudaMemset(y_dev, 0, size * sizeof(DTYPE));
  cudaMemset(temp_dev, 0, blocksPerGrid * sizeof(DTYPE));
  cudaEventRecord(start);
  for (int row = 0; row < size; ++row) {
    kernelAx_SM<<<grid, threads>>>(a_dev, x_dev, temp_dev, size, row);
    cudaMemcpy(temp_host, temp_dev, blocksPerGrid * sizeof(DTYPE),
               cudaMemcpyDeviceToHost);  /////////braucht nicht zu kopieren????

    kernelAx_SM_atomic<<<1, blocksPerGrid>>>(y_dev, temp_dev, row);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelAx_SM_atomic_time, start, stop);

  // //Device->Host Memcpy für y_dev -> yd_host
  cudaEventRecord(start);
  cudaMemcpy(yd_Ax_SM_atomic_host, y_dev, size * sizeof(DTYPE),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dth_time, start, stop);
  printf("GPU timing in ms: h->d: %f kernelAx_SM_atomic: %f d->h: %f\n",
         htd_time, kernelAx_SM_atomic_time, dth_time);

  // kernelAx_SM_DP ausführen und Zeit messen
  cudaMemset(y_dev, 0, size * sizeof(DTYPE));
  cudaEventRecord(start);
  kernelAx_SM_DP<<<grid, threads>>>(a_dev, x_dev, y_dev, temp_dev, size,
                                    blocksPerGrid, grid, threads);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelAx_SM_DP_time, start, stop);

  // //Device->Host Memcpy für y_dev -> yd_host
  cudaEventRecord(start);
  cudaMemcpy(yd_Ax_SM_DP_host, y_dev, size * sizeof(DTYPE),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dth_time, start, stop);
  printf("GPU timing in ms: h->d: %f kernelAx_SM_DP: %f d->h: %f\n", htd_time,
         kernelAx_SM_DP_time, dth_time);

  // Nutzen hier timespec um CPU Zeit zu messen
  struct timespec start_h, end_h;
  double hostA_time, hostAT_time;

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_h);
  // A*x auf Host
  hostAx(a_host, x_host, yh_Ax_host, size);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_h);
  hostA_time = (double)((end_h.tv_nsec + end_h.tv_sec * 1E9) -
                        (start_h.tv_nsec + start_h.tv_sec * 1E9)) /
               1E6;

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_h);
  // A^T*x auf Host
  hostATx(a_host, x_host, yh_ATx_host, size);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_h);
  hostAT_time = (double)((end_h.tv_nsec + end_h.tv_sec * 1E9) -
                         (start_h.tv_nsec + start_h.tv_sec * 1E9)) /
                1E6;

  printf("CPU timing in ms: kernel: Ax: %f  ATx: %f\n", hostA_time,
         hostAT_time);

  // checkResult aufrufen
  printf(
      "   CPU_Ax   ;    CPU_ATx   ;    GPU_Ax   ;    GPU_ATx   ;   "
      "GPU_Ax_SM_Reduction ;  GPU_Ax_SM_atoimc ;GPU_Ax_SM_DP  ;\n");
  printf(
      "  %f      %f      %f      %f          %f                %f       %f\n",
      hostA_time, hostAT_time, kernelA_time, kernelAT_time,
      kernelAx_SM_Reduction_time, kernelAx_SM_atomic_time, kernelAx_SM_DP_time);
  checkResult(yh_Ax_host, yh_ATx_host, yd_Ax_host, yd_ATx_host,
              yd_Ax_SM_Reduction_host, yd_Ax_SM_atomic_host, yd_Ax_SM_DP_host,
              size);

  // Speicher freigeben (Host UND Device)
  free(a_host);
  free(x_host);
  free(yh_Ax_host);
  free(yh_ATx_host);
  free(yd_Ax_host);
  free(yd_ATx_host);
  free(yd_Ax_SM_Reduction_host);
  free(yd_Ax_SM_atomic_host);
  free(yd_Ax_SM_DP_host);

  cudaFree(a_dev);
  cudaFree(x_dev);
  cudaFree(y_dev);

  // CUDA Events zerstören
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
