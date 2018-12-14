#pragma once

#define HIGH_RESOURCE true
#define NVML false
#define CPU_SHARES false
#define BENCHMARK false

#if HIGH_RESOURCE
#define DEFAULT_BLOCK_SIZE 512
#define DEFAULT_THREADS_PER_BLOCK 1024
#else
#define DEFAULT_BLOCK_SIZE 512
#define DEFAULT_THREADS_PER_BLOCK 512
#endif

#if CPU_SHARES
#define WORK_PER_THREAD 256
#else
#define WORK_PER_THREAD 256
#endif