// VeriBlock PoW GPU Miner
// Copyright 2017-2018 VeriBlock, Inc.
// All rights reserved.
// https://www.veriblock.org
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.

#include "Miner.h"

#include <thread>

#include "absl/strings/str_join.h"

#include "Log.h"

#if NVML
nvmlDevice_t device;

void readyNVML(int deviceIndex) {
  nvmlInit();
  nvmlDeviceGetHandleByIndex(deviceIndex, &device);
}

int getTemperature() {
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  return temperature;
}

int getCoreClock() {
  unsigned int clock;
  nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT,
                     &clock);
  return clock;
}

int getMemoryClock() {
  unsigned int memClock;
  nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &memClock);
  return memClock;
}
#else
void readyNVML(int deviceIndex) {}

int getTemperature() { return -1; }

int getCoreClock() { return -1; }

int getMemoryClock() { return -1; }
#endif

/**
 * Takes the provided timestamp and places it in the header
 */
void embedTimestampInHeader(uint8_t* header, uint32_t timestamp) {
  header[55] = (timestamp & 0x000000FF);
  header[54] = (timestamp & 0x0000FF00) >> 8;
  header[53] = (timestamp & 0x00FF0000) >> 16;
  header[52] = (timestamp & 0xFF000000) >> 24;
}

/**
 * Returns a 64-byte header to attempt to mine with.
 */
uint64_t* getWork(UCPClient& ucpClient, uint32_t timestamp) {
  uint64_t* header = new uint64_t[8];
  ucpClient.copyHeaderToHash((byte*)header);
  embedTimestampInHeader((uint8_t*)header, timestamp);
  return header;
}

void minerWorker(UCPClient& ucpClient, int deviceIndex, std::string deviceName,
                 int threadsPerBlock, int blockSize) {
  int ret = cudaSetDevice(deviceIndex);
  if (ret != cudaSuccess) {
    sprintf(outputBuffer,
            "CUDA encountered an error while setting the device to %d:%d",
            deviceIndex, ret);
    std::cerr << outputBuffer << std::endl;
    Log::error(outputBuffer);
  }

  cudaDeviceReset();

  // Don't have GPU busy-wait on GPU
  ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  cudaError_t e = cudaGetLastError();
  sprintf(outputBuffer, "[GPU #%d] : Last error: %s\n", deviceIndex, cudaGetErrorString(e));
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);

  // Run initialization of device before beginning timer
  uint64_t* header = getWork(ucpClient, (uint32_t)time(0));

  cudaError_t cudaStatus;
  // Device memory
  uint32_t *dev_nonceStart = 0;
  uint64_t *dev_header = 0;
  uint32_t *dev_nonceResult = 0;
  uint64_t *dev_hashStart = 0;

  // Allocate GPU buffers for nonce result and header
  cudaStatus = cudaMalloc((void**)&dev_nonceStart, 1 * sizeof(uint32_t));
  cudaStatus = cudaMalloc((void**)&dev_header, 8 * sizeof(uint64_t));
  cudaStatus = cudaMalloc((void**)&dev_nonceResult, 1 * sizeof(uint32_t));
  cudaStatus = cudaMalloc((void**)&dev_hashStart, 1 * sizeof(uint64_t));
  if (cudaStatus != cudaSuccess) {
    sprintf(outputBuffer, "cudaMalloc failed!");
    std::cerr << outputBuffer << endl;
    Log::error(outputBuffer);
    cudaError_t e = cudaGetLastError();
    sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
    std::cerr << outputBuffer << endl;
    Log::error(outputBuffer);
  }

  unsigned long long startTime = time(0);
  unsigned long long lastKnownDowntime = 0;
  uint32_t nonceResult[1] = {0};
  uint64_t hashStart[1] = {0};

  unsigned long long hashes = 0;

  uint32_t count = 0;

  int numLines = 0;

  // Mining loop
  while (true) {
    vprintf("top of mining loop\n");
    count++;
    long timestamp = (long)time(0);
    delete[] header;
    vprintf("Getting work...\n");
    header = getWork(ucpClient, timestamp);
    vprintf("Getting job id...\n");
    int jobId = ucpClient.getJobId();
    count++;
    vprintf("Running kernel...\n");
    cudaStatus = grindNonces(dev_nonceStart, dev_header, dev_nonceResult, dev_hashStart,
                             nonceResult, hashStart, header, deviceIndex,
                             threadsPerBlock, blockSize);
    vprintf("Kernel finished...\n");
    if (cudaStatus != cudaSuccess) {
      cudaError_t e = cudaGetLastError();
      sprintf(
          outputBuffer,
          "Error from running grindNonces: %s\nThis often occurs when a GPU "
          "overheats, has an unstable overclock, or has too aggressive launch "
          "parameters\nfor the vBlake kernel.\nYou can try using less "
          "aggressive settings, like:\n-tpb 256 -bs 256\nAnd try increasing "
          "these numbers until you hit instability issues again.",
          cudaGetErrorString(e));
      std::cerr << outputBuffer << std::endl;
      Log::error(outputBuffer);
      promptExit(-1);
    }
	
    if (lastKnownDowntime < ucpClient.getLastDowntime()) {
	  lastKnownDowntime = ucpClient.getLastDowntime();
	  hashes = 0;
	  startTime = time(0) - 1;
    }

    unsigned long long totalTime = time(0) - startTime;
    hashes += (blockSize * threadsPerBlock * WORK_PER_THREAD);

    double hashSpeed = (double)hashes;
    hashSpeed /= (totalTime * 1024 * 1024);

    if (count % 10 == 0) {
      int validShares = ucpClient.getValidShares();
      int invalidShares = ucpClient.getInvalidShares();
      int totalAccountedForShares = invalidShares + validShares;
      int totalSubmittedShares = ucpClient.getSentShares();
      int unaccountedForShares = totalSubmittedShares - totalAccountedForShares;

      double percentage = ((double)validShares) / totalAccountedForShares;
      percentage *= 100;
      // printf("[GPU #%d (%s)] : %f MH/second    valid shares: %d/%d/%d
      // (%.3f%%)\n", deviceIndex, deviceName.c_str(), hashSpeed,
      // validShares, totalAccountedForShares, totalSubmittedShares,
      // percentage);

      printf("[GPU #%d (%s)] : %0.2f MH/s shares: %d/%d/%d (%.3f%%)\n",
             deviceIndex, deviceName.c_str(), hashSpeed, validShares,
             totalAccountedForShares, totalSubmittedShares, percentage);
    }

    if (nonceResult[0] != 0x01000000 && nonceResult[0] != 0) {
      uint32_t nonce = *nonceResult;
      nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) |
               ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));

      ucpClient.submitWork(jobId, timestamp, nonce);

      nonceResult[0] = 0;

      char line[100];

      // Hash coming from GPU is reversed
      uint64_t hashFlipped = 0;
      hashFlipped |= (hashStart[0] & 0x00000000000000FF) << 56;
      hashFlipped |= (hashStart[0] & 0x000000000000FF00) << 40;
      hashFlipped |= (hashStart[0] & 0x0000000000FF0000) << 24;
      hashFlipped |= (hashStart[0] & 0x00000000FF000000) << 8;
      hashFlipped |= (hashStart[0] & 0x000000FF00000000) >> 8;
      hashFlipped |= (hashStart[0] & 0x0000FF0000000000) >> 24;
      hashFlipped |= (hashStart[0] & 0x00FF000000000000) >> 40;
      hashFlipped |= (hashStart[0] & 0xFF00000000000000) >> 56;

#if CPU_SHARES
      sprintf(line, "\tShare Found @ 2^24! {%#018llx} [nonce: %#08lx]",
              hashFlipped, nonce);
#else
      sprintf(line, "\tShare Found @ 2^32 by GPU #%d! {%#018llx} [nonce: %#08lx]",
              deviceIndex, hashFlipped, nonce);
#endif

      std::cout << line << std::endl;
      vprintf("Logging\n");
      Log::info(line);
      vprintf("Done logging\n");
      vprintf("Made line\n");

      numLines++;

      // Uncomment these lines to get access to this data for display purposes
      /*
      long long extraNonce = ucpClient.getStartExtraNonce();
      int jobId = ucpClient.getJobId();
      int encodedDifficulty = ucpClient.getEncodedDifficulty();
      string previousBlockHashHex = ucpClient.getPreviousBlockHash();
      string merkleRoot = ucpClient.getMerkleRoot();
      */
    }
    vprintf("About to restart loop...\n");
  }

  printf("Resetting device...\n");
  cudaFree(dev_nonceStart);
  cudaFree(dev_header);
  cudaFree(dev_nonceResult);
  cudaFree(dev_hashStart);
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
  }
  printf("Done resetting device...\n");

  getchar();
}

void startMining(UCPClient& ucpClient, std::set<int>& deviceList,
                 int threadsPerBlock, int blockSize) {
  int version, ret;
  ret = cudaDriverGetVersion(&version);
  if (ret != cudaSuccess) {
    sprintf(outputBuffer, "Error when getting CUDA driver version: %d", ret);
    std::cout << outputBuffer << std::endl;
    Log::error(outputBuffer);
    promptExit(-1);
  }

  int runtimeVersion;
  ret = cudaRuntimeGetVersion(&runtimeVersion);
  if (ret != cudaSuccess) {
    sprintf(outputBuffer, "Error when getting CUDA runtime version: %d", ret);
    std::cout << outputBuffer << std::endl;
    Log::error(outputBuffer);
    promptExit(-1);
  }

  int deviceCount;
  ret = cudaGetDeviceCount(&deviceCount);
  if (ret != cudaSuccess) {
    sprintf(outputBuffer, "Error when getting CUDA device count: %d", ret);
    std::cout << outputBuffer << std::endl;
    Log::error(outputBuffer);
    promptExit(-1);
  }

  cudaDeviceProp deviceProp;

#if NVML
  char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
  nvmlSystemGetDriverVersion(driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
#else
  char driver[] = "???.?? (NVML NOT ENABLED)";
#endif

  sprintf(outputBuffer, "CUDA Version: %.1f", ((float)version / 1000));
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);
  sprintf(outputBuffer, "CUDA Runtime Version: %d", runtimeVersion);
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);
  sprintf(outputBuffer, "NVidia Driver Version: %s", driver);
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);
  sprintf(outputBuffer, "CUDA Devices: %d", deviceCount);
  std::cout << outputBuffer << std::endl << std::endl;
  Log::info(outputBuffer);

  if (deviceList.size() == 0) {
    sprintf(outputBuffer, "No Devices specified, mining on all %d CUDA devices...",
            deviceCount);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
	
	for (int device = 0; device < deviceCount; device++) {
		deviceList.insert(device);
	}
  }
  
  std::vector<std::thread> minerThreads;
  for (int deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++) {
    ret = cudaGetDeviceProperties(&deviceProp, deviceIndex);
    if (ret != cudaSuccess) {
      sprintf(outputBuffer,
              "An error occurred while getting the CUDA device properties: %d",
              ret);
      std::cerr << outputBuffer << std::endl;
      Log::error(outputBuffer);
    }

    sprintf(outputBuffer, "Device #%d (%s):", deviceIndex, deviceProp.name);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Clock Rate:              %d MHz",
            (deviceProp.clockRate / 1024));
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Is Integrated:           %s",
            (deviceProp.integrated == 0 ? "false" : "true"));
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Compute Capability:      %d.%d",
            deviceProp.major, deviceProp.minor);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Kernel Concurrency:      %d",
            deviceProp.concurrentKernels);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Max Grid Size:           %d x %d x %d",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Max Threads per Block:   %d",
            deviceProp.maxThreadsPerBlock);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Registers per Block:     %d",
            deviceProp.regsPerBlock);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Registers per SM:        %d",
            deviceProp.regsPerMultiprocessor);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Processor Count:         %d",
            deviceProp.multiProcessorCount);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Shared Memory/Block:     %zd",
            deviceProp.sharedMemPerBlock);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Shared Memory/Proc:      %zd",
            deviceProp.sharedMemPerMultiprocessor);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);
    sprintf(outputBuffer, "    Warp Size:               %d",
            deviceProp.warpSize);
    std::cout << outputBuffer << std::endl;
    Log::info(outputBuffer);

    if (deviceList.count(deviceIndex) == 0) {
      continue;
    }

    // No effect if NVML is not enabled.
    readyNVML(deviceIndex);

    minerThreads.push_back(std::thread(&minerWorker, std::ref(ucpClient),
                                       deviceIndex, deviceProp.name,
                                       threadsPerBlock, blockSize));
  }

  sprintf(outputBuffer, "Mining on Device%s: %s\n", deviceList.size() > 1 ? "s" : "",
          absl::StrJoin(deviceList, ",").c_str());
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);

if (minerThreads.size() != 0) {
  for (std::thread& t : minerThreads) {
    t.join();
  }
} else {
  sprintf(outputBuffer, "No valid devices were listed! Available CUDA devices: %d!", deviceCount);
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);
  exit(-1);
  }
}