// VeriBlock PoW GPU Miner
// Copyright 2017-2018 VeriBlock, Inc.
// All rights reserved.
// https://www.veriblock.org
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.

#pragma once

#include <set>

#include "cuda_runtime.h"
#include "UCPClient.h"

extern cudaError_t grindNonces(uint32_t *dev_nonceStart, uint64_t* dev_header, uint32_t* dev_nonceResult,
                               uint64_t* dev_hashStart, uint32_t *nonceResult, uint64_t *hashStart, const
                               uint64_t *header, int deviceIndex, int threadsPerBlock, int blockSize);

void startMining(UCPClient& ucpClient, std::set<int>& deviceList,
                 int threadsPerBlock, int blockSize);