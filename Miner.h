// VeriBlock PoW GPU Miner
// Copyright 2017-2018 VeriBlock, Inc.
// All rights reserved.
// https://www.veriblock.org
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.

#pragma once

#include "UCPClient.h"
#include "cuda_runtime.h"

extern cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart,
                               const uint64_t *header, int deviceToUse,
                               int threadsPerBlock, int blockSize);

void startMining(UCPClient &ucpClient, int deviceToUse, int threadsPerBlock,
                 int blockSize);
