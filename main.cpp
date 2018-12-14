// VeriBlock PoW GPU Miner
// Copyright 2017-2018 VeriBlock, Inc.
// All rights reserved.
// https://www.veriblock.org
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.

#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#include <VersionHelpers.h>
#elif __linux__
#include <sys/socket.h>
#include <netdb.h>
#endif

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

#include "Constants.h"
#include "Log.h"
#include "Miner.h"
#include "UCPClient.h"

// TODO(mks): Rework logging.
bool verboseOutput = false;
char outputBuffer[8192];

void vprintf(char* toprint) {
  if (verboseOutput) {
    printf(toprint);
  }
}

void promptExit(int exitCode) {
  std::cout << "Exiting in 10 seconds..." << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  exit(exitCode);
}

#ifdef _WIN32
static WSADATA g_wsa_data;

char net_init(void) {
  return (WSAStartup(MAKEWORD(2, 2), &g_wsa_data) == NO_ERROR);
}

void net_deinit(void) { WSACleanup(); }
#else
char net_init(void) { return 1; }

void net_deinit(void) {}
#endif

string net_dns_resolve(const char* hostname) {
  struct addrinfo hints, *results, *item;
  int status;
  char ipstr[INET6_ADDRSTRLEN];

  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC; /* AF_INET6 to force version */
  hints.ai_socktype = SOCK_STREAM;

  if ((status = getaddrinfo(hostname, NULL, &hints, &results)) != 0) {
    fprintf(stderr, "failed to resolve hostname \"%s\": %s", hostname,
            gai_strerror(status));
    return "invalid hostname";
  }

  printf("IP addresses for %s:\n\n", hostname);

  string ret;

  for (item = results; item != NULL; item = item->ai_next) {
    void* addr;
    char* ipver;

    /* get pointer to the address itself */
    /* different fields in IPv4 and IPv6 */
    if (item->ai_family == AF_INET) /* address is IPv4 */
    {
      struct sockaddr_in* ipv4 = (struct sockaddr_in*)item->ai_addr;
      addr = &(ipv4->sin_addr);
      ipver = "IPv4";
    } else /* address is IPv6 */
    {
      struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)item->ai_addr;
      addr = &(ipv6->sin6_addr);
      ipver = "IPv6";
    }

    /* convert IP to a string and print it */
    inet_ntop(item->ai_family, addr, ipstr, sizeof ipstr);
    printf("  %s: %s\n", ipver, ipstr);
    ret = ipstr;
  }

  freeaddrinfo(results);
  return ret;
}

void printHelpAndExit() {
  printf("VeriBlock vBlake GPU CUDA Miner v1.0\n");
  printf("Required Arguments:\n");
  printf(
      "-o <poolAddress>           The pool address to mine to in the format "
      "host:port\n");
  printf(
      "-u <username>              The username (often an address) used at the "
      "pool\n");
  printf("Optional Arguments:\n");
  printf(
      "-p <password>              The miner/worker password to use on the "
      "pool\n");
  printf(
      "-d <deviceList>            Comma-separated list of device numbers to "
      "use (default all).\n");
  printf(
      "-tpb <threadPerBlock>      The threads per block to use with the Blake "
      "kernel (default %d)\n",
      DEFAULT_THREADS_PER_BLOCK);
  printf(
      "-bs <blockSize>            The blocksize to use with the vBlake kernel "
      "(default %d)\n",
      DEFAULT_BLOCK_SIZE);
  printf(
      "-l <enableLogging>         Whether to log to a file (default true)\n");
  printf(
      "-v <enableVerboseOutput>   Whether to enable verbose output for "
      "debugging (default false)\n");
  printf("\n");
  printf("Example command line:\n");
  printf(
      "VeriBlock-NodeCore-PoW-CUDA -u VHT36jJyoVFN7ap5Gu77Crua2BMv5j -o "
      "testnet-pool-gpu.veriblock.org:8501 -l false\n");
  promptExit(0);
}

int main(int argc, char* argv[]) {
  // Check for help argument (only -h)
  for (int i = 1; i < argc; i++) {
    char* argument = argv[i];

    if (!strcmp(argument, "-h")) {
      printHelpAndExit();
    }
  }

  if (argc % 2 != 1) {
    sprintf(outputBuffer, "GPU miner must be provided valid argument pairs!");
    std::cerr << outputBuffer << std::endl;
    printHelpAndExit();
  }

  string hostAndPort = "";  //  "94.130.64.18:8501";
  string username = "";     // "VGX71bcRsEh4HZzhbA9Nj7GQNH5jGw";
  string password = "";

  int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
  int blockSize = DEFAULT_BLOCK_SIZE;
  std::set<int> deviceList;
  if (argc > 1) {
    for (int i = 1; i < argc; i += 2) {
      char* argument = argv[i];
      printf("%s\n", argument);
      if (argument[0] == '-' && argument[1] == 'd') {
        std::string arg(argv[i + 1]);
        std::set<std::string> devices = absl::StrSplit(arg, ',');
        for (const string& d : devices) {
          int i;
          if (!absl::SimpleAtoi(d, &i)) {
            sprintf(outputBuffer, "Invalid GPU index: %s\n", d.c_str());
            std::cerr << outputBuffer << std::endl;
            exit(1);
          }
          deviceList.insert(i);
        }
      } else if (!strcmp(argument, "-o")) {
        hostAndPort = string(argv[i + 1]);
      } else if (!strcmp(argument, "-u")) {
        username = string(argv[i + 1]);
      } else if (!strcmp(argument, "-p")) {
        password = string(argv[i + 1]);
      } else if (!strcmp(argument, "-tpb")) {
        threadsPerBlock = std::stoi(argv[i + 1]);
      } else if (!strcmp(argument, "-bs")) {
        blockSize = std::stoi(argv[i + 1]);
      } else if (!strcmp(argument, "-l")) {
        // to lower case conversion
        for (int j = 0; j < strlen(argv[i + 1]); j++) {
          argv[i + 1][j] = tolower(argv[i + 1][j]);
        }
        if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t")) {
          Log::setEnabled(true);
        } else {
          Log::setEnabled(false);
        }
      } else if (!strcmp(argument, "-v")) {
        // to lower case conversion
        for (int j = 0; j < strlen(argv[i + 1]); j++) {
          argv[i + 1][j] = tolower(argv[i + 1][j]);
        }
        if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t")) {
          verboseOutput = true;
        } else {
          verboseOutput = false;
        }
      }
    }
  } else {
    printHelpAndExit();
  }

  if (HIGH_RESOURCE) {
    sprintf(outputBuffer, "Resource Utilization: HIGH");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  } else {
    sprintf(outputBuffer, "Resource Utilization: LOW");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  }

  if (NVML) {
    sprintf(outputBuffer, "NVML Status: ENABLED");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  } else {
    sprintf(outputBuffer, "NVML Status: DISABLED");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  }

  if (CPU_SHARES) {
    sprintf(outputBuffer, "Share Type: CPU");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  } else {
    sprintf(outputBuffer, "Share Type: GPU");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  }

  if (BENCHMARK) {
    sprintf(outputBuffer, "Benchmark Mode: ENABLED");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  } else {
    sprintf(outputBuffer, "Benchmark Mode: DISABLED");
    std::cerr << outputBuffer << std::endl;
    Log::info(outputBuffer);
  }

#ifdef _WIN32
  HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
#else
#endif

  if (hostAndPort.compare("") == 0) {
    string error =
        "You must specify a host in the command line arguments! Example: \n-o "
        "127.0.0.1:8501 or localhost:8501";
    std::cerr << error << std::endl;
    Log::error(error);
    promptExit(-1);
  }

  if (username.compare("") == 0) {
    string error =
        "You must specify a username in the command line arguments! Example: "
        "\n-u V5bLSbCqj9VzQR3MNANqL13YC2tUep";
    std::cerr << error << std::endl;
    Log::error(error);
    promptExit(-1);
  }

  string host = hostAndPort.substr(0, hostAndPort.find(":"));
  // GetHostByName
  net_init();
  host = net_dns_resolve(host.c_str());
  net_deinit();

  string portString = hostAndPort.substr(hostAndPort.find(":") + 1);

  // Ensure that port is numeric
  if (portString.find_first_not_of("1234567890") != string::npos) {
    string error =
        "You must specify a host in the command line arguments! Example: \n-o "
        "127.0.0.1:8501 or localhost:8501";
    std::cerr << error << std::endl;
    Log::error(error);
    promptExit(-1);
  }

  int port = std::stoi(portString);

  sprintf(
      outputBuffer,
      "Attempting to mine to pool %s:%d with username %s and password %s...",
      host.c_str(), port, username.c_str(), password.c_str());
  std::cout << outputBuffer << std::endl;
  Log::info(outputBuffer);

  UCPClient ucpClient(host, port, username, password);
  startMining(ucpClient, deviceList, threadsPerBlock, blockSize);

  return 0;
}
