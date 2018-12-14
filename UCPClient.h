// VeriBlock PoW GPU Miner
// Copyright 2017-2018 VeriBlock, Inc.
// All rights reserved.
// https://www.veriblock.org
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.

#pragma once
#pragma comment(lib, "Ws2_32.lib")

#ifdef _WIN32
#include <WS2tcpip.h>
#include <WinSock2.h>
#include <Windows.h>
#elif __linux__
#include <arpa/inet.h>
#include <errno.h>
#include <unistd.h>
#define boolean bool
#define byte uint8_t
#define SOCKET_ERROR -1
#define SOCKADDR_IN sockaddr_in
#define SOCKADDR sockaddr
#define SOCKET int
#endif
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include "Constants.h"
#include "Log.h"
#include "picojson.h"

#define SCK_VERSION2 0x0202

#define UPDATE_FREQUENCY_MS 500.0
#define MESSAGE_BUFFER_SIZE 16 * 1024
#define BLOCK_HASH_SIZE_BYTES 24

#define BLOCK_NUM_SIZE_BYTES 4
#define VERSION_SIZE_BYTES 2
#define PREVIOUS_BLOCK_HASH_SIZE_BYTES 12
#define SECOND_PREVIOUS_BLOCK_HASH_SIZE_BYTES 9
#define THIRD_PREVIOUS_BLOCK_HASH_SIZE_BYTES 9
#define TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES 16

#define MAX_PACKET_DATA 1460

#define VERIBLOCK_BLOCK_HEADER_SIZE 64

// Offset for reading Capability BITFLAG
#define MINING_AUTH_OFFSET 0
#define MINING_SUBSCRIBE_OFFSET 1
#define MINING_SUBMIT_OFFSET 2
#define MINING_UNSUBSCRIBE_OFFSET 3
#define MINING_RESET_ACK_OFFSET 4
#define MINING_MEMPOOL_UPDATE_ACK_OFFSET 5

#define ERROR_INITIAL_SETUP_FAILED -1

using namespace std;

class UCPClient {
  string storedHost;
  short storedPort;
  string storedUsername;
  string storedPassword;

  int validShares = 0;
  int invalidShares = 0;
  int sentShares = 0;
  int lastAcknowledgement = 0;
  boolean successfulConnect = false;
  boolean workAvailable = false;

  byte headerToHash[VERIBLOCK_BLOCK_HEADER_SIZE];
  unsigned int jobId = 0xFFFFFFFF;
  int64_t startExtraNonce = 0xFFFFFFFFFFFFFFFF;
  unsigned int encodedDifficulty;

  int blockHeight = -1;
  string previousBlockHash = "...";
  string secondPreviousBlockHash = "...";
  string thirdPreviousBlockHash = "...";
  string merkleRoot = "...";

  char outputBuffer[2048];

  byte miningTarget[BLOCK_HASH_SIZE_BYTES];

  thread runThread;

  SOCKET ucpServerSocket;

 private:
  enum ServerCommand {
    Capabilities,
    MiningAuthFailure,
    MiningAuthSuccess,
    MiningSubscribeFailure,
    MiningSubscribeSuccess,
    MiningSubmitFailure,
    MiningSubmitSuccess,
    MiningJob,
    MiningMempoolUpdate,
    Unsupported,
    Invalid
  };

  void promptExit(int exitCode) {
    cout << "Exiting in 10 seconds..." << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    exit(exitCode);
  }

  double submitOffset = 2500000;

  string getMiningSubmitString(unsigned int jobId, unsigned int timestamp,
                               unsigned int nonce) {
    picojson::object top;
    picojson::object requestIdObj;
    picojson::object jobIdObj;
    picojson::object nTimeObj;
    picojson::object nonceObj;
    picojson::object extraNonceObj;

    top["command"] = picojson::value("MINING_SUBMIT");

    requestIdObj["type"] = picojson::value("REQUEST_ID");
    requestIdObj["data"] = picojson::value(submitOffset++);

    jobIdObj["type"] = picojson::value("JOB_ID");
    jobIdObj["data"] = picojson::value((double)jobId);

    nTimeObj["type"] = picojson::value("TIMESTAMP");
    nTimeObj["data"] = picojson::value((double)timestamp);

    nonceObj["type"] = picojson::value("NONCE");
    nonceObj["data"] = picojson::value((double)nonce);

    extraNonceObj["type"] = picojson::value("EXTRA_NONCE");
    extraNonceObj["data"] = picojson::value(startExtraNonce);

    top["request_id"] = picojson::value(requestIdObj);
    top["job_id"] = picojson::value(jobIdObj);
    top["nTime"] = picojson::value(nTimeObj);
    top["nonce"] = picojson::value(nonceObj);
    top["extra_nonce"] = picojson::value(extraNonceObj);

    return picojson::value(top).serialize();
  }

  string getMiningAuthString(string username, string password) {
    picojson::object top;
    picojson::object requestIdObj;
    picojson::object usernameObj;
    picojson::object passwordObj;

    top["command"] = picojson::value("MINING_AUTH");

    requestIdObj["type"] = picojson::value("REQUEST_ID");
    requestIdObj["data"] = picojson::value(1.0);

    usernameObj["type"] = picojson::value("USERNAME");
    usernameObj["data"] = picojson::value(username);

    passwordObj["type"] = picojson::value("PASSWORD");
    passwordObj["data"] = picojson::value(password);

    top["request_id"] = picojson::value(requestIdObj);
    top["username"] = picojson::value(usernameObj);
    top["password"] = picojson::value(passwordObj);

    return picojson::value(top).serialize();
  }

  string getMiningSubscribeString() {
    picojson::object top;
    picojson::object requestIdObj;
    picojson::object updateFrequencyMS;

    top["command"] = picojson::value("MINING_SUBSCRIBE");

    requestIdObj["type"] = picojson::value("REQUEST_ID");
    requestIdObj["data"] = picojson::value(2.0);

    updateFrequencyMS["type"] = picojson::value("FREQUENCY_MS");
    updateFrequencyMS["data"] = picojson::value(UPDATE_FREQUENCY_MS);

    top["request_id"] = picojson::value(requestIdObj);
    top["update_frequency_ms"] = picojson::value(updateFrequencyMS);

    return picojson::value(top).serialize();
  }

  ServerCommand getCommandType(string line) {
    picojson::value jsonMaster;
    string err = picojson::parse(jsonMaster, line);

    if (!err.empty()) {
      sprintf(
          outputBuffer,
          "An error has been encountered while reading a command! Command: %s",
          line.c_str());
      cout << outputBuffer << endl;
      Log::error(outputBuffer);
      return Invalid;
    }

    if (!jsonMaster.is<picojson::object>()) {
      sprintf(outputBuffer, "Top-level line %s is not a JSON object!",
              line.c_str());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      return Invalid;
    }

    const picojson::value::object& jsonMasterObj =
        jsonMaster.get<picojson::object>();

    picojson::value::object::const_iterator jsonCommandIterator =
        jsonMasterObj.find("command");

    if (jsonCommandIterator != jsonMasterObj.end()) {
      picojson::value commandValue = jsonCommandIterator->second;
      string command = commandValue.to_str();

      if (command.compare("CAPABILITIES") == 0) {
        return Capabilities;
      } else if (command.compare("MINING_AUTH_FAILURE") == 0) {
        return MiningAuthFailure;
      } else if (command.compare("MINING_AUTH_SUCCESS") == 0) {
        return MiningAuthSuccess;
      } else if (command.compare("MINING_SUBSCRIBE_FAILURE") == 0) {
        return MiningSubscribeFailure;
      } else if (command.compare("MINING_SUBSCRIBE_SUCCESS") == 0) {
        return MiningSubscribeSuccess;
      } else if (command.compare("MINING_SUBMIT_FAILURE") == 0) {
        return MiningSubmitFailure;
      } else if (command.compare("MINING_SUBMIT_SUCCESS") == 0) {
        return MiningSubmitSuccess;
      } else if (command.compare("MINING_JOB") == 0) {
        return MiningJob;
      } else if (command.compare("MINING_MEMPOOL_UPDATE") == 0) {
        return MiningMempoolUpdate;
      } else {
        return Unsupported;
      }
    } else {
      return Invalid;
    }
  }

  bool setupLoginAndSubscribe(string username, string password) {
    char message[MESSAGE_BUFFER_SIZE];
    long success = recv(ucpServerSocket, message, sizeof(message),
#ifdef _WIN32
                        NULL
#else
                        0
#endif
    );

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(outputBuffer,
              "Reading from socket during setup resulted in an error %d",
              WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(outputBuffer,
              "Reading from socket during setup resulted in an error %d",
              success);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      return false;
    }

    if (getCommandType(message) != Capabilities) {
      sprintf(outputBuffer,
              "Server did not send its capabilities at the beginning of the "
              "setup process! Instead, it sent the command: %s",
              message);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      return false;
    }

    picojson::value capabilitiesCommand;
    string err = picojson::parse(capabilitiesCommand, message);
    if (!err.empty()) {
      sprintf(outputBuffer,
              "An error has occurred while attempting to read the server "
              "capabilities: %s",
              err.c_str());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      return false;
    }

    if (!capabilitiesCommand.is<picojson::object>()) {
      sprintf(outputBuffer,
              "Provided JSON (%s) does not contain a top-level JSON-object!",
              message);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
    }

    const picojson::value::object& capabilitiesCommandObj =
        capabilitiesCommand.get<picojson::object>();

    picojson::value::object::const_iterator capabilitiesIter =
        capabilitiesCommandObj.find("capabilities");

    if (capabilitiesIter != capabilitiesCommandObj.end()) {
      picojson::value capabilitiesSection = capabilitiesIter->second;
      const picojson::value::object& capabilitiesObject =
          capabilitiesSection.get<picojson::object>();
      picojson::value::object::const_iterator capabilitiesInternalIter =
          capabilitiesObject.find("data");
      if (capabilitiesInternalIter != capabilitiesObject.end()) {
        picojson::value bitflagValue = capabilitiesInternalIter->second;
        string bitflag = bitflagValue.to_str();
        char MINING_AUTH = bitflag[bitflag.length() - 1 - MINING_AUTH_OFFSET];
        char MINING_SUBSCRIBE =
            bitflag[bitflag.length() - 1 - MINING_SUBSCRIBE_OFFSET];
        char MINING_SUBMIT =
            bitflag[bitflag.length() - 1 - MINING_SUBMIT_OFFSET];
        char MINING_UNSUBSCRIBE =
            bitflag[bitflag.length() - 1 - MINING_UNSUBSCRIBE_OFFSET];
        char MINING_RESET_ACK =
            bitflag[bitflag.length() - 1 - MINING_RESET_ACK_OFFSET];
        char MINING_MEMPOOL_UPDATE_ACK =
            bitflag[bitflag.length() - 1 - MINING_MEMPOOL_UPDATE_ACK_OFFSET];

        boolean capabilitiesCorrect = true;

        if (MINING_AUTH != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support MINING_AUTH according "
                  "to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (MINING_SUBSCRIBE != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support MINING_SUBSCRIBE "
                  "according to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (MINING_SUBMIT != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support MINING_SUBMIT "
                  "according to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (MINING_UNSUBSCRIBE != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support MINING_UNSUBSCRIBE "
                  "according to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (MINING_RESET_ACK != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support MINING_RESET_ACK "
                  "according to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (MINING_MEMPOOL_UPDATE_ACK != '1') {
          capabilitiesCorrect = false;
          sprintf(outputBuffer,
                  "The specified server does not support "
                  "MINING_MEMPOOL_UPDATE_ACK according to its bitflag (%s)",
                  bitflag.c_str());
          cerr << outputBuffer << endl;
          Log::error(outputBuffer);
        }

        if (capabilitiesCorrect) {
          sprintf(outputBuffer,
                  "The specified server supports all necessary commands "
                  "(bitflag: %s)",
                  bitflag.c_str());
          cout << outputBuffer << endl;
          Log::info(outputBuffer);
        } else {
          return false;
        }
      } else {
        sprintf(outputBuffer,
                "The server did not send a valid capabilities command!");
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }
    }

    string authenticate = getMiningAuthString(username, password) + "\n";
    success = send(ucpServerSocket, authenticate.c_str(),
                   (int)strlen(authenticate.c_str()), 0);

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(outputBuffer,
              "Sending authentication string failed with error %d",
              WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(outputBuffer,
              "Sending authentication string failed with error %d", errno);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      return false;
    }

    success = recv(ucpServerSocket, message, sizeof(message),
#ifdef _WIN32
                   NULL
#else
                   0
#endif
    );

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(
          outputBuffer,
          "Reading from socket during authentication resulted in an error %d",
          WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(
          outputBuffer,
          "Reading from socket during authentication resulted in an error %d",
          errno);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      return false;
    }

    if (getCommandType(message) == MiningAuthSuccess) {
      sprintf(outputBuffer, "Successfully authenticated to server!");
      cout << outputBuffer << endl;
      Log::info(outputBuffer);
    } else {
      picojson::value authenticationResponseCommand;
      string err = picojson::parse(authenticationResponseCommand, message);
      if (!err.empty()) {
        sprintf(outputBuffer,
                "An error has occurred while attempting to read the server "
                "authentication response: %s",
                err.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }

      if (!authenticationResponseCommand.is<picojson::object>()) {
        sprintf(outputBuffer,
                "Provided JSON (%s) does not contain a top-level JSON-object!",
                err.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }

      const picojson::value::object& authenticationResponseCommandObj =
          authenticationResponseCommand.get<picojson::object>();

      picojson::value::object::const_iterator authenticationResponseIter =
          authenticationResponseCommandObj.find("reason");

      if (authenticationResponseIter !=
          authenticationResponseCommandObj.end()) {
        picojson::value reasonValue = authenticationResponseIter->second;
        string reason = reasonValue.to_str();
        sprintf(outputBuffer, "Unable to authenticate to the server: %s",
                reason.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      } else {
        sprintf(outputBuffer,
                "The server did not send a valid mining authentication "
                "response command!");
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }
    }

    string subscribe = getMiningSubscribeString() + "\n";
    success = send(ucpServerSocket, subscribe.c_str(),
                   (int)strlen(subscribe.c_str()), 0);

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(outputBuffer, "Sending subscription string failed with error %d",
              WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(outputBuffer, "Sending subscription string failed with error %d",
              errno);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      return false;
    }

    success = recv(ucpServerSocket, message, sizeof(message),
#ifdef _WIN32
                   NULL
#else
                   0
#endif
    );

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(outputBuffer,
              "Reading from socket during subscription resulted in an error %d",
              WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(outputBuffer,
              "Reading from socket during subscription resulted in an error %d",
              errno);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      return false;
    }

    if (getCommandType(message) == MiningSubscribeSuccess) {
      sprintf(outputBuffer, "Successfully subscribed to server!");
      cout << outputBuffer << endl;
      Log::info(outputBuffer);
    } else {
      picojson::value subscriptionResponseCommand;
      string err = picojson::parse(subscriptionResponseCommand, message);
      if (!err.empty()) {
        sprintf(
            outputBuffer,
            "Reading from socket during subscription resulted in an error %s",
            err.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }

      if (!subscriptionResponseCommand.is<picojson::object>()) {
        sprintf(outputBuffer,
                "Provided JSON (%s) does not contain a top-level JSON-object!",
                message);
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }

      const picojson::value::object& subscriptionResponseCommandObj =
          subscriptionResponseCommand.get<picojson::object>();

      picojson::value::object::const_iterator subscriptionResponseIter =
          subscriptionResponseCommandObj.find("reason");

      if (subscriptionResponseIter != subscriptionResponseCommandObj.end()) {
        picojson::value reasonValue = subscriptionResponseIter->second;
        string reason = reasonValue.to_str();
        sprintf(outputBuffer, "Unable to subscribe to server: %s",
                reason.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      } else {
        sprintf(outputBuffer,
                "The server did not send a valid mining subscription response "
                "command!");
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        return false;
      }
    }
    return true;
  }

  picojson::value extractDataValueFromJSONById(string JSON, string id) {
    JSON.erase(remove(JSON.begin(), JSON.end(), '\n'), JSON.end());
    picojson::value command;
    string err = picojson::parse(command, JSON);
    if (!err.empty()) {
      sprintf(outputBuffer,
              "An error has occurred while attempting to read the server "
              "response: %s",
              err.c_str());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      throw invalid_argument("Provided JSON (" + JSON + ") is not valid!");
    }

    if (!command.is<picojson::object>()) {
      sprintf(outputBuffer,
              "Provided JSON (%s) does not contain a top-level JSON-object!",
              JSON.c_str());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      throw invalid_argument("Provided JSON (" + JSON +
                             ") does not contain an object on its top level!");
    }

    const picojson::value::object& commandObj = command.get<picojson::object>();

    picojson::value::object::const_iterator jobIdIter = commandObj.find(id);

    if (jobIdIter != commandObj.end()) {
      picojson::value jobIdSection = jobIdIter->second;
      const picojson::value::object& jobIdSectionObject =
          jobIdSection.get<picojson::object>();
      picojson::value::object::const_iterator jobIdInternalIter =
          jobIdSectionObject.find("data");

      if (jobIdInternalIter != jobIdSectionObject.end()) {
        picojson::value jobIdValue = jobIdInternalIter->second;
        return jobIdValue;
      } else {
        sprintf(outputBuffer, "The JSON blob (%s) does not contain an id %s!",
                JSON.c_str(), id.c_str());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        throw invalid_argument("The JSON blob (" + JSON +
                               ") does not contain an id " + id + "!");
      }
    } else {
      sprintf(outputBuffer, "The JSON blob (%s) does not contain an id %s!",
              JSON.c_str(), id.c_str());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      throw invalid_argument("The JSON blob (" + JSON +
                             ") does not contain an id " + id + "!");
    }
  }

  int getDataIntFromJSONById(string JSON, string id) {
    picojson::value value = extractDataValueFromJSONById(JSON, id);
    return (unsigned int)value.get<double>();
  }

  string getDataStringFromJSONById(string JSON, string id) {
    picojson::value value = extractDataValueFromJSONById(JSON, id);
    return value.get<string>();
  }

  string validLowerCaseHex = "0123456789abcdef";
  bool isLowerCaseHexCharacter(char toTest) {
    return validLowerCaseHex.find(toTest) != string::npos;
  }

  int getValueFromLowerCaseHex(char toRoute) {
    if (!isLowerCaseHexCharacter(toRoute)) {
      throw invalid_argument("The provided character " + string(1, toRoute) +
                             " is not valid!");
    }

    // Process upper- and lower-case hex with ASCII offsets
    if (toRoute >= 48 && toRoute <= 57) {
      return toRoute - 48;
    } else if (toRoute >= 97 && toRoute <= 102) {
      return toRoute - 87;
    } else {
      throw invalid_argument(
          "The provided character " + string(1, toRoute) +
          " is invalid and was not rejected in preliminary hex checks!");
    }
  }

  byte extractByteFromHex(string hex, int byteIndex) {
    if (hex.length() % 2 != 0) {
      throw invalid_argument("Provided hex " + hex + " is not valid!");
    }

    char hi = tolower(hex.at(byteIndex + 0));
    char lo = tolower(hex.at(byteIndex + 1));

    if (!isLowerCaseHexCharacter(hi)) {
      throw invalid_argument("Hex character " + string(1, hi) +
                             " is not valid!");
    }
    if (!isLowerCaseHexCharacter(lo)) {
      throw invalid_argument("Hex character " + string(1, lo) +
                             " is not valid!");
    }

    int hiVal = getValueFromLowerCaseHex(hi) * 16;
    int loVal = getValueFromLowerCaseHex(lo) * 1;

    if (hiVal + loVal > 255) {
      throw invalid_argument("The provided hex (" + hex + ") at index " +
                             to_string(byteIndex) + " is not a valid byte!");
    }

    return (byte)(hiVal + loVal);
  }

  void fillNull(char* toFill, int length) {
    for (int i = 0; i < length; i++) {
      toFill[i] = '\0';
    }
  }

  void cyclicRun() {
    char message[MESSAGE_BUFFER_SIZE];
    char extraMessage[MESSAGE_BUFFER_SIZE];
    for (;;) {
      fillNull(message, MESSAGE_BUFFER_SIZE);
      fillNull(extraMessage, MESSAGE_BUFFER_SIZE);
      long success = recv(ucpServerSocket, message, sizeof(message),
#ifdef _WIN32
                          NULL
#else
                          0
#endif
      );

      int cursor = success;

      boolean check1 = message[cursor - 1] == '\n';

      if ((!check1)) {
        Log::info(
            "Message from server was chopped into multiple packets, reading "
            "additional packets...");
      }

      while ((!check1)) {
        try {
          long result = recv(ucpServerSocket, message + cursor, 1,
#ifdef _WIN32
                             NULL
#else
                             0
#endif
          );
        } catch (char* e) {
          // ex
        }

        cursor++;

        if ((message != nullptr) && (message[0] == '\0')) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } else {
          check1 = message[cursor - 1] == '\n';
        }
      }

      Log::info("Processing command:");
      Log::info(message);

      if (success == SOCKET_ERROR) {
#ifdef _WIN32
        sprintf(outputBuffer,
                "Reading from socket during normal operations resulted in an "
                "error %d, are the server credentials correct?",
                WSAGetLastError());
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        closesocket(ucpServerSocket);
        WSACleanup();
#else
        sprintf(outputBuffer,
                "Reading from socket during normal operations resulted in an "
                "error %d, are the server credentials correct?",
                errno);
        cerr << outputBuffer << endl;
        Log::error(outputBuffer);
        close(ucpServerSocket);
#endif
        return;
      }

      ServerCommand commandType = getCommandType(message);

      if (commandType == MiningJob) {
        unsigned int blockVersion;
        byte previousBlockHashBytes[PREVIOUS_BLOCK_HASH_SIZE_BYTES];
        byte
            secondPreviousBlockHashBytes[SECOND_PREVIOUS_BLOCK_HASH_SIZE_BYTES];
        byte thirdPreviousBlockHashBytes[THIRD_PREVIOUS_BLOCK_HASH_SIZE_BYTES];
        byte topLevelMerkleRootBytes[TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES];
        unsigned int timestamp;

        jobId = getDataIntFromJSONById(message, "job_id");

        blockVersion = getDataIntFromJSONById(message, "block_version");

        previousBlockHash =
            getDataStringFromJSONById(message, "previous_block_hash");
        for (int i = BLOCK_HASH_SIZE_BYTES - PREVIOUS_BLOCK_HASH_SIZE_BYTES;
             i < BLOCK_HASH_SIZE_BYTES; i++) {
          previousBlockHashBytes[i - (BLOCK_HASH_SIZE_BYTES -
                                      PREVIOUS_BLOCK_HASH_SIZE_BYTES)] =
              extractByteFromHex(previousBlockHash, i * 2);
        }

        secondPreviousBlockHash =
            getDataStringFromJSONById(message, "second_previous_block_hash");
        for (int i =
                 BLOCK_HASH_SIZE_BYTES - SECOND_PREVIOUS_BLOCK_HASH_SIZE_BYTES;
             i < BLOCK_HASH_SIZE_BYTES; i++) {
          secondPreviousBlockHashBytes
              [i - (BLOCK_HASH_SIZE_BYTES -
                    SECOND_PREVIOUS_BLOCK_HASH_SIZE_BYTES)] =
                  extractByteFromHex(secondPreviousBlockHash, i * 2);
        }
        thirdPreviousBlockHash =
            getDataStringFromJSONById(message, "third_previous_block_hash");
        for (int i =
                 BLOCK_HASH_SIZE_BYTES - THIRD_PREVIOUS_BLOCK_HASH_SIZE_BYTES;
             i < BLOCK_HASH_SIZE_BYTES; i++) {
          thirdPreviousBlockHashBytes[i -
                                      (BLOCK_HASH_SIZE_BYTES -
                                       THIRD_PREVIOUS_BLOCK_HASH_SIZE_BYTES)] =
              extractByteFromHex(thirdPreviousBlockHash, i * 2);
        }

        merkleRoot = getDataStringFromJSONById(message, "merkle_root");
        for (int i = 0; i < TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES; i++) {
          topLevelMerkleRootBytes[i] = extractByteFromHex(merkleRoot, i * 2);
        }

        blockHeight = getDataIntFromJSONById(message, "block_index");

        timestamp = getDataIntFromJSONById(message, "timestamp");

        encodedDifficulty = getDataIntFromJSONById(message, "difficulty");

        string miningTargetHex =
            getDataStringFromJSONById(message, "mining_target");
        for (int i = 0; i < BLOCK_HASH_SIZE_BYTES; i++) {
          miningTarget[i] = extractByteFromHex(miningTargetHex, i * 2);
        }

        picojson::value value =
            extractDataValueFromJSONById(message, "extra_nonce_start");
        int64_t test = value.get<int64_t>();
        startExtraNonce = test;

        for (int i = 0; i < 4; i++) {
          headerToHash[i] = (blockHeight >> ((3 - i) * 8));
        }

        for (int i = 0; i < 2; i++) {
          headerToHash[i + 4] = (blockVersion >> ((1 - i) * 8));
        }

        memcpy(headerToHash + 6, previousBlockHashBytes,
               PREVIOUS_BLOCK_HASH_SIZE_BYTES);
        memcpy(headerToHash + 18, secondPreviousBlockHashBytes,
               SECOND_PREVIOUS_BLOCK_HASH_SIZE_BYTES);
        memcpy(headerToHash + 27, thirdPreviousBlockHashBytes,
               THIRD_PREVIOUS_BLOCK_HASH_SIZE_BYTES);
        memcpy(headerToHash + 36, topLevelMerkleRootBytes,
               TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES);

        for (int i = 0; i < 4; i++) {
          headerToHash[52 + i] = (timestamp >> ((3 - i) * 8));
        }

        for (int i = 0; i < 4; i++) {
          headerToHash[56 + i] = (encodedDifficulty >> ((3 - i) * 8));
        }

        workAvailable = true;

      } else if (commandType == MiningMempoolUpdate) {
        byte topLevelMerkleRoot[TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES];
        string topLevelMerkleRootHex =
            getDataStringFromJSONById(message, "new_merkle_root");
        for (int i = 0; i < TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES; i++) {
          topLevelMerkleRoot[i] =
              extractByteFromHex(topLevelMerkleRootHex, i * 2);
        }

        memcpy(headerToHash + 36, topLevelMerkleRoot,
               TOP_LEVEL_MERKLE_ROOT_SIZE_BYTES);

        jobId = getDataIntFromJSONById(message, "job_id");
      } else if (commandType == MiningSubmitSuccess) {
        Log::info("Successfully mined a share!");
        validShares++;
        lastAcknowledgement = sentShares;
      } else if (commandType == MiningSubmitFailure) {
        invalidShares++;
        string failureReason = getDataStringFromJSONById(message, "reason");
        sprintf(outputBuffer,
                "Submitting a share failed for the following reason: %s",
                failureReason.c_str());
        cerr << outputBuffer << endl;
        lastAcknowledgement = sentShares;
        Log::error(outputBuffer);
      }

      continue;
    }
  }

 public:
  UCPClient(string host, short port, string username, string password) {
    storedHost = host;
    storedPort = port;
    storedUsername = username;
    storedPassword = password;
    boolean result = connectToServer(true);
    if (!result) {
      if (BENCHMARK) {
        sprintf(
            outputBuffer,
            "Ignoring initial serer connection setup failure as benchmarking "
            "mode is enabled...");
        cout << outputBuffer << endl;
        Log::warn(outputBuffer);
      } else {
        promptExit(-1);
      }
    }
  }

  boolean wasSuccessful() { return successfulConnect; }

  void copyHeaderToHash(byte* destination) {
    memcpy(destination, headerToHash, VERIBLOCK_BLOCK_HEADER_SIZE);
  }

  unsigned int getJobId() { return jobId; }

  string getPreviousBlockHash() { return previousBlockHash; }

  string getMerkleRoot() { return merkleRoot; }

  unsigned int getEncodedDifficulty() { return encodedDifficulty; }

  unsigned long long getStartExtraNonce() { return startExtraNonce; }

  void copyMiningTarget(byte* destination) {
    memcpy(destination, miningTarget, BLOCK_HASH_SIZE_BYTES);
  }

  boolean hasWorkReady() { return workAvailable; }

  void start() {
    runThread = thread([this] { this->cyclicRun(); });
  }

  int getValidShares() { return validShares; }

  int getInvalidShares() { return invalidShares; }

  int getSentShares() { return sentShares; }

  void submitWork(unsigned int jobId, unsigned int timestamp,
                  unsigned int nonce) {
    string miningSubmit = getMiningSubmitString(jobId, timestamp, nonce) + "\n";
    long success = send(ucpServerSocket, miningSubmit.c_str(),
                        (int)strlen(miningSubmit.c_str()), 0);

    sentShares++;

    if (lastAcknowledgement + 5 < sentShares) {
      sprintf(outputBuffer, "Pool server appears unresponsive! Exiting...");
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
#ifdef _WIN32
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      close(ucpServerSocket);
#endif
      promptExit(-1);
    }

    if (success == SOCKET_ERROR) {
#ifdef _WIN32
      sprintf(outputBuffer,
              "Sending mining submit / share submission string failed with "
              "error %d",
              WSAGetLastError());
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      closesocket(ucpServerSocket);
      WSACleanup();
#else
      sprintf(outputBuffer,
              "Sending mining submit / share submission string failed with "
              "error %d",
              errno);
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      close(ucpServerSocket);
#endif
      promptExit(-1);
      return;
    }
  }

  bool reconnect() {
    bool success = connectToServer(false);

    if (success) {
      // Reset sent shares that previously failed during a previous connection
      sentShares = validShares + invalidShares;
    }

    return success;
  }

  bool connectToServer(bool startThread) {
    string DUMMY;

#ifdef _WIN32
    long successfulStartup;
    WSAData WinSockData;
    WORD DLLVersion;
    DLLVersion = MAKEWORD(2, 1);
    successfulStartup = WSAStartup(DLLVersion, &WinSockData);

    if (!(successfulStartup == 0)) {
      sprintf(outputBuffer, "Unable to correctly perform WSAStartup!");
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      return false;
    }
#endif

    string RESPONSE;
    string CONVERTER;

    SOCKADDR_IN ADDRESS;

#ifdef _WIN32
    ucpServerSocket = socket(AF_INET, SOCK_STREAM, NULL);
#else
    ucpServerSocket = socket(AF_INET, SOCK_STREAM, 0);
#endif
    inet_pton(AF_INET, storedHost.c_str(), &(ADDRESS.sin_addr.s_addr));
    ADDRESS.sin_family = AF_INET;
    ADDRESS.sin_port = htons(storedPort);

    connect(ucpServerSocket, (SOCKADDR*)&ADDRESS, sizeof(ADDRESS));
    bool successfulStart =
        UCPClient::setupLoginAndSubscribe(storedUsername, storedPassword);

    if (!successfulStart) {
      sprintf(outputBuffer, "Initial server connection and setup failed!");
      cerr << outputBuffer << endl;
      Log::error(outputBuffer);
      successfulConnect = false;
      return false;
    } else {
      successfulConnect = true;
    }

    if (startThread) {
      start();
    }

    int tries = 0;
    while (!workAvailable) {
      tries++;
      this_thread::sleep_for(std::chrono::milliseconds(2));

      if (tries > 4000) {
        return false;
      }
    }

    return true;
  }
};
