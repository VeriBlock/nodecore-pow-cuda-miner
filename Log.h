#pragma once
#pragma comment(lib, "Ws2_32.lib")

#include <fstream>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#elif __linux__
#define boolean bool
#define byte uint8_t
#endif

#define LOG_FILE "cuda-miner.log"

using namespace std;

// TODO(mks): Rework logging.
extern bool verboseOutput;
extern char outputBuffer[8192];
extern void vprintf(char* toprint);
extern void promptExit(int exitCode);

class Log {
 public:
  static inline bool& log() {
    static bool opt = true;
    return opt;
  }

  static void setEnabled(bool enabled) { log() = enabled; }

  static void info(char* toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile.write(toLog, strlen(toLog));
      logfile.write("\n", 2);
      logfile.close();
    }
  }

  static void info(string& toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile << toLog << endl;
      logfile.close();
    }
  }

  static void warn(char* toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile.write("[WARN] ", 8);
      logfile.write(toLog, strlen(toLog));
      logfile.write("\n", 2);
      logfile.close();
    }
  }

  static void warn(string& toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile << "[WARN] " << toLog << endl;
      logfile.close();
    }
  }

  static void error(char* toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile.write("[ERROR] ", 9);
      logfile.write(toLog, strlen(toLog));
      logfile.write("\n", 2);
      logfile.close();
    }
  }

  static void error(string& toLog) {
    if (log()) {
      static std::ofstream logfile;
      logfile = ofstream();
      logfile.open(LOG_FILE, ios::app | ios::out);
      logfile << "[ERROR] " << toLog << endl;
      logfile.close();
    }
  }
};
