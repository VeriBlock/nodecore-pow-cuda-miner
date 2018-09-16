# NodeCore PoW CUDA Miner
## Contents
[Introduction](#introduction)

[Getting Started](#getting_started)

[Command Line Arguments](#command_line)

[Performance Tuning](#performance_tuning)

[Compiling on Linux](#compile_linux)

[Compiling on Windows](#compile_windows)


<br><br>
## Introduction
<a name="introduction"></a>
This is a CUDA GPU miner which mines the vBlake algorithm (for the [VeriBlock](http://veriblock.org/) blockchain). It connects using the new UCP protocol supported by the built-in pool software of the VeriBlock NodeCore software. 

Running this software requires the latest (398.xx or 399.xx on Windows, 396.xx on Linux) NVidia drivers, as well as one or more Maxwell-series, Pascal-series, Volta-series, or Turing-series GPUs.

Currently, each instance of the GPU miner supports mining on a single card.

<br><br>
## Getting Started
<a name="getting_started"></a>
Download (or Compile [Linux](#compile_linux) or [Windows](#compile_windows)) the latest NodeCore NVidia CUDA Miner, and configure a .bat (Windows) or .sh (Linux) file to start up the miner with appropriate command line arguments that, at a minimum, specify the pool you want to mine to (IP:port), and the username (generally an address) that you would like to mine with.

<br><br>
## Command Line Arguments
<a name="command_line"></a>
```VeriBlock vBlake GPU CUDA Miner v1.0
Required Arguments:
-o <poolAddress>           The pool address to mine to in the format host:port
-u <username>              The username (often an address) used at the pool

Optional Arguments:
-p <password>              The miner/worker password to use on the pool
-d <deviceNum>             The ordinal of the device to use (default 0)
-tpb <threadPerBlock>      The threads per block to use with the Blake kernel (default 1024)
-bs <blockSize>            The blocksize to use with the vBlake kernel (default 512)
-l <enableLogging>         Whether to log to a file (default true)
-v <enableVerboseOutput>   Whether to enable verbose output for debugging (default false)
```

Example command line:
```
VeriBlock-NodeCore-PoW-CUDA -u VHT36jJyoVFN7ap5Gu77Crua2BMv5j -o 94.130.64.18:8501 -l false
```


<br><br>
## Performance Tuning
<a name="performance_tuning"></a>
This GPU miner lets the user change their "Threads per Block" and "Block Size" parameters to tune the program's kernel launch dimensions to their specific hardware. The default values are a block-size of 512 and a threads-per-block of 1024. Generally higher values represent more load on the card and higher hashrates up to a saturation point (at which point larger numbers will generally decrease performance). The default parameters are a good starting point for most modern mid- and high-range cards. If you want to change these parameters, they can be specified as startup arguments:

```
VeriBlock-NodeCore-PoW-CUDA -u VHT36jJyoVFN7ap5Gu77Crua2BMv5j -o 94.130.64.18:8501 -l false -bs 512 -tpb 1024
```

### Selecting Parameters

When you start-up your GPU miner, it will print out the details of all available CUDA devices on your computer:

```
Device #0 (TITAN V):
    Clock Rate:              1420 MHz
    Is Integrated:           false
    Compute Capability:      7.0
    Kernel Concurrency:      1
    Max Grid Size:           2147483647 x 65535 x 65535
    Max Threads per Block:   1024
    Registers per Block:     65536
    Registers per SM:        65536
    Processor Count:         80
    Shared Memory/Block:     49152
    Shared Memory/Proc:      98304
    Warp Size:               32
```

For Maxwell, Pascal, Volta, and Turing, **the maximum threads-per-block that can be set is 1024** (as you can see in the above device info). 

There is no (practical) limit to the block-size, but most cards will not run with a block-size larger than 8192. 

Threads-per-block should generally be a multiple of the warp size (shown above). Suggested Values: 256, 512, 1024.

Block-size should generally be a multiple of the processor count a power of 2 (128, 256, 512...), or a multiple of a power of 2 (5 * 128 = 640, 5 * 256 = 1280...).


<br><br>
## Supported NVidia GPU Architectures
<a name="architectures"></a>
Use the following table to select the correct version of the miner to use (or compile) based on the card(s) you intend to mine on:

<table>
  <tr>
    <td>
      GPU Family
    </td>
    <td>
Example GPUs
    </td>
    <td>
      Architecture
    </td>
    <td>
      Code Generation Architecture
     </td>
  </tr>
  <tr>
    <td rowspan="2">
      Maxwell
    </td>
    <td>
      750 Ti
    </td>
     <td>
        sm_50
    </td>
    <td>compute_50;sm_50
    </td>
  </tr>
  <tr>
    <td>
950<br>960<br>970<br>980<br>980 Ti<br>Titan X(M)
    </td>
    <td>
      sm_52
    </td>
    <td>
      compute_52;sm_52
    </td>
  </tr>
  <tr>
    <td>
      Pascal
    </td>
    <td>
      1050 Ti<br>1060<br>1070<br>1080<br>1080 Ti<br>Titan X(p)
    </td>
    <td>
    sm_60
    </td>
    <td>
      compute_60;sm_60
    </td>
  </tr>
  <tr>
    <td>
      Volta
    </td>
    <td>
      V100<br>Titan V
    </td>
    <td>
        sm_70
    </td>
    <td>
      compute_70;sm_70
    </td>
  </tr>
  <tr>
    <td>
      Turing
    </td>
    <td>
      2060, 2070, 2080, 2080 Ti
    </td>
    <td>
      sm_72
    </td>
    <td>
      compute_80;sm_80
    </td>
  </tr>
  </table>


<br><br>
## Compiling on Linux (Ubuntu 16.04)
<a name="compile_linux"></a>
         
### 1. Install CUDA 9.2 Toolkit
Get the latest version of the CUDA toolkit for your system here: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64 (note: installation examples below are using CUDA 9.2.148; you may need to change them to a different version in the future).

If you already have a version of CUDA installed, ensure that the compiler version is 9.2:
```
max@ubuntu ~ # nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:07:04_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```
If you do not have 9.2, install 9.2 with the instructions below.

```
sudo apt-get update

# Get the latest .deb installation file; you should update the URL to the latest version of CUDA 9.2
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb

# Install the official NVidia CUDA Repo
sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb

# Install Keys Required to Use the Repo
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# Install CUDA (includes compiler and library metapackages)
sudo apt-get update
sudo apt-get -y install cuda

# Default installation doesn't create soft-links to NVCC
ln -s /usr/local/cuda-9.2/bin/nvcc /usr/bin/nvcc
```

### 2. Compile

After cloning the git repository and cd'ing into it, you can (optionally) edit build.sh and remove/add gencode architectures and other compilation parameters (like the maximum register count allowed, or specifying an alternate directory for resources for the linker). When you're ready to build: 

```
chmod a+x build.sh
./build.sh
```

If you don't see any errors, then the build was successful and you can run the newly-made nodecore_pow_cuda binary.

### 3. (Optional) Install

If you want to be able to run the CUDA miner from any directory on your system, copy the compiled binary to your bin folder:

```
cp nodecore_pow_cuda /usr/bin/
```

<br><br>
## Compiling on Windows (7, 10)
<a name="compile_windows"></a>
### 1. Install Visual Studio 2015
See https://stackoverflow.com/questions/44290672/how-to-download-visual-studio-community-edition-2015-not-2017 for help with getting 2015 instead of 2017.

### 2. Install the CUDA 9.2 Toolkit
Downlaod and install the appropriate version of the CUDA 9.2 toolkit from https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64 and make sure to allow it to install the Visual Studio 2015 plugin.

### 3. Create a new CUDA 9.2 Runtime Project
File -> New -> Project...
If you do not see the CUDA 9.2 Runtime project template, ensure that you installed the Visual Studio plugin as part of the 9.2 toolkit installer (and re-run the installer if you did not).

![compile1](http://mirror1.veriblock.org/windows-compile/compile-windows-1.png "Compile Step 1")

### 4. Import the CUDA miner Source Code
Add all .h and .cu files from this repo to the project:

![compile2](http://mirror1.veriblock.org/windows-compile/compile-windows-2.png "Compile Step 2")

### 5. Set the Appropriate Build Architecture
Set the project properties to compile for the correct architecture (see architecture table at top). The CUDA miner uses the lop3 PTX ISA instruction, so it cannot be compiled for any architecture previous to sm_50.

![compile3](http://mirror1.veriblock.org/windows-compile/compile-windows-3.png "Compile Step 3")

![compile4](http://mirror1.veriblock.org/windows-compile/compile-windows-4.png "Compile Step 4")

### 6. Set the Build Configuration to Release x64

![compile5](http://mirror1.veriblock.org/windows-compile/compile-windows-5.png "Compile Step 5")

### 7. Compile/Execute
Either click on "Local Windows Debugger" to build+run, or use another Build method and then navigate to the x64 binary output directory in your project structure and run the compiled .exe. If you run it without giving it any command-line arguments (the default if you just run the Local Windows Debugger), you should see:

![compile6](http://mirror1.veriblock.org/windows-compile/compile-windows-6.png "Compile Step 6")

This indicates that the program was built correctly. Try running it from the command line (or a .bat file) with arguments, or configure your project to pass in command line arguments for the pool IP:port, username, and any other additional optional command line arguments you would like to use.





