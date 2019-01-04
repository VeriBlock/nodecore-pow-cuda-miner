cc_binary(
    name = "nodecore-pow-cuda-miner",
    srcs = [
        "main.cpp",
    ],
    deps = [
        ":miner",
    ],
    copts = [
        "-Iexternal/cuda_local/include",
    ],
    linkopts = [
        "-pthread",
    ],
)

cc_library(
    name = "miner",
    srcs = [
        "kernel.o",
        "Miner.cpp",
    ],
    hdrs = [
        "picojson.h",
        "Constants.h",
        "Log.h",
        "Miner.h",
        "UCPClient.h",
    ],
    copts = [
        "-Iexternal/cuda_local/include",
    ],
    deps = [
        "@cuda_local//:cuda",
    ],
)

genrule(
    name = "cuda_kernel",
    srcs = [
        "kernel.cu",
        "Constants.h",
        "Log.h",
        "@cuda_local//:cuda",
    ],
    outs = ["kernel.o"],
    cmd = "nvcc -gencode=arch=compute_50,code=\\\"sm_50,compute_50\\\"" +
          " -gencode=arch=compute_52,code=\\\"sm_52,compute_52\\\" " +
          " -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\" " +
          " -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" " +
          " -O3 -Xcompiler -Wall --std=c++11  -D_FORCE_INLINES  --ptxas-options=\"-v\"" +
          " --compiler-options=\"-fPIC\",\"-Iexternal/cuda_local/include\"" +
          " --maxrregcount=64 -c $(location :kernel.cu) -o $@"
)
