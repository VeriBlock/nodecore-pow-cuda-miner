cc_binary(
    name = "nodecore-pow-cuda",
    srcs = [
        "main.cpp",
    ],
    deps = [
        ":miner",
    ],
    linkopts = [
        "-pthread",
        "-lcuda",
        "-lcudart",
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
)

genrule(
    name = "cuda_kernel",
    srcs = [
        "kernel.cu",
        "Constants.h",
        "Log.h",
    ],
    outs = ["kernel.o"],
    cmd = "nvcc -gencode=arch=compute_50,code=\\\"sm_50,compute_50\\\"" +
          " -gencode=arch=compute_52,code=\\\"sm_52,compute_52\\\" " +
          " -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\" " +
          " -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" " +
          " -I. -O3 -Xcompiler -Wall  -D_FORCE_INLINES  --ptxas-options=\"-v\"" +
          " --compiler-options=\"-fPIC\"" +
          " --maxrregcount=64 -c $(location :kernel.cu) -o $@"
)
