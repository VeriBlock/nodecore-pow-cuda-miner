CUDA_LOCAL_INCLUDE = select({
    "@bazel_tools//src/conditions:linux_x86_64":
        ["-Iexternal/cuda_local_linux/include"],
    "@bazel_tools//src/conditions:windows": [
        "/Iexternal/cuda_local_windows/include",
    ],
})

cc_binary(
    name = "nodecore-pow-cuda-miner",
    srcs = [
        "main.cpp",
    ],
    deps = [
        ":miner",
        "@abseil-cpp//absl/strings",
    ],
    copts = [
    ] + CUDA_LOCAL_INCLUDE,
    linkopts = [
        "-pthread",
    ],
)

CUDA_LOCAL = select({
    "@bazel_tools//src/conditions:linux_x86_64": [
        "@cuda_local_linux//:cuda",
    ],
    "@bazel_tools//src/conditions:windows": [
        "@cuda_local_windows//:cuda",
    ],
})

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
    ] + CUDA_LOCAL_INCLUDE,
    deps = [
        "@abseil-cpp//absl/strings",
    ] + CUDA_LOCAL,
)

genrule(
    name = "cuda_kernel",
    srcs = [
        "kernel.cu",
        "Constants.h",
        "Log.h",
    ] + CUDA_LOCAL,
    outs = ["kernel.o"],
    cmd = "nvcc -gencode=arch=compute_50,code=\\\"sm_50,compute_50\\\"" +
          " -gencode=arch=compute_52,code=\\\"sm_52,compute_52\\\" " +
          " -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\" " +
          " -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" " +
          " -O3 -Xcompiler -Wall  -D_FORCE_INLINES  --ptxas-options=\"-v\"" +
          " --compiler-options=" + select({
              "@bazel_tools//src/conditions:linux_x86_64":
                  "\" -fPIC\"," + "\" -Iexternal/cuda_local_linux/include",
              "@bazel_tools//src/conditions:windows":
	          "\" /MDd\"," + "\" /Iexternal/cuda_local_windows/include",
          }) + "\"" +
          " --maxrregcount=64 -c $(location :kernel.cu) -o $@",
)
