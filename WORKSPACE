load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "abseil-cpp",
    remote = "https://github.com/abseil/abseil-cpp.git",
    commit = "455dc17ba1af9635f0b60155bc565bc572a1e722",
)

new_local_repository(
    name = "cuda_local",
    path = "/usr/local/cuda-9.2",
    build_file = "BUILD.cuda-local",
)
