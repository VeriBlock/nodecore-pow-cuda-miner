/usr/local/cuda/bin/nvcc -gencode=arch=compute_61,code=\"sm_61,compute_61\" -I/usr/local/cuda/include -I. -O3 -Xcompiler -Wall  -D_FORCE_INLINES  --ptxas-options="-v" --maxrregcount=64 -o nodecore_pow_cuda kernel.cu -std=c++11


