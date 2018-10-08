#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <ctime>
#include <stdio.h>

#include <stdlib.h>
#include "UCPClient.h"

#ifdef _WIN32
#include <Windows.h>
#include <VersionHelpers.h>
#elif __linux__
#include <sys/socket.h> 
#include <netdb.h>
#endif

#include <ctime>
#include "Log.h"
#include <sstream>
#include "Constants.h"

#if NVML
#include "nvml.h"
#endif

// #pragma comment(lib, "nvml.lib")
// #pragma comment(lib, "nvapi.lib")
// #pragma comment(lib, "nvapi64.lib")

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif

cudaStream_t cudastream;

uint32_t *blockHeadermobj = nullptr;
uint32_t *midStatemobj = nullptr;
uint32_t *nonceOutmobj = nullptr;

cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header);

__device__ __forceinline__
uint64_t vBlake(const uint64_t h0, const uint64_t h1, const uint64_t h2, const uint64_t h3, const uint64_t h4, const uint64_t h5, const uint64_t h6, const uint64_t h7)
{
	uint64_t result;
	asm(
		// v
		"{.reg .b32 r0;\n\t"
		".reg .b32 r1;\n\t"
		".reg .b32 r2;\n\t"
		".reg .b32 r3;\n\t"
		".reg .b32 r4;\n\t"
		".reg .b32 r5;\n\t"
		".reg .b32 r6;\n\t"
		".reg .b32 r7;\n\t"
		".reg .b32 r8;\n\t"
		".reg .b32 r9;\n\t"
		".reg .b32 r10;\n\t"
		".reg .b32 r11;\n\t"
		".reg .b32 r12;\n\t"
		".reg .b32 r13;\n\t"
		".reg .b32 r14;\n\t"
		".reg .b32 r15;\n\t"
		".reg .b32 r16;\n\t"
		".reg .b32 r17;\n\t"
		".reg .b32 r18;\n\t"
		".reg .b32 r19;\n\t"
		".reg .b32 r20;\n\t"
		".reg .b32 r21;\n\t"
		".reg .b32 r22;\n\t"
		".reg .b32 r23;\n\t"
		".reg .b32 r24;\n\t"
		".reg .b32 r25;\n\t"
		".reg .b32 r26;\n\t"
		".reg .b32 r27;\n\t"
		".reg .b32 r28;\n\t"
		".reg .b32 r29;\n\t"
		".reg .b32 r30;\n\t"
		".reg .b32 r31;\n\t"
		// header
		".reg .b32 r32;\n\t"
		".reg .b32 r33;\n\t"
		".reg .b32 r34;\n\t"
		".reg .b32 r35;\n\t"
		".reg .b32 r36;\n\t"
		".reg .b32 r37;\n\t"
		".reg .b32 r38;\n\t"
		".reg .b32 r39;\n\t"
		".reg .b32 r40;\n\t"
		".reg .b32 r41;\n\t"
		".reg .b32 r42;\n\t"
		".reg .b32 r43;\n\t"
		".reg .b32 r44;\n\t"
		".reg .b32 r45;\n\t"
		".reg .b32 r46;\n\t"
		".reg .b32 r47;\n\t"
		// temp
		".reg .b32 r48;\n\t"
		".reg .b32 r49;\n\t"
		".reg .b32 r50;\n\t"
		".reg .b32 r51;\n\t"
		"mov.u32 r0, 0xF107AD85;\n\t"
		"mov.u32 r1, 0x4BBF42C1;\n\t"
		"mov.u32 r16, 0xF006AD9D;\n\t"
		"mov.u32 r17, 0x4BBF42C1;\n\t"
		"mov.u32 r2, 0xB5AEB12E;\n\t"
		"mov.u32 r3, 0x5D11A8C3;\n\t"
		"mov.u32 r18, 0xB5AEB12E;\n\t"
		"mov.u32 r19, 0x5D11A8C3;\n\t"
		"mov.u32 r4, 0xC2774652;\n\t"
		"mov.u32 r5, 0xA64AB78D;\n\t"
		"mov.u32 r20, 0xC2774652;\n\t"
		"mov.u32 r21, 0xA64AB78D;\n\t"
		"mov.u32 r6, 0x4658F253;\n\t"
		"mov.u32 r7, 0xC6759572;\n\t"
		"mov.u32 r22, 0x4658F253;\n\t"
		"mov.u32 r23, 0xC6759572;\n\t"
		"mov.u32 r8, 0xCB891E56;\n\t"
		"mov.u32 r9, 0xB8864E79;\n\t"
		"mov.u32 r24, 0xCB891E56;\n\t"
		"mov.u32 r25, 0xB8864E79;\n\t"
		"mov.u32 r10, 0x29FB41A1;\n\t"
		"mov.u32 r11, 0x12ED593E;\n\t"
		"mov.u32 r26, 0x29FB41A1;\n\t"
		"mov.u32 r27, 0x12ED593E;\n\t"
		"mov.u32 r12, 0x3C60BAA8;\n\t"
		"mov.u32 r13, 0xB1DA3AB6;\n\t"
		"mov.u32 r28, 0x3C60BAA8;\n\t"
		"mov.u32 r29, 0xB1DA3AB6;\n\t"
		"mov.u32 r14, 0x1F954DED;\n\t"
		"mov.u32 r15, 0x6D20E50C;\n\t"
		"mov.u32 r30, 0x1F954DED;\n\t"
		"mov.u32 r31, 0x6D20E50C;\n\t"
		// header[0]
		"mov.b64 {r32, r33}, %1;\n\t"
		// header[1]
		"mov.b64 {r34, r35}, %2;\n\t"
		// header[2]
		"mov.b64 {r36, r37}, %3;\n\t"
		// header[3]
		"mov.b64 {r38, r39}, %4;\n\t"
		// header[4]
		"mov.b64 {r40, r41}, %5;\n\t"
		// header[5]
		"mov.b64 {r42, r43}, %6;\n\t"
		// header[6]
		"mov.b64 {r44, r45}, %7;\n\t"
		// header[7]
		"mov.b64 {r46, r47}, %8;\n\t"
		"xor.b32 r24, r24, 64;\n\t"
		"xor.b32 r28, r28, 0xFFFFFFFF;\n\t"
		"xor.b32 r29, r29, 0xFFFFFFFF;\n\t"
		/*
		* |------------------------[ROUND 0.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8,  r9}           |
		* |            v[ 5]            |           {r10, r11}           |
		* |            v[ 6]            |           {r12, r13}           |
		* |            v[ 7]            |           {r14, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r25}           |
		* |            v[13]            |           {r26, r27}           |
		* |            v[14]            |           {r28, r29}           |
		* |            v[15]            |           {r30, r31}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r25}
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r9;\n\t"
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r25}
		"xor.b32 r48, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r48, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r25}
		"xor.b32 r24, r24, r0;\n\t"
		"xor.b32 r25, r25, r1;\n\t"
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r25}
		"shf.r.wrap.b32 r48, r24, r25, 60;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 60;\n\t"
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r48}
		"add.cc.u32 r16, r16, r24;\n\t"
		"addc.u32 r17, r17, r48;\n\t"
		// A = {r0, r1}    B = {r8, r9}    C = {r16, r17}    D = {r24, r48}
		"xor.b32 r8, r8, r16;\n\t"
		"xor.b32 r9, r9, r17;\n\t"
		"shf.r.wrap.b32 r25, r8, r9, 43;\n\t"
		"shf.r.wrap.b32 r8, r9, r8, 43;\n\t"
		// A = {r0, r1}    B = {r8, r25}    C = {r16, r17}    D = {r24, r48}
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r25;\n\t"
		// A = {r0, r1}    B = {r8, r25}    C = {r16, r17}    D = {r24, r48}
		"xor.b32 r9, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r8, r25}    C = {r16, r17}    D = {r24, r48}
		"xor.b32 r24, r24, r0;\n\t"
		"xor.b32 r48, r48, r1;\n\t"
		"shf.r.wrap.b32 r9, r24, r48, 5;\n\t"
		"shf.r.wrap.b32 r24, r48, r24, 5;\n\t"
		// A = {r0, r1}    B = {r8, r25}    C = {r16, r17}    D = {r9, r24}
		"add.cc.u32 r16, r16, r9;\n\t"
		"addc.u32 r17, r17, r24;\n\t"
		// A = {r0, r1}    B = {r8, r25}    C = {r16, r17}    D = {r9, r24}
		"xor.b32 r8, r8, r16;\n\t"
		"xor.b32 r25, r25, r17;\n\t"
		"shf.r.wrap.b32 r48, r8, r25, 18;\n\t"
		"shf.r.wrap.b32 r8, r25, r8, 18;\n\t"

		// A = {r0, r1}    B = {r48, r8}    C = {r16, r17}    D = {r9, r24}
		"lop3.b32 r25, r0, r48, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r8, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r48, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r8, r17, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r8, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r48, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r8, r17, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r8, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r48, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r8, r17, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r8, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r48, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r8, r17, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r48, r8}    C = {r16, r17}    D = {r9, r24}
		/*
		* |------------------------[ROUND 0.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r10, r11}           |
		* |            v[ 6]            |           {r12, r13}           |
		* |            v[ 7]            |           {r14, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r24}           |
		* |            v[13]            |           {r26, r27}           |
		* |            v[14]            |           {r28, r29}           |
		* |            v[15]            |           {r30, r31}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r27}
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r11;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r27}
		"xor.b32 r25, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r25, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r27}
		"xor.b32 r26, r26, r2;\n\t"
		"xor.b32 r27, r27, r3;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r27}
		"shf.r.wrap.b32 r25, r26, r27, 60;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 60;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r25}
		"add.cc.u32 r18, r18, r26;\n\t"
		"addc.u32 r19, r19, r25;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r18, r19}    D = {r26, r25}
		"xor.b32 r10, r10, r18;\n\t"
		"xor.b32 r11, r11, r19;\n\t"
		"shf.r.wrap.b32 r27, r10, r11, 43;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 43;\n\t"
		// A = {r2, r3}    B = {r10, r27}    C = {r18, r19}    D = {r26, r25}
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r27;\n\t"
		// A = {r2, r3}    B = {r10, r27}    C = {r18, r19}    D = {r26, r25}
		"xor.b32 r11, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r10, r27}    C = {r18, r19}    D = {r26, r25}
		"xor.b32 r26, r26, r2;\n\t"
		"xor.b32 r25, r25, r3;\n\t"
		"shf.r.wrap.b32 r11, r26, r25, 5;\n\t"
		"shf.r.wrap.b32 r26, r25, r26, 5;\n\t"
		// A = {r2, r3}    B = {r10, r27}    C = {r18, r19}    D = {r11, r26}
		"add.cc.u32 r18, r18, r11;\n\t"
		"addc.u32 r19, r19, r26;\n\t"
		// A = {r2, r3}    B = {r10, r27}    C = {r18, r19}    D = {r11, r26}
		"xor.b32 r10, r10, r18;\n\t"
		"xor.b32 r27, r27, r19;\n\t"
		"shf.r.wrap.b32 r25, r10, r27, 18;\n\t"
		"shf.r.wrap.b32 r10, r27, r10, 18;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r18, r19}    D = {r11, r26}
		"lop3.b32 r27, r2, r25, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r10, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r25, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r10, r19, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r10, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r25, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r10, r19, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r10, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r25, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r10, r19, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r10, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r25, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r10, r19, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r18, r19}    D = {r11, r26}
		/*
		* |------------------------[ROUND 0.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r25, r10}           |
		* |            v[ 6]            |           {r12, r13}           |
		* |            v[ 7]            |           {r14, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r24}           |
		* |            v[13]            |           {r11, r26}           |
		* |            v[14]            |           {r28, r29}           |
		* |            v[15]            |           {r30, r31}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r29}
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r13;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r29}
		"xor.b32 r27, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r4, r27, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r29}
		"xor.b32 r28, r28, r4;\n\t"
		"xor.b32 r29, r29, r5;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r29}
		"shf.r.wrap.b32 r27, r28, r29, 60;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 60;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r27}
		"add.cc.u32 r20, r20, r28;\n\t"
		"addc.u32 r21, r21, r27;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r20, r21}    D = {r28, r27}
		"xor.b32 r12, r12, r20;\n\t"
		"xor.b32 r13, r13, r21;\n\t"
		"shf.r.wrap.b32 r29, r12, r13, 43;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 43;\n\t"
		// A = {r4, r5}    B = {r12, r29}    C = {r20, r21}    D = {r28, r27}
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r29;\n\t"
		// A = {r4, r5}    B = {r12, r29}    C = {r20, r21}    D = {r28, r27}
		"xor.b32 r13, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r12, r29}    C = {r20, r21}    D = {r28, r27}
		"xor.b32 r28, r28, r4;\n\t"
		"xor.b32 r27, r27, r5;\n\t"
		"shf.r.wrap.b32 r13, r28, r27, 5;\n\t"
		"shf.r.wrap.b32 r28, r27, r28, 5;\n\t"
		// A = {r4, r5}    B = {r12, r29}    C = {r20, r21}    D = {r13, r28}
		"add.cc.u32 r20, r20, r13;\n\t"
		"addc.u32 r21, r21, r28;\n\t"
		// A = {r4, r5}    B = {r12, r29}    C = {r20, r21}    D = {r13, r28}
		"xor.b32 r12, r12, r20;\n\t"
		"xor.b32 r29, r29, r21;\n\t"
		"shf.r.wrap.b32 r27, r12, r29, 18;\n\t"
		"shf.r.wrap.b32 r12, r29, r12, 18;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r20, r21}    D = {r13, r28}
		"lop3.b32 r29, r4, r27, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r12, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r27, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r12, r21, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r12, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r27, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r12, r21, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r12, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r27, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r12, r21, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r12, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r27, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r12, r21, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r20, r21}    D = {r13, r28}
		/*
		* |------------------------[ROUND 0.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r25, r10}           |
		* |            v[ 6]            |           {r27, r12}           |
		* |            v[ 7]            |           {r14, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r24}           |
		* |            v[13]            |           {r11, r26}           |
		* |            v[14]            |           {r13, r28}           |
		* |            v[15]            |           {r30, r31}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r31}
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r15;\n\t"
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r31}
		"xor.b32 r29, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r6, r29, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r31}
		"xor.b32 r30, r30, r6;\n\t"
		"xor.b32 r31, r31, r7;\n\t"
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r31}
		"shf.r.wrap.b32 r29, r30, r31, 60;\n\t"
		"shf.r.wrap.b32 r30, r31, r30, 60;\n\t"
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r29}
		"add.cc.u32 r22, r22, r30;\n\t"
		"addc.u32 r23, r23, r29;\n\t"
		// A = {r6, r7}    B = {r14, r15}    C = {r22, r23}    D = {r30, r29}
		"xor.b32 r14, r14, r22;\n\t"
		"xor.b32 r15, r15, r23;\n\t"
		"shf.r.wrap.b32 r31, r14, r15, 43;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 43;\n\t"
		// A = {r6, r7}    B = {r14, r31}    C = {r22, r23}    D = {r30, r29}
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r31;\n\t"
		// A = {r6, r7}    B = {r14, r31}    C = {r22, r23}    D = {r30, r29}
		"xor.b32 r15, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r14, r31}    C = {r22, r23}    D = {r30, r29}
		"xor.b32 r30, r30, r6;\n\t"
		"xor.b32 r29, r29, r7;\n\t"
		"shf.r.wrap.b32 r15, r30, r29, 5;\n\t"
		"shf.r.wrap.b32 r30, r29, r30, 5;\n\t"
		// A = {r6, r7}    B = {r14, r31}    C = {r22, r23}    D = {r15, r30}
		"add.cc.u32 r22, r22, r15;\n\t"
		"addc.u32 r23, r23, r30;\n\t"
		// A = {r6, r7}    B = {r14, r31}    C = {r22, r23}    D = {r15, r30}
		"xor.b32 r14, r14, r22;\n\t"
		"xor.b32 r31, r31, r23;\n\t"
		"shf.r.wrap.b32 r29, r14, r31, 18;\n\t"
		"shf.r.wrap.b32 r14, r31, r14, 18;\n\t"
		// A = {r6, r7}    B = {r29, r14}    C = {r22, r23}    D = {r15, r30}
		"lop3.b32 r31, r6, r29, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r14, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r29, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r14, r23, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r14, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r29, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r14, r23, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r14, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r29, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r14, r23, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r14, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r29, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r14, r23, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r29, r14}    C = {r22, r23}    D = {r15, r30}
		/*
		* |------------------------[ROUND 0.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r25, r10}           |
		* |            v[ 6]            |           {r27, r12}           |
		* |            v[ 7]            |           {r29, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r24}           |
		* |            v[13]            |           {r11, r26}           |
		* |            v[14]            |           {r13, r28}           |
		* |            v[15]            |           {r15, r30}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r30}
		"add.cc.u32 r0, r0, r25;\n\t"
		"addc.u32 r1, r1, r10;\n\t"
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r30}
		"xor.b32 r31, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r31, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r30}
		"xor.b32 r15, r15, r0;\n\t"
		"xor.b32 r30, r30, r1;\n\t"
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r30}
		"shf.r.wrap.b32 r31, r15, r30, 60;\n\t"
		"shf.r.wrap.b32 r15, r30, r15, 60;\n\t"
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r31}
		"add.cc.u32 r20, r20, r15;\n\t"
		"addc.u32 r21, r21, r31;\n\t"
		// A = {r0, r1}    B = {r25, r10}    C = {r20, r21}    D = {r15, r31}
		"xor.b32 r25, r25, r20;\n\t"
		"xor.b32 r10, r10, r21;\n\t"
		"shf.r.wrap.b32 r30, r25, r10, 43;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 43;\n\t"
		// A = {r0, r1}    B = {r25, r30}    C = {r20, r21}    D = {r15, r31}
		"add.cc.u32 r0, r0, r25;\n\t"
		"addc.u32 r1, r1, r30;\n\t"
		// A = {r0, r1}    B = {r25, r30}    C = {r20, r21}    D = {r15, r31}
		"xor.b32 r10, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r25, r30}    C = {r20, r21}    D = {r15, r31}
		"xor.b32 r15, r15, r0;\n\t"
		"xor.b32 r31, r31, r1;\n\t"
		"shf.r.wrap.b32 r10, r15, r31, 5;\n\t"
		"shf.r.wrap.b32 r15, r31, r15, 5;\n\t"
		// A = {r0, r1}    B = {r25, r30}    C = {r20, r21}    D = {r10, r15}
		"add.cc.u32 r20, r20, r10;\n\t"
		"addc.u32 r21, r21, r15;\n\t"
		// A = {r0, r1}    B = {r25, r30}    C = {r20, r21}    D = {r10, r15}
		"xor.b32 r25, r25, r20;\n\t"
		"xor.b32 r30, r30, r21;\n\t"
		"shf.r.wrap.b32 r31, r25, r30, 18;\n\t"
		"shf.r.wrap.b32 r25, r30, r25, 18;\n\t"
		// A = {r0, r1}    B = {r31, r25}    C = {r20, r21}    D = {r10, r15}
		"lop3.b32 r30, r0, r31, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r25, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r31, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r25, r21, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r25, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r31, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r25, r21, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r25, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r31, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r25, r21, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r25, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r31, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r25, r21, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r31, r25}    C = {r20, r21}    D = {r10, r15}
		/*
		* |------------------------[ROUND 0.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r31, r25}           |
		* |            v[ 6]            |           {r27, r12}           |
		* |            v[ 7]            |           {r29, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r24}           |
		* |            v[13]            |           {r11, r26}           |
		* |            v[14]            |           {r13, r28}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r24}
		"add.cc.u32 r2, r2, r27;\n\t"
		"addc.u32 r3, r3, r12;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r24}
		"xor.b32 r30, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r2, r30, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r24}
		"xor.b32 r9, r9, r2;\n\t"
		"xor.b32 r24, r24, r3;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r24}
		"shf.r.wrap.b32 r30, r9, r24, 60;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 60;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r30}
		"add.cc.u32 r22, r22, r9;\n\t"
		"addc.u32 r23, r23, r30;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r22, r23}    D = {r9, r30}
		"xor.b32 r27, r27, r22;\n\t"
		"xor.b32 r12, r12, r23;\n\t"
		"shf.r.wrap.b32 r24, r27, r12, 43;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 43;\n\t"
		// A = {r2, r3}    B = {r27, r24}    C = {r22, r23}    D = {r9, r30}
		"add.cc.u32 r2, r2, r27;\n\t"
		"addc.u32 r3, r3, r24;\n\t"
		// A = {r2, r3}    B = {r27, r24}    C = {r22, r23}    D = {r9, r30}
		"xor.b32 r12, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r27, r24}    C = {r22, r23}    D = {r9, r30}
		"xor.b32 r9, r9, r2;\n\t"
		"xor.b32 r30, r30, r3;\n\t"
		"shf.r.wrap.b32 r12, r9, r30, 5;\n\t"
		"shf.r.wrap.b32 r9, r30, r9, 5;\n\t"
		// A = {r2, r3}    B = {r27, r24}    C = {r22, r23}    D = {r12, r9}
		"add.cc.u32 r22, r22, r12;\n\t"
		"addc.u32 r23, r23, r9;\n\t"
		// A = {r2, r3}    B = {r27, r24}    C = {r22, r23}    D = {r12, r9}
		"xor.b32 r27, r27, r22;\n\t"
		"xor.b32 r24, r24, r23;\n\t"
		"shf.r.wrap.b32 r30, r27, r24, 18;\n\t"
		"shf.r.wrap.b32 r27, r24, r27, 18;\n\t"
		// A = {r2, r3}    B = {r30, r27}    C = {r22, r23}    D = {r12, r9}
		"lop3.b32 r24, r2, r30, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r27, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r30, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r27, r23, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r27, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r30, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r27, r23, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r27, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r30, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r27, r23, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r27, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r30, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r27, r23, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r30, r27}    C = {r22, r23}    D = {r12, r9}
		/*
		* |------------------------[ROUND 0.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r31, r25}           |
		* |            v[ 6]            |           {r30, r27}           |
		* |            v[ 7]            |           {r29, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12,  r9}           |
		* |            v[13]            |           {r11, r26}           |
		* |            v[14]            |           {r13, r28}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r26}
		"add.cc.u32 r4, r4, r29;\n\t"
		"addc.u32 r5, r5, r14;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r26}
		"xor.b32 r24, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r4, r24, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r26}
		"xor.b32 r11, r11, r4;\n\t"
		"xor.b32 r26, r26, r5;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r26}
		"shf.r.wrap.b32 r24, r11, r26, 60;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 60;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r24}
		"add.cc.u32 r16, r16, r11;\n\t"
		"addc.u32 r17, r17, r24;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r16, r17}    D = {r11, r24}
		"xor.b32 r29, r29, r16;\n\t"
		"xor.b32 r14, r14, r17;\n\t"
		"shf.r.wrap.b32 r26, r29, r14, 43;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 43;\n\t"
		// A = {r4, r5}    B = {r29, r26}    C = {r16, r17}    D = {r11, r24}
		"add.cc.u32 r4, r4, r29;\n\t"
		"addc.u32 r5, r5, r26;\n\t"
		// A = {r4, r5}    B = {r29, r26}    C = {r16, r17}    D = {r11, r24}
		"xor.b32 r14, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r29, r26}    C = {r16, r17}    D = {r11, r24}
		"xor.b32 r11, r11, r4;\n\t"
		"xor.b32 r24, r24, r5;\n\t"
		"shf.r.wrap.b32 r14, r11, r24, 5;\n\t"
		"shf.r.wrap.b32 r11, r24, r11, 5;\n\t"
		// A = {r4, r5}    B = {r29, r26}    C = {r16, r17}    D = {r14, r11}
		"add.cc.u32 r16, r16, r14;\n\t"
		"addc.u32 r17, r17, r11;\n\t"
		// A = {r4, r5}    B = {r29, r26}    C = {r16, r17}    D = {r14, r11}
		"xor.b32 r29, r29, r16;\n\t"
		"xor.b32 r26, r26, r17;\n\t"
		"shf.r.wrap.b32 r24, r29, r26, 18;\n\t"
		"shf.r.wrap.b32 r29, r26, r29, 18;\n\t"
		// A = {r4, r5}    B = {r24, r29}    C = {r16, r17}    D = {r14, r11}
		"lop3.b32 r26, r4, r24, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r29, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r24, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r29, r17, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r29, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r24, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r29, r17, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r29, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r24, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r29, r17, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r29, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r24, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r29, r17, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r24, r29}    C = {r16, r17}    D = {r14, r11}
		/*
		* |------------------------[ROUND 0.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48,  r8}           |
		* |            v[ 5]            |           {r31, r25}           |
		* |            v[ 6]            |           {r30, r27}           |
		* |            v[ 7]            |           {r24, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12,  r9}           |
		* |            v[13]            |           {r14, r11}           |
		* |            v[14]            |           {r13, r28}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r28}
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r8;\n\t"
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r28}
		"xor.b32 r26, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r26, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r28}
		"xor.b32 r13, r13, r6;\n\t"
		"xor.b32 r28, r28, r7;\n\t"
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r28}
		"shf.r.wrap.b32 r26, r13, r28, 60;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 60;\n\t"
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r26}
		"add.cc.u32 r18, r18, r13;\n\t"
		"addc.u32 r19, r19, r26;\n\t"
		// A = {r6, r7}    B = {r48, r8}    C = {r18, r19}    D = {r13, r26}
		"xor.b32 r48, r48, r18;\n\t"
		"xor.b32 r8, r8, r19;\n\t"
		"shf.r.wrap.b32 r28, r48, r8, 43;\n\t"
		"shf.r.wrap.b32 r48, r8, r48, 43;\n\t"
		// A = {r6, r7}    B = {r48, r28}    C = {r18, r19}    D = {r13, r26}
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r28;\n\t"
		// A = {r6, r7}    B = {r48, r28}    C = {r18, r19}    D = {r13, r26}
		"xor.b32 r8, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r48, r28}    C = {r18, r19}    D = {r13, r26}
		"xor.b32 r13, r13, r6;\n\t"
		"xor.b32 r26, r26, r7;\n\t"
		"shf.r.wrap.b32 r8, r13, r26, 5;\n\t"
		"shf.r.wrap.b32 r13, r26, r13, 5;\n\t"
		// A = {r6, r7}    B = {r48, r28}    C = {r18, r19}    D = {r8, r13}
		"add.cc.u32 r18, r18, r8;\n\t"
		"addc.u32 r19, r19, r13;\n\t"
		// A = {r6, r7}    B = {r48, r28}    C = {r18, r19}    D = {r8, r13}
		"xor.b32 r48, r48, r18;\n\t"
		"xor.b32 r28, r28, r19;\n\t"
		"shf.r.wrap.b32 r26, r48, r28, 18;\n\t"
		"shf.r.wrap.b32 r48, r28, r48, 18;\n\t"
		// A = {r6, r7}    B = {r26, r48}    C = {r18, r19}    D = {r8, r13}
		"lop3.b32 r28, r6, r26, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r48, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r26, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r48, r19, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r48, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r26, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r48, r19, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r48, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r26, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r48, r19, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r48, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r26, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r48, r19, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r26, r48}    C = {r18, r19}    D = {r8, r13}
		/*
		* |------------------------[ROUND 1.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r48}           |
		* |            v[ 5]            |           {r31, r25}           |
		* |            v[ 6]            |           {r30, r27}           |
		* |            v[ 7]            |           {r24, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12,  r9}           |
		* |            v[13]            |           {r14, r11}           |
		* |            v[14]            |           { r8, r13}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r9}
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r48;\n\t"
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r9}
		"xor.b32 r28, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r0, r28, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r9}
		"xor.b32 r12, r12, r0;\n\t"
		"xor.b32 r9, r9, r1;\n\t"
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r9}
		"shf.r.wrap.b32 r28, r12, r9, 60;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 60;\n\t"
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r28}
		"add.cc.u32 r16, r16, r12;\n\t"
		"addc.u32 r17, r17, r28;\n\t"
		// A = {r0, r1}    B = {r26, r48}    C = {r16, r17}    D = {r12, r28}
		"xor.b32 r26, r26, r16;\n\t"
		"xor.b32 r48, r48, r17;\n\t"
		"shf.r.wrap.b32 r9, r26, r48, 43;\n\t"
		"shf.r.wrap.b32 r26, r48, r26, 43;\n\t"
		// A = {r0, r1}    B = {r26, r9}    C = {r16, r17}    D = {r12, r28}
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r9;\n\t"
		// A = {r0, r1}    B = {r26, r9}    C = {r16, r17}    D = {r12, r28}
		"xor.b32 r48, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r26, r9}    C = {r16, r17}    D = {r12, r28}
		"xor.b32 r12, r12, r0;\n\t"
		"xor.b32 r28, r28, r1;\n\t"
		"shf.r.wrap.b32 r48, r12, r28, 5;\n\t"
		"shf.r.wrap.b32 r12, r28, r12, 5;\n\t"
		// A = {r0, r1}    B = {r26, r9}    C = {r16, r17}    D = {r48, r12}
		"add.cc.u32 r16, r16, r48;\n\t"
		"addc.u32 r17, r17, r12;\n\t"
		// A = {r0, r1}    B = {r26, r9}    C = {r16, r17}    D = {r48, r12}
		"xor.b32 r26, r26, r16;\n\t"
		"xor.b32 r9, r9, r17;\n\t"
		"shf.r.wrap.b32 r28, r26, r9, 18;\n\t"
		"shf.r.wrap.b32 r26, r9, r26, 18;\n\t"
		// A = {r0, r1}    B = {r28, r26}    C = {r16, r17}    D = {r48, r12}
		"lop3.b32 r9, r0, r28, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r26, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r28, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r26, r17, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r26, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r28, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r26, r17, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r26, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r28, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r26, r17, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r26, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r28, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r26, r17, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r28, r26}    C = {r16, r17}    D = {r48, r12}
		/*
		* |------------------------[ROUND 1.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           {r31, r25}           |
		* |            v[ 6]            |           {r30, r27}           |
		* |            v[ 7]            |           {r24, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r12}           |
		* |            v[13]            |           {r14, r11}           |
		* |            v[14]            |           { r8, r13}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r11}
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r25;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r11}
		"xor.b32 r9, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r2, r9, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r11}
		"xor.b32 r14, r14, r2;\n\t"
		"xor.b32 r11, r11, r3;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r11}
		"shf.r.wrap.b32 r9, r14, r11, 60;\n\t"
		"shf.r.wrap.b32 r14, r11, r14, 60;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r9}
		"add.cc.u32 r18, r18, r14;\n\t"
		"addc.u32 r19, r19, r9;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r18, r19}    D = {r14, r9}
		"xor.b32 r31, r31, r18;\n\t"
		"xor.b32 r25, r25, r19;\n\t"
		"shf.r.wrap.b32 r11, r31, r25, 43;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 43;\n\t"
		// A = {r2, r3}    B = {r31, r11}    C = {r18, r19}    D = {r14, r9}
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r11;\n\t"
		// A = {r2, r3}    B = {r31, r11}    C = {r18, r19}    D = {r14, r9}
		"xor.b32 r25, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r31, r11}    C = {r18, r19}    D = {r14, r9}
		"xor.b32 r14, r14, r2;\n\t"
		"xor.b32 r9, r9, r3;\n\t"
		"shf.r.wrap.b32 r25, r14, r9, 5;\n\t"
		"shf.r.wrap.b32 r14, r9, r14, 5;\n\t"
		// A = {r2, r3}    B = {r31, r11}    C = {r18, r19}    D = {r25, r14}
		"add.cc.u32 r18, r18, r25;\n\t"
		"addc.u32 r19, r19, r14;\n\t"
		// A = {r2, r3}    B = {r31, r11}    C = {r18, r19}    D = {r25, r14}
		"xor.b32 r31, r31, r18;\n\t"
		"xor.b32 r11, r11, r19;\n\t"
		"shf.r.wrap.b32 r9, r31, r11, 18;\n\t"
		"shf.r.wrap.b32 r31, r11, r31, 18;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r18, r19}    D = {r25, r14}
		"lop3.b32 r11, r2, r9, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r31, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r9, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r31, r19, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r31, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r9, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r31, r19, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r31, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r9, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r31, r19, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r31, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r9, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r31, r19, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r18, r19}    D = {r25, r14}
		/*
		* |------------------------[ROUND 1.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           { r9, r31}           |
		* |            v[ 6]            |           {r30, r27}           |
		* |            v[ 7]            |           {r24, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r12}           |
		* |            v[13]            |           {r25, r14}           |
		* |            v[14]            |           { r8, r13}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r13}
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r27;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r13}
		"xor.b32 r11, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r4, r11, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r13}
		"xor.b32 r8, r8, r4;\n\t"
		"xor.b32 r13, r13, r5;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r13}
		"shf.r.wrap.b32 r11, r8, r13, 60;\n\t"
		"shf.r.wrap.b32 r8, r13, r8, 60;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r11}
		"add.cc.u32 r20, r20, r8;\n\t"
		"addc.u32 r21, r21, r11;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r20, r21}    D = {r8, r11}
		"xor.b32 r30, r30, r20;\n\t"
		"xor.b32 r27, r27, r21;\n\t"
		"shf.r.wrap.b32 r13, r30, r27, 43;\n\t"
		"shf.r.wrap.b32 r30, r27, r30, 43;\n\t"
		// A = {r4, r5}    B = {r30, r13}    C = {r20, r21}    D = {r8, r11}
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r13;\n\t"
		// A = {r4, r5}    B = {r30, r13}    C = {r20, r21}    D = {r8, r11}
		"xor.b32 r27, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r30, r13}    C = {r20, r21}    D = {r8, r11}
		"xor.b32 r8, r8, r4;\n\t"
		"xor.b32 r11, r11, r5;\n\t"
		"shf.r.wrap.b32 r27, r8, r11, 5;\n\t"
		"shf.r.wrap.b32 r8, r11, r8, 5;\n\t"
		// A = {r4, r5}    B = {r30, r13}    C = {r20, r21}    D = {r27, r8}
		"add.cc.u32 r20, r20, r27;\n\t"
		"addc.u32 r21, r21, r8;\n\t"
		// A = {r4, r5}    B = {r30, r13}    C = {r20, r21}    D = {r27, r8}
		"xor.b32 r30, r30, r20;\n\t"
		"xor.b32 r13, r13, r21;\n\t"
		"shf.r.wrap.b32 r11, r30, r13, 18;\n\t"
		"shf.r.wrap.b32 r30, r13, r30, 18;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r20, r21}    D = {r27, r8}
		"lop3.b32 r13, r4, r11, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r30, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r11, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r30, r21, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r30, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r11, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r30, r21, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r30, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r11, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r30, r21, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r30, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r11, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r30, r21, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r20, r21}    D = {r27, r8}
		/*
		* |------------------------[ROUND 1.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           { r9, r31}           |
		* |            v[ 6]            |           {r11, r30}           |
		* |            v[ 7]            |           {r24, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r12}           |
		* |            v[13]            |           {r25, r14}           |
		* |            v[14]            |           {r27,  r8}           |
		* |            v[15]            |           {r10, r15}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r15}
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r29;\n\t"
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r15}
		"xor.b32 r13, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r6, r13, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r15}
		"xor.b32 r10, r10, r6;\n\t"
		"xor.b32 r15, r15, r7;\n\t"
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r15}
		"shf.r.wrap.b32 r13, r10, r15, 60;\n\t"
		"shf.r.wrap.b32 r10, r15, r10, 60;\n\t"
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r13}
		"add.cc.u32 r22, r22, r10;\n\t"
		"addc.u32 r23, r23, r13;\n\t"
		// A = {r6, r7}    B = {r24, r29}    C = {r22, r23}    D = {r10, r13}
		"xor.b32 r24, r24, r22;\n\t"
		"xor.b32 r29, r29, r23;\n\t"
		"shf.r.wrap.b32 r15, r24, r29, 43;\n\t"
		"shf.r.wrap.b32 r24, r29, r24, 43;\n\t"
		// A = {r6, r7}    B = {r24, r15}    C = {r22, r23}    D = {r10, r13}
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r15;\n\t"
		// A = {r6, r7}    B = {r24, r15}    C = {r22, r23}    D = {r10, r13}
		"xor.b32 r29, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r24, r15}    C = {r22, r23}    D = {r10, r13}
		"xor.b32 r10, r10, r6;\n\t"
		"xor.b32 r13, r13, r7;\n\t"
		"shf.r.wrap.b32 r29, r10, r13, 5;\n\t"
		"shf.r.wrap.b32 r10, r13, r10, 5;\n\t"
		// A = {r6, r7}    B = {r24, r15}    C = {r22, r23}    D = {r29, r10}
		"add.cc.u32 r22, r22, r29;\n\t"
		"addc.u32 r23, r23, r10;\n\t"
		// A = {r6, r7}    B = {r24, r15}    C = {r22, r23}    D = {r29, r10}
		"xor.b32 r24, r24, r22;\n\t"
		"xor.b32 r15, r15, r23;\n\t"
		"shf.r.wrap.b32 r13, r24, r15, 18;\n\t"
		"shf.r.wrap.b32 r24, r15, r24, 18;\n\t"
		// A = {r6, r7}    B = {r13, r24}    C = {r22, r23}    D = {r29, r10}
		"lop3.b32 r15, r6, r13, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r24, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r13, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r24, r23, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r24, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r13, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r24, r23, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r24, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r13, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r24, r23, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r24, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r13, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r24, r23, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r13, r24}    C = {r22, r23}    D = {r29, r10}
		/*
		* |------------------------[ROUND 1.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           { r9, r31}           |
		* |            v[ 6]            |           {r11, r30}           |
		* |            v[ 7]            |           {r13, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r12}           |
		* |            v[13]            |           {r25, r14}           |
		* |            v[14]            |           {r27,  r8}           |
		* |            v[15]            |           {r29, r10}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r10}
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r31;\n\t"
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r10}
		"xor.b32 r15, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r15, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r10}
		"xor.b32 r29, r29, r0;\n\t"
		"xor.b32 r10, r10, r1;\n\t"
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r10}
		"shf.r.wrap.b32 r15, r29, r10, 60;\n\t"
		"shf.r.wrap.b32 r29, r10, r29, 60;\n\t"
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r15}
		"add.cc.u32 r20, r20, r29;\n\t"
		"addc.u32 r21, r21, r15;\n\t"
		// A = {r0, r1}    B = {r9, r31}    C = {r20, r21}    D = {r29, r15}
		"xor.b32 r9, r9, r20;\n\t"
		"xor.b32 r31, r31, r21;\n\t"
		"shf.r.wrap.b32 r10, r9, r31, 43;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 43;\n\t"
		// A = {r0, r1}    B = {r9, r10}    C = {r20, r21}    D = {r29, r15}
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r10;\n\t"
		// A = {r0, r1}    B = {r9, r10}    C = {r20, r21}    D = {r29, r15}
		"xor.b32 r31, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r9, r10}    C = {r20, r21}    D = {r29, r15}
		"xor.b32 r29, r29, r0;\n\t"
		"xor.b32 r15, r15, r1;\n\t"
		"shf.r.wrap.b32 r31, r29, r15, 5;\n\t"
		"shf.r.wrap.b32 r29, r15, r29, 5;\n\t"
		// A = {r0, r1}    B = {r9, r10}    C = {r20, r21}    D = {r31, r29}
		"add.cc.u32 r20, r20, r31;\n\t"
		"addc.u32 r21, r21, r29;\n\t"
		// A = {r0, r1}    B = {r9, r10}    C = {r20, r21}    D = {r31, r29}
		"xor.b32 r9, r9, r20;\n\t"
		"xor.b32 r10, r10, r21;\n\t"
		"shf.r.wrap.b32 r15, r9, r10, 18;\n\t"
		"shf.r.wrap.b32 r9, r10, r9, 18;\n\t"
		// A = {r0, r1}    B = {r15, r9}    C = {r20, r21}    D = {r31, r29}
		"lop3.b32 r10, r0, r15, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r9, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r15, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r9, r21, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r9, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r15, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r9, r21, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r9, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r15, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r9, r21, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r9, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r15, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r9, r21, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r15, r9}    C = {r20, r21}    D = {r31, r29}
		/*
		* |------------------------[ROUND 1.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           {r15,  r9}           |
		* |            v[ 6]            |           {r11, r30}           |
		* |            v[ 7]            |           {r13, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r12}           |
		* |            v[13]            |           {r25, r14}           |
		* |            v[14]            |           {r27,  r8}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r12}
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r30;\n\t"
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r12}
		"xor.b32 r10, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r2, r10, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r12}
		"xor.b32 r48, r48, r2;\n\t"
		"xor.b32 r12, r12, r3;\n\t"
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r12}
		"shf.r.wrap.b32 r10, r48, r12, 60;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 60;\n\t"
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r10}
		"add.cc.u32 r22, r22, r48;\n\t"
		"addc.u32 r23, r23, r10;\n\t"
		// A = {r2, r3}    B = {r11, r30}    C = {r22, r23}    D = {r48, r10}
		"xor.b32 r11, r11, r22;\n\t"
		"xor.b32 r30, r30, r23;\n\t"
		"shf.r.wrap.b32 r12, r11, r30, 43;\n\t"
		"shf.r.wrap.b32 r11, r30, r11, 43;\n\t"
		// A = {r2, r3}    B = {r11, r12}    C = {r22, r23}    D = {r48, r10}
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r12;\n\t"
		// A = {r2, r3}    B = {r11, r12}    C = {r22, r23}    D = {r48, r10}
		"xor.b32 r30, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r2, r2, r30;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r11, r12}    C = {r22, r23}    D = {r48, r10}
		"xor.b32 r48, r48, r2;\n\t"
		"xor.b32 r10, r10, r3;\n\t"
		"shf.r.wrap.b32 r30, r48, r10, 5;\n\t"
		"shf.r.wrap.b32 r48, r10, r48, 5;\n\t"
		// A = {r2, r3}    B = {r11, r12}    C = {r22, r23}    D = {r30, r48}
		"add.cc.u32 r22, r22, r30;\n\t"
		"addc.u32 r23, r23, r48;\n\t"
		// A = {r2, r3}    B = {r11, r12}    C = {r22, r23}    D = {r30, r48}
		"xor.b32 r11, r11, r22;\n\t"
		"xor.b32 r12, r12, r23;\n\t"
		"shf.r.wrap.b32 r10, r11, r12, 18;\n\t"
		"shf.r.wrap.b32 r11, r12, r11, 18;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r22, r23}    D = {r30, r48}
		"lop3.b32 r12, r2, r10, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r11, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r10, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r11, r23, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r11, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r10, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r11, r23, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r11, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r10, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r11, r23, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r11, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r10, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r11, r23, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r10, r11}    C = {r22, r23}    D = {r30, r48}
		/*
		* |------------------------[ROUND 1.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           {r15,  r9}           |
		* |            v[ 6]            |           {r10, r11}           |
		* |            v[ 7]            |           {r13, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r48}           |
		* |            v[13]            |           {r25, r14}           |
		* |            v[14]            |           {r27,  r8}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r14}
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r24;\n\t"
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r14}
		"xor.b32 r12, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r4, r12, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r14}
		"xor.b32 r25, r25, r4;\n\t"
		"xor.b32 r14, r14, r5;\n\t"
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r14}
		"shf.r.wrap.b32 r12, r25, r14, 60;\n\t"
		"shf.r.wrap.b32 r25, r14, r25, 60;\n\t"
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r12}
		"add.cc.u32 r16, r16, r25;\n\t"
		"addc.u32 r17, r17, r12;\n\t"
		// A = {r4, r5}    B = {r13, r24}    C = {r16, r17}    D = {r25, r12}
		"xor.b32 r13, r13, r16;\n\t"
		"xor.b32 r24, r24, r17;\n\t"
		"shf.r.wrap.b32 r14, r13, r24, 43;\n\t"
		"shf.r.wrap.b32 r13, r24, r13, 43;\n\t"
		// A = {r4, r5}    B = {r13, r14}    C = {r16, r17}    D = {r25, r12}
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r14;\n\t"
		// A = {r4, r5}    B = {r13, r14}    C = {r16, r17}    D = {r25, r12}
		"xor.b32 r24, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r4, r4, r24;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r13, r14}    C = {r16, r17}    D = {r25, r12}
		"xor.b32 r25, r25, r4;\n\t"
		"xor.b32 r12, r12, r5;\n\t"
		"shf.r.wrap.b32 r24, r25, r12, 5;\n\t"
		"shf.r.wrap.b32 r25, r12, r25, 5;\n\t"
		// A = {r4, r5}    B = {r13, r14}    C = {r16, r17}    D = {r24, r25}
		"add.cc.u32 r16, r16, r24;\n\t"
		"addc.u32 r17, r17, r25;\n\t"
		// A = {r4, r5}    B = {r13, r14}    C = {r16, r17}    D = {r24, r25}
		"xor.b32 r13, r13, r16;\n\t"
		"xor.b32 r14, r14, r17;\n\t"
		"shf.r.wrap.b32 r12, r13, r14, 18;\n\t"
		"shf.r.wrap.b32 r13, r14, r13, 18;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r16, r17}    D = {r24, r25}
		"lop3.b32 r14, r4, r12, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r13, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r12, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r13, r17, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r13, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r12, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r13, r17, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r13, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r12, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r13, r17, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r13, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r12, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r13, r17, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r12, r13}    C = {r16, r17}    D = {r24, r25}
		/*
		* |------------------------[ROUND 1.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r28, r26}           |
		* |            v[ 5]            |           {r15,  r9}           |
		* |            v[ 6]            |           {r10, r11}           |
		* |            v[ 7]            |           {r12, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r48}           |
		* |            v[13]            |           {r24, r25}           |
		* |            v[14]            |           {r27,  r8}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r8}
		"add.cc.u32 r6, r6, r28;\n\t"
		"addc.u32 r7, r7, r26;\n\t"
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r8}
		"xor.b32 r14, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r14, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r8}
		"xor.b32 r27, r27, r6;\n\t"
		"xor.b32 r8, r8, r7;\n\t"
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r8}
		"shf.r.wrap.b32 r14, r27, r8, 60;\n\t"
		"shf.r.wrap.b32 r27, r8, r27, 60;\n\t"
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r14}
		"add.cc.u32 r18, r18, r27;\n\t"
		"addc.u32 r19, r19, r14;\n\t"
		// A = {r6, r7}    B = {r28, r26}    C = {r18, r19}    D = {r27, r14}
		"xor.b32 r28, r28, r18;\n\t"
		"xor.b32 r26, r26, r19;\n\t"
		"shf.r.wrap.b32 r8, r28, r26, 43;\n\t"
		"shf.r.wrap.b32 r28, r26, r28, 43;\n\t"
		// A = {r6, r7}    B = {r28, r8}    C = {r18, r19}    D = {r27, r14}
		"add.cc.u32 r6, r6, r28;\n\t"
		"addc.u32 r7, r7, r8;\n\t"
		// A = {r6, r7}    B = {r28, r8}    C = {r18, r19}    D = {r27, r14}
		"xor.b32 r26, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r6, r6, r26;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r28, r8}    C = {r18, r19}    D = {r27, r14}
		"xor.b32 r27, r27, r6;\n\t"
		"xor.b32 r14, r14, r7;\n\t"
		"shf.r.wrap.b32 r26, r27, r14, 5;\n\t"
		"shf.r.wrap.b32 r27, r14, r27, 5;\n\t"
		// A = {r6, r7}    B = {r28, r8}    C = {r18, r19}    D = {r26, r27}
		"add.cc.u32 r18, r18, r26;\n\t"
		"addc.u32 r19, r19, r27;\n\t"
		// A = {r6, r7}    B = {r28, r8}    C = {r18, r19}    D = {r26, r27}
		"xor.b32 r28, r28, r18;\n\t"
		"xor.b32 r8, r8, r19;\n\t"
		"shf.r.wrap.b32 r14, r28, r8, 18;\n\t"
		"shf.r.wrap.b32 r28, r8, r28, 18;\n\t"
		// A = {r6, r7}    B = {r14, r28}    C = {r18, r19}    D = {r26, r27}
		"lop3.b32 r8, r6, r14, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r28, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r14, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r28, r19, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r28, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r14, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r28, r19, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r28, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r14, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r28, r19, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r28, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r14, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r28, r19, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r14, r28}    C = {r18, r19}    D = {r26, r27}
		/*
		* |------------------------[ROUND 2.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r28}           |
		* |            v[ 5]            |           {r15,  r9}           |
		* |            v[ 6]            |           {r10, r11}           |
		* |            v[ 7]            |           {r12, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r48}           |
		* |            v[13]            |           {r24, r25}           |
		* |            v[14]            |           {r26, r27}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r48}
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r28;\n\t"
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r48}
		"xor.b32 r8, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r0, r8, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r48}
		"xor.b32 r30, r30, r0;\n\t"
		"xor.b32 r48, r48, r1;\n\t"
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r48}
		"shf.r.wrap.b32 r8, r30, r48, 60;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 60;\n\t"
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r8}
		"add.cc.u32 r16, r16, r30;\n\t"
		"addc.u32 r17, r17, r8;\n\t"
		// A = {r0, r1}    B = {r14, r28}    C = {r16, r17}    D = {r30, r8}
		"xor.b32 r14, r14, r16;\n\t"
		"xor.b32 r28, r28, r17;\n\t"
		"shf.r.wrap.b32 r48, r14, r28, 43;\n\t"
		"shf.r.wrap.b32 r14, r28, r14, 43;\n\t"
		// A = {r0, r1}    B = {r14, r48}    C = {r16, r17}    D = {r30, r8}
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r48;\n\t"
		// A = {r0, r1}    B = {r14, r48}    C = {r16, r17}    D = {r30, r8}
		"xor.b32 r28, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r0, r0, r28;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r14, r48}    C = {r16, r17}    D = {r30, r8}
		"xor.b32 r30, r30, r0;\n\t"
		"xor.b32 r8, r8, r1;\n\t"
		"shf.r.wrap.b32 r28, r30, r8, 5;\n\t"
		"shf.r.wrap.b32 r30, r8, r30, 5;\n\t"
		// A = {r0, r1}    B = {r14, r48}    C = {r16, r17}    D = {r28, r30}
		"add.cc.u32 r16, r16, r28;\n\t"
		"addc.u32 r17, r17, r30;\n\t"
		// A = {r0, r1}    B = {r14, r48}    C = {r16, r17}    D = {r28, r30}
		"xor.b32 r14, r14, r16;\n\t"
		"xor.b32 r48, r48, r17;\n\t"
		"shf.r.wrap.b32 r8, r14, r48, 18;\n\t"
		"shf.r.wrap.b32 r14, r48, r14, 18;\n\t"
		// A = {r0, r1}    B = {r8, r14}    C = {r16, r17}    D = {r28, r30}
		"lop3.b32 r48, r0, r8, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r14, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r8, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r14, r17, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r14, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r8, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r14, r17, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r14, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r8, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r14, r17, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r14, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r8, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r14, r17, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r8, r14}    C = {r16, r17}    D = {r28, r30}
		/*
		* |------------------------[ROUND 2.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r15,  r9}           |
		* |            v[ 6]            |           {r10, r11}           |
		* |            v[ 7]            |           {r12, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r30}           |
		* |            v[13]            |           {r24, r25}           |
		* |            v[14]            |           {r26, r27}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r25}
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r9;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r25}
		"xor.b32 r48, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r2, r48, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r25}
		"xor.b32 r24, r24, r2;\n\t"
		"xor.b32 r25, r25, r3;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r25}
		"shf.r.wrap.b32 r48, r24, r25, 60;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 60;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r48}
		"add.cc.u32 r18, r18, r24;\n\t"
		"addc.u32 r19, r19, r48;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r18, r19}    D = {r24, r48}
		"xor.b32 r15, r15, r18;\n\t"
		"xor.b32 r9, r9, r19;\n\t"
		"shf.r.wrap.b32 r25, r15, r9, 43;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 43;\n\t"
		// A = {r2, r3}    B = {r15, r25}    C = {r18, r19}    D = {r24, r48}
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r25;\n\t"
		// A = {r2, r3}    B = {r15, r25}    C = {r18, r19}    D = {r24, r48}
		"xor.b32 r9, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r15, r25}    C = {r18, r19}    D = {r24, r48}
		"xor.b32 r24, r24, r2;\n\t"
		"xor.b32 r48, r48, r3;\n\t"
		"shf.r.wrap.b32 r9, r24, r48, 5;\n\t"
		"shf.r.wrap.b32 r24, r48, r24, 5;\n\t"
		// A = {r2, r3}    B = {r15, r25}    C = {r18, r19}    D = {r9, r24}
		"add.cc.u32 r18, r18, r9;\n\t"
		"addc.u32 r19, r19, r24;\n\t"
		// A = {r2, r3}    B = {r15, r25}    C = {r18, r19}    D = {r9, r24}
		"xor.b32 r15, r15, r18;\n\t"
		"xor.b32 r25, r25, r19;\n\t"
		"shf.r.wrap.b32 r48, r15, r25, 18;\n\t"
		"shf.r.wrap.b32 r15, r25, r15, 18;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r18, r19}    D = {r9, r24}
		"lop3.b32 r25, r2, r48, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r15, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r48, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r15, r19, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r15, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r48, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r15, r19, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r15, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r48, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r15, r19, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r15, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r48, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r15, r19, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r18, r19}    D = {r9, r24}
		/*
		* |------------------------[ROUND 2.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r48, r15}           |
		* |            v[ 6]            |           {r10, r11}           |
		* |            v[ 7]            |           {r12, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r30}           |
		* |            v[13]            |           { r9, r24}           |
		* |            v[14]            |           {r26, r27}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r27}
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r11;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r27}
		"xor.b32 r25, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r4, r25, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r27}
		"xor.b32 r26, r26, r4;\n\t"
		"xor.b32 r27, r27, r5;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r27}
		"shf.r.wrap.b32 r25, r26, r27, 60;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 60;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r25}
		"add.cc.u32 r20, r20, r26;\n\t"
		"addc.u32 r21, r21, r25;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r20, r21}    D = {r26, r25}
		"xor.b32 r10, r10, r20;\n\t"
		"xor.b32 r11, r11, r21;\n\t"
		"shf.r.wrap.b32 r27, r10, r11, 43;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 43;\n\t"
		// A = {r4, r5}    B = {r10, r27}    C = {r20, r21}    D = {r26, r25}
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r27;\n\t"
		// A = {r4, r5}    B = {r10, r27}    C = {r20, r21}    D = {r26, r25}
		"xor.b32 r11, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r10, r27}    C = {r20, r21}    D = {r26, r25}
		"xor.b32 r26, r26, r4;\n\t"
		"xor.b32 r25, r25, r5;\n\t"
		"shf.r.wrap.b32 r11, r26, r25, 5;\n\t"
		"shf.r.wrap.b32 r26, r25, r26, 5;\n\t"
		// A = {r4, r5}    B = {r10, r27}    C = {r20, r21}    D = {r11, r26}
		"add.cc.u32 r20, r20, r11;\n\t"
		"addc.u32 r21, r21, r26;\n\t"
		// A = {r4, r5}    B = {r10, r27}    C = {r20, r21}    D = {r11, r26}
		"xor.b32 r10, r10, r20;\n\t"
		"xor.b32 r27, r27, r21;\n\t"
		"shf.r.wrap.b32 r25, r10, r27, 18;\n\t"
		"shf.r.wrap.b32 r10, r27, r10, 18;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r20, r21}    D = {r11, r26}
		"lop3.b32 r27, r4, r25, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r10, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r25, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r10, r21, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r10, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r25, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r10, r21, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r10, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r25, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r10, r21, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r10, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r25, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r10, r21, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r20, r21}    D = {r11, r26}
		/*
		* |------------------------[ROUND 2.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r48, r15}           |
		* |            v[ 6]            |           {r25, r10}           |
		* |            v[ 7]            |           {r12, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r30}           |
		* |            v[13]            |           { r9, r24}           |
		* |            v[14]            |           {r11, r26}           |
		* |            v[15]            |           {r31, r29}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r29}
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r13;\n\t"
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r29}
		"xor.b32 r27, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r27, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r29}
		"xor.b32 r31, r31, r6;\n\t"
		"xor.b32 r29, r29, r7;\n\t"
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r29}
		"shf.r.wrap.b32 r27, r31, r29, 60;\n\t"
		"shf.r.wrap.b32 r31, r29, r31, 60;\n\t"
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r27}
		"add.cc.u32 r22, r22, r31;\n\t"
		"addc.u32 r23, r23, r27;\n\t"
		// A = {r6, r7}    B = {r12, r13}    C = {r22, r23}    D = {r31, r27}
		"xor.b32 r12, r12, r22;\n\t"
		"xor.b32 r13, r13, r23;\n\t"
		"shf.r.wrap.b32 r29, r12, r13, 43;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 43;\n\t"
		// A = {r6, r7}    B = {r12, r29}    C = {r22, r23}    D = {r31, r27}
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r29;\n\t"
		// A = {r6, r7}    B = {r12, r29}    C = {r22, r23}    D = {r31, r27}
		"xor.b32 r13, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r12, r29}    C = {r22, r23}    D = {r31, r27}
		"xor.b32 r31, r31, r6;\n\t"
		"xor.b32 r27, r27, r7;\n\t"
		"shf.r.wrap.b32 r13, r31, r27, 5;\n\t"
		"shf.r.wrap.b32 r31, r27, r31, 5;\n\t"
		// A = {r6, r7}    B = {r12, r29}    C = {r22, r23}    D = {r13, r31}
		"add.cc.u32 r22, r22, r13;\n\t"
		"addc.u32 r23, r23, r31;\n\t"
		// A = {r6, r7}    B = {r12, r29}    C = {r22, r23}    D = {r13, r31}
		"xor.b32 r12, r12, r22;\n\t"
		"xor.b32 r29, r29, r23;\n\t"
		"shf.r.wrap.b32 r27, r12, r29, 18;\n\t"
		"shf.r.wrap.b32 r12, r29, r12, 18;\n\t"
		// A = {r6, r7}    B = {r27, r12}    C = {r22, r23}    D = {r13, r31}
		"lop3.b32 r29, r6, r27, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r12, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r27, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r12, r23, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r12, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r27, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r12, r23, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r12, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r27, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r12, r23, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r12, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r27, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r12, r23, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r27, r12}    C = {r22, r23}    D = {r13, r31}
		/*
		* |------------------------[ROUND 2.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r48, r15}           |
		* |            v[ 6]            |           {r25, r10}           |
		* |            v[ 7]            |           {r27, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r30}           |
		* |            v[13]            |           { r9, r24}           |
		* |            v[14]            |           {r11, r26}           |
		* |            v[15]            |           {r13, r31}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r31}
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r15;\n\t"
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r31}
		"xor.b32 r29, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r29, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r31}
		"xor.b32 r13, r13, r0;\n\t"
		"xor.b32 r31, r31, r1;\n\t"
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r31}
		"shf.r.wrap.b32 r29, r13, r31, 60;\n\t"
		"shf.r.wrap.b32 r13, r31, r13, 60;\n\t"
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r29}
		"add.cc.u32 r20, r20, r13;\n\t"
		"addc.u32 r21, r21, r29;\n\t"
		// A = {r0, r1}    B = {r48, r15}    C = {r20, r21}    D = {r13, r29}
		"xor.b32 r48, r48, r20;\n\t"
		"xor.b32 r15, r15, r21;\n\t"
		"shf.r.wrap.b32 r31, r48, r15, 43;\n\t"
		"shf.r.wrap.b32 r48, r15, r48, 43;\n\t"
		// A = {r0, r1}    B = {r48, r31}    C = {r20, r21}    D = {r13, r29}
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r31;\n\t"
		// A = {r0, r1}    B = {r48, r31}    C = {r20, r21}    D = {r13, r29}
		"xor.b32 r15, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r48, r31}    C = {r20, r21}    D = {r13, r29}
		"xor.b32 r13, r13, r0;\n\t"
		"xor.b32 r29, r29, r1;\n\t"
		"shf.r.wrap.b32 r15, r13, r29, 5;\n\t"
		"shf.r.wrap.b32 r13, r29, r13, 5;\n\t"
		// A = {r0, r1}    B = {r48, r31}    C = {r20, r21}    D = {r15, r13}
		"add.cc.u32 r20, r20, r15;\n\t"
		"addc.u32 r21, r21, r13;\n\t"
		// A = {r0, r1}    B = {r48, r31}    C = {r20, r21}    D = {r15, r13}
		"xor.b32 r48, r48, r20;\n\t"
		"xor.b32 r31, r31, r21;\n\t"
		"shf.r.wrap.b32 r29, r48, r31, 18;\n\t"
		"shf.r.wrap.b32 r48, r31, r48, 18;\n\t"
		// A = {r0, r1}    B = {r29, r48}    C = {r20, r21}    D = {r15, r13}
		"lop3.b32 r31, r0, r29, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r48, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r29, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r48, r21, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r48, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r29, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r48, r21, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r48, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r29, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r48, r21, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r48, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r29, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r48, r21, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r29, r48}    C = {r20, r21}    D = {r15, r13}
		/*
		* |------------------------[ROUND 2.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r29, r48}           |
		* |            v[ 6]            |           {r25, r10}           |
		* |            v[ 7]            |           {r27, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r30}           |
		* |            v[13]            |           { r9, r24}           |
		* |            v[14]            |           {r11, r26}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r30}
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r10;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r30}
		"xor.b32 r31, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r2, r31, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r30}
		"xor.b32 r28, r28, r2;\n\t"
		"xor.b32 r30, r30, r3;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r30}
		"shf.r.wrap.b32 r31, r28, r30, 60;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 60;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r31}
		"add.cc.u32 r22, r22, r28;\n\t"
		"addc.u32 r23, r23, r31;\n\t"
		// A = {r2, r3}    B = {r25, r10}    C = {r22, r23}    D = {r28, r31}
		"xor.b32 r25, r25, r22;\n\t"
		"xor.b32 r10, r10, r23;\n\t"
		"shf.r.wrap.b32 r30, r25, r10, 43;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 43;\n\t"
		// A = {r2, r3}    B = {r25, r30}    C = {r22, r23}    D = {r28, r31}
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r30;\n\t"
		// A = {r2, r3}    B = {r25, r30}    C = {r22, r23}    D = {r28, r31}
		"xor.b32 r10, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r25, r30}    C = {r22, r23}    D = {r28, r31}
		"xor.b32 r28, r28, r2;\n\t"
		"xor.b32 r31, r31, r3;\n\t"
		"shf.r.wrap.b32 r10, r28, r31, 5;\n\t"
		"shf.r.wrap.b32 r28, r31, r28, 5;\n\t"
		// A = {r2, r3}    B = {r25, r30}    C = {r22, r23}    D = {r10, r28}
		"add.cc.u32 r22, r22, r10;\n\t"
		"addc.u32 r23, r23, r28;\n\t"
		// A = {r2, r3}    B = {r25, r30}    C = {r22, r23}    D = {r10, r28}
		"xor.b32 r25, r25, r22;\n\t"
		"xor.b32 r30, r30, r23;\n\t"
		"shf.r.wrap.b32 r31, r25, r30, 18;\n\t"
		"shf.r.wrap.b32 r25, r30, r25, 18;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r22, r23}    D = {r10, r28}
		"lop3.b32 r30, r2, r31, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r25, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r31, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r25, r23, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r25, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r31, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r25, r23, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r25, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r31, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r25, r23, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r25, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r31, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r25, r23, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r31, r25}    C = {r22, r23}    D = {r10, r28}
		/*
		* |------------------------[ROUND 2.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r29, r48}           |
		* |            v[ 6]            |           {r31, r25}           |
		* |            v[ 7]            |           {r27, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r28}           |
		* |            v[13]            |           { r9, r24}           |
		* |            v[14]            |           {r11, r26}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r24}
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r12;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r24}
		"xor.b32 r30, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r4, r30, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r24}
		"xor.b32 r9, r9, r4;\n\t"
		"xor.b32 r24, r24, r5;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r24}
		"shf.r.wrap.b32 r30, r9, r24, 60;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 60;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r30}
		"add.cc.u32 r16, r16, r9;\n\t"
		"addc.u32 r17, r17, r30;\n\t"
		// A = {r4, r5}    B = {r27, r12}    C = {r16, r17}    D = {r9, r30}
		"xor.b32 r27, r27, r16;\n\t"
		"xor.b32 r12, r12, r17;\n\t"
		"shf.r.wrap.b32 r24, r27, r12, 43;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 43;\n\t"
		// A = {r4, r5}    B = {r27, r24}    C = {r16, r17}    D = {r9, r30}
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r24;\n\t"
		// A = {r4, r5}    B = {r27, r24}    C = {r16, r17}    D = {r9, r30}
		"xor.b32 r12, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r27, r24}    C = {r16, r17}    D = {r9, r30}
		"xor.b32 r9, r9, r4;\n\t"
		"xor.b32 r30, r30, r5;\n\t"
		"shf.r.wrap.b32 r12, r9, r30, 5;\n\t"
		"shf.r.wrap.b32 r9, r30, r9, 5;\n\t"
		// A = {r4, r5}    B = {r27, r24}    C = {r16, r17}    D = {r12, r9}
		"add.cc.u32 r16, r16, r12;\n\t"
		"addc.u32 r17, r17, r9;\n\t"
		// A = {r4, r5}    B = {r27, r24}    C = {r16, r17}    D = {r12, r9}
		"xor.b32 r27, r27, r16;\n\t"
		"xor.b32 r24, r24, r17;\n\t"
		"shf.r.wrap.b32 r30, r27, r24, 18;\n\t"
		"shf.r.wrap.b32 r27, r24, r27, 18;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r16, r17}    D = {r12, r9}
		"lop3.b32 r24, r4, r30, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r27, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r30, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r27, r17, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r27, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r30, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r27, r17, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r27, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r30, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r27, r17, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r27, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r30, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r27, r17, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r30, r27}    C = {r16, r17}    D = {r12, r9}
		/*
		* |------------------------[ROUND 2.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r8, r14}           |
		* |            v[ 5]            |           {r29, r48}           |
		* |            v[ 6]            |           {r31, r25}           |
		* |            v[ 7]            |           {r30, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r28}           |
		* |            v[13]            |           {r12,  r9}           |
		* |            v[14]            |           {r11, r26}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r26}
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r14;\n\t"
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r26}
		"xor.b32 r24, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r6, r24, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r26}
		"xor.b32 r11, r11, r6;\n\t"
		"xor.b32 r26, r26, r7;\n\t"
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r26}
		"shf.r.wrap.b32 r24, r11, r26, 60;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 60;\n\t"
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r24}
		"add.cc.u32 r18, r18, r11;\n\t"
		"addc.u32 r19, r19, r24;\n\t"
		// A = {r6, r7}    B = {r8, r14}    C = {r18, r19}    D = {r11, r24}
		"xor.b32 r8, r8, r18;\n\t"
		"xor.b32 r14, r14, r19;\n\t"
		"shf.r.wrap.b32 r26, r8, r14, 43;\n\t"
		"shf.r.wrap.b32 r8, r14, r8, 43;\n\t"
		// A = {r6, r7}    B = {r8, r26}    C = {r18, r19}    D = {r11, r24}
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r26;\n\t"
		// A = {r6, r7}    B = {r8, r26}    C = {r18, r19}    D = {r11, r24}
		"xor.b32 r14, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r8, r26}    C = {r18, r19}    D = {r11, r24}
		"xor.b32 r11, r11, r6;\n\t"
		"xor.b32 r24, r24, r7;\n\t"
		"shf.r.wrap.b32 r14, r11, r24, 5;\n\t"
		"shf.r.wrap.b32 r11, r24, r11, 5;\n\t"
		// A = {r6, r7}    B = {r8, r26}    C = {r18, r19}    D = {r14, r11}
		"add.cc.u32 r18, r18, r14;\n\t"
		"addc.u32 r19, r19, r11;\n\t"
		// A = {r6, r7}    B = {r8, r26}    C = {r18, r19}    D = {r14, r11}
		"xor.b32 r8, r8, r18;\n\t"
		"xor.b32 r26, r26, r19;\n\t"
		"shf.r.wrap.b32 r24, r8, r26, 18;\n\t"
		"shf.r.wrap.b32 r8, r26, r8, 18;\n\t"
		// A = {r6, r7}    B = {r24, r8}    C = {r18, r19}    D = {r14, r11}
		"lop3.b32 r26, r6, r24, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r8, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r24, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r8, r19, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r8, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r24, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r8, r19, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r8, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r24, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r8, r19, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r8, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r24, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r8, r19, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r24, r8}    C = {r18, r19}    D = {r14, r11}
		/*
		* |------------------------[ROUND 3.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24,  r8}           |
		* |            v[ 5]            |           {r29, r48}           |
		* |            v[ 6]            |           {r31, r25}           |
		* |            v[ 7]            |           {r30, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r28}           |
		* |            v[13]            |           {r12,  r9}           |
		* |            v[14]            |           {r14, r11}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r28}
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r8;\n\t"
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r28}
		"xor.b32 r26, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r26, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r28}
		"xor.b32 r10, r10, r0;\n\t"
		"xor.b32 r28, r28, r1;\n\t"
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r28}
		"shf.r.wrap.b32 r26, r10, r28, 60;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 60;\n\t"
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r26}
		"add.cc.u32 r16, r16, r10;\n\t"
		"addc.u32 r17, r17, r26;\n\t"
		// A = {r0, r1}    B = {r24, r8}    C = {r16, r17}    D = {r10, r26}
		"xor.b32 r24, r24, r16;\n\t"
		"xor.b32 r8, r8, r17;\n\t"
		"shf.r.wrap.b32 r28, r24, r8, 43;\n\t"
		"shf.r.wrap.b32 r24, r8, r24, 43;\n\t"
		// A = {r0, r1}    B = {r24, r28}    C = {r16, r17}    D = {r10, r26}
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r28;\n\t"
		// A = {r0, r1}    B = {r24, r28}    C = {r16, r17}    D = {r10, r26}
		"xor.b32 r8, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r24, r28}    C = {r16, r17}    D = {r10, r26}
		"xor.b32 r10, r10, r0;\n\t"
		"xor.b32 r26, r26, r1;\n\t"
		"shf.r.wrap.b32 r8, r10, r26, 5;\n\t"
		"shf.r.wrap.b32 r10, r26, r10, 5;\n\t"
		// A = {r0, r1}    B = {r24, r28}    C = {r16, r17}    D = {r8, r10}
		"add.cc.u32 r16, r16, r8;\n\t"
		"addc.u32 r17, r17, r10;\n\t"
		// A = {r0, r1}    B = {r24, r28}    C = {r16, r17}    D = {r8, r10}
		"xor.b32 r24, r24, r16;\n\t"
		"xor.b32 r28, r28, r17;\n\t"
		"shf.r.wrap.b32 r26, r24, r28, 18;\n\t"
		"shf.r.wrap.b32 r24, r28, r24, 18;\n\t"
		// A = {r0, r1}    B = {r26, r24}    C = {r16, r17}    D = {r8, r10}
		"lop3.b32 r28, r0, r26, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r24, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r26, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r24, r17, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r24, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r26, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r24, r17, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r24, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r26, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r24, r17, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r24, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r26, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r24, r17, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r26, r24}    C = {r16, r17}    D = {r8, r10}
		/*
		* |------------------------[ROUND 3.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r29, r48}           |
		* |            v[ 6]            |           {r31, r25}           |
		* |            v[ 7]            |           {r30, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r10}           |
		* |            v[13]            |           {r12,  r9}           |
		* |            v[14]            |           {r14, r11}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r9}
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r48;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r9}
		"xor.b32 r28, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r2, r28, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r9}
		"xor.b32 r12, r12, r2;\n\t"
		"xor.b32 r9, r9, r3;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r9}
		"shf.r.wrap.b32 r28, r12, r9, 60;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 60;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r28}
		"add.cc.u32 r18, r18, r12;\n\t"
		"addc.u32 r19, r19, r28;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r18, r19}    D = {r12, r28}
		"xor.b32 r29, r29, r18;\n\t"
		"xor.b32 r48, r48, r19;\n\t"
		"shf.r.wrap.b32 r9, r29, r48, 43;\n\t"
		"shf.r.wrap.b32 r29, r48, r29, 43;\n\t"
		// A = {r2, r3}    B = {r29, r9}    C = {r18, r19}    D = {r12, r28}
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r9;\n\t"
		// A = {r2, r3}    B = {r29, r9}    C = {r18, r19}    D = {r12, r28}
		"xor.b32 r48, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r29, r9}    C = {r18, r19}    D = {r12, r28}
		"xor.b32 r12, r12, r2;\n\t"
		"xor.b32 r28, r28, r3;\n\t"
		"shf.r.wrap.b32 r48, r12, r28, 5;\n\t"
		"shf.r.wrap.b32 r12, r28, r12, 5;\n\t"
		// A = {r2, r3}    B = {r29, r9}    C = {r18, r19}    D = {r48, r12}
		"add.cc.u32 r18, r18, r48;\n\t"
		"addc.u32 r19, r19, r12;\n\t"
		// A = {r2, r3}    B = {r29, r9}    C = {r18, r19}    D = {r48, r12}
		"xor.b32 r29, r29, r18;\n\t"
		"xor.b32 r9, r9, r19;\n\t"
		"shf.r.wrap.b32 r28, r29, r9, 18;\n\t"
		"shf.r.wrap.b32 r29, r9, r29, 18;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r18, r19}    D = {r48, r12}
		"lop3.b32 r9, r2, r28, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r29, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r28, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r29, r19, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r29, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r28, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r29, r19, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r29, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r28, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r29, r19, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r29, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r28, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r29, r19, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r18, r19}    D = {r48, r12}
		/*
		* |------------------------[ROUND 3.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r28, r29}           |
		* |            v[ 6]            |           {r31, r25}           |
		* |            v[ 7]            |           {r30, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r10}           |
		* |            v[13]            |           {r48, r12}           |
		* |            v[14]            |           {r14, r11}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r11}
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r25;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r11}
		"xor.b32 r9, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r9, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r11}
		"xor.b32 r14, r14, r4;\n\t"
		"xor.b32 r11, r11, r5;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r11}
		"shf.r.wrap.b32 r9, r14, r11, 60;\n\t"
		"shf.r.wrap.b32 r14, r11, r14, 60;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r9}
		"add.cc.u32 r20, r20, r14;\n\t"
		"addc.u32 r21, r21, r9;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r20, r21}    D = {r14, r9}
		"xor.b32 r31, r31, r20;\n\t"
		"xor.b32 r25, r25, r21;\n\t"
		"shf.r.wrap.b32 r11, r31, r25, 43;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 43;\n\t"
		// A = {r4, r5}    B = {r31, r11}    C = {r20, r21}    D = {r14, r9}
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r11;\n\t"
		// A = {r4, r5}    B = {r31, r11}    C = {r20, r21}    D = {r14, r9}
		"xor.b32 r25, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r31, r11}    C = {r20, r21}    D = {r14, r9}
		"xor.b32 r14, r14, r4;\n\t"
		"xor.b32 r9, r9, r5;\n\t"
		"shf.r.wrap.b32 r25, r14, r9, 5;\n\t"
		"shf.r.wrap.b32 r14, r9, r14, 5;\n\t"
		// A = {r4, r5}    B = {r31, r11}    C = {r20, r21}    D = {r25, r14}
		"add.cc.u32 r20, r20, r25;\n\t"
		"addc.u32 r21, r21, r14;\n\t"
		// A = {r4, r5}    B = {r31, r11}    C = {r20, r21}    D = {r25, r14}
		"xor.b32 r31, r31, r20;\n\t"
		"xor.b32 r11, r11, r21;\n\t"
		"shf.r.wrap.b32 r9, r31, r11, 18;\n\t"
		"shf.r.wrap.b32 r31, r11, r31, 18;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r20, r21}    D = {r25, r14}
		"lop3.b32 r11, r4, r9, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r31, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r9, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r31, r21, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r31, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r9, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r31, r21, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r31, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r9, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r31, r21, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r31, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r9, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r31, r21, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r20, r21}    D = {r25, r14}
		/*
		* |------------------------[ROUND 3.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r28, r29}           |
		* |            v[ 6]            |           { r9, r31}           |
		* |            v[ 7]            |           {r30, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r10}           |
		* |            v[13]            |           {r48, r12}           |
		* |            v[14]            |           {r25, r14}           |
		* |            v[15]            |           {r15, r13}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r13}
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r27;\n\t"
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r13}
		"xor.b32 r11, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r6, r11, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r13}
		"xor.b32 r15, r15, r6;\n\t"
		"xor.b32 r13, r13, r7;\n\t"
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r13}
		"shf.r.wrap.b32 r11, r15, r13, 60;\n\t"
		"shf.r.wrap.b32 r15, r13, r15, 60;\n\t"
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r11}
		"add.cc.u32 r22, r22, r15;\n\t"
		"addc.u32 r23, r23, r11;\n\t"
		// A = {r6, r7}    B = {r30, r27}    C = {r22, r23}    D = {r15, r11}
		"xor.b32 r30, r30, r22;\n\t"
		"xor.b32 r27, r27, r23;\n\t"
		"shf.r.wrap.b32 r13, r30, r27, 43;\n\t"
		"shf.r.wrap.b32 r30, r27, r30, 43;\n\t"
		// A = {r6, r7}    B = {r30, r13}    C = {r22, r23}    D = {r15, r11}
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r13;\n\t"
		// A = {r6, r7}    B = {r30, r13}    C = {r22, r23}    D = {r15, r11}
		"xor.b32 r27, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r30, r13}    C = {r22, r23}    D = {r15, r11}
		"xor.b32 r15, r15, r6;\n\t"
		"xor.b32 r11, r11, r7;\n\t"
		"shf.r.wrap.b32 r27, r15, r11, 5;\n\t"
		"shf.r.wrap.b32 r15, r11, r15, 5;\n\t"
		// A = {r6, r7}    B = {r30, r13}    C = {r22, r23}    D = {r27, r15}
		"add.cc.u32 r22, r22, r27;\n\t"
		"addc.u32 r23, r23, r15;\n\t"
		// A = {r6, r7}    B = {r30, r13}    C = {r22, r23}    D = {r27, r15}
		"xor.b32 r30, r30, r22;\n\t"
		"xor.b32 r13, r13, r23;\n\t"
		"shf.r.wrap.b32 r11, r30, r13, 18;\n\t"
		"shf.r.wrap.b32 r30, r13, r30, 18;\n\t"
		// A = {r6, r7}    B = {r11, r30}    C = {r22, r23}    D = {r27, r15}
		"lop3.b32 r13, r6, r11, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r30, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r11, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r30, r23, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r30, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r11, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r30, r23, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r30, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r11, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r30, r23, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r30, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r11, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r30, r23, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r11, r30}    C = {r22, r23}    D = {r27, r15}
		/*
		* |------------------------[ROUND 3.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r28, r29}           |
		* |            v[ 6]            |           { r9, r31}           |
		* |            v[ 7]            |           {r11, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r10}           |
		* |            v[13]            |           {r48, r12}           |
		* |            v[14]            |           {r25, r14}           |
		* |            v[15]            |           {r27, r15}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r15}
		"add.cc.u32 r0, r0, r28;\n\t"
		"addc.u32 r1, r1, r29;\n\t"
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r15}
		"xor.b32 r13, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r0, r13, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r15}
		"xor.b32 r27, r27, r0;\n\t"
		"xor.b32 r15, r15, r1;\n\t"
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r15}
		"shf.r.wrap.b32 r13, r27, r15, 60;\n\t"
		"shf.r.wrap.b32 r27, r15, r27, 60;\n\t"
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r13}
		"add.cc.u32 r20, r20, r27;\n\t"
		"addc.u32 r21, r21, r13;\n\t"
		// A = {r0, r1}    B = {r28, r29}    C = {r20, r21}    D = {r27, r13}
		"xor.b32 r28, r28, r20;\n\t"
		"xor.b32 r29, r29, r21;\n\t"
		"shf.r.wrap.b32 r15, r28, r29, 43;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 43;\n\t"
		// A = {r0, r1}    B = {r28, r15}    C = {r20, r21}    D = {r27, r13}
		"add.cc.u32 r0, r0, r28;\n\t"
		"addc.u32 r1, r1, r15;\n\t"
		// A = {r0, r1}    B = {r28, r15}    C = {r20, r21}    D = {r27, r13}
		"xor.b32 r29, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r28, r15}    C = {r20, r21}    D = {r27, r13}
		"xor.b32 r27, r27, r0;\n\t"
		"xor.b32 r13, r13, r1;\n\t"
		"shf.r.wrap.b32 r29, r27, r13, 5;\n\t"
		"shf.r.wrap.b32 r27, r13, r27, 5;\n\t"
		// A = {r0, r1}    B = {r28, r15}    C = {r20, r21}    D = {r29, r27}
		"add.cc.u32 r20, r20, r29;\n\t"
		"addc.u32 r21, r21, r27;\n\t"
		// A = {r0, r1}    B = {r28, r15}    C = {r20, r21}    D = {r29, r27}
		"xor.b32 r28, r28, r20;\n\t"
		"xor.b32 r15, r15, r21;\n\t"
		"shf.r.wrap.b32 r13, r28, r15, 18;\n\t"
		"shf.r.wrap.b32 r28, r15, r28, 18;\n\t"
		// A = {r0, r1}    B = {r13, r28}    C = {r20, r21}    D = {r29, r27}
		"lop3.b32 r15, r0, r13, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r28, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r13, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r28, r21, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r28, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r13, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r28, r21, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r28, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r13, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r28, r21, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r28, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r13, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r28, r21, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r13, r28}    C = {r20, r21}    D = {r29, r27}
		/*
		* |------------------------[ROUND 3.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r13, r28}           |
		* |            v[ 6]            |           { r9, r31}           |
		* |            v[ 7]            |           {r11, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r10}           |
		* |            v[13]            |           {r48, r12}           |
		* |            v[14]            |           {r25, r14}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r10}
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r31;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r10}
		"xor.b32 r15, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r15, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r10}
		"xor.b32 r8, r8, r2;\n\t"
		"xor.b32 r10, r10, r3;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r10}
		"shf.r.wrap.b32 r15, r8, r10, 60;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 60;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r15}
		"add.cc.u32 r22, r22, r8;\n\t"
		"addc.u32 r23, r23, r15;\n\t"
		// A = {r2, r3}    B = {r9, r31}    C = {r22, r23}    D = {r8, r15}
		"xor.b32 r9, r9, r22;\n\t"
		"xor.b32 r31, r31, r23;\n\t"
		"shf.r.wrap.b32 r10, r9, r31, 43;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 43;\n\t"
		// A = {r2, r3}    B = {r9, r10}    C = {r22, r23}    D = {r8, r15}
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r10;\n\t"
		// A = {r2, r3}    B = {r9, r10}    C = {r22, r23}    D = {r8, r15}
		"xor.b32 r31, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r9, r10}    C = {r22, r23}    D = {r8, r15}
		"xor.b32 r8, r8, r2;\n\t"
		"xor.b32 r15, r15, r3;\n\t"
		"shf.r.wrap.b32 r31, r8, r15, 5;\n\t"
		"shf.r.wrap.b32 r8, r15, r8, 5;\n\t"
		// A = {r2, r3}    B = {r9, r10}    C = {r22, r23}    D = {r31, r8}
		"add.cc.u32 r22, r22, r31;\n\t"
		"addc.u32 r23, r23, r8;\n\t"
		// A = {r2, r3}    B = {r9, r10}    C = {r22, r23}    D = {r31, r8}
		"xor.b32 r9, r9, r22;\n\t"
		"xor.b32 r10, r10, r23;\n\t"
		"shf.r.wrap.b32 r15, r9, r10, 18;\n\t"
		"shf.r.wrap.b32 r9, r10, r9, 18;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r22, r23}    D = {r31, r8}
		"lop3.b32 r10, r2, r15, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r9, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r15, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r9, r23, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r9, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r15, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r9, r23, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r9, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r15, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r9, r23, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r9, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r15, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r9, r23, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r15, r9}    C = {r22, r23}    D = {r31, r8}
		/*
		* |------------------------[ROUND 3.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r13, r28}           |
		* |            v[ 6]            |           {r15,  r9}           |
		* |            v[ 7]            |           {r11, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31,  r8}           |
		* |            v[13]            |           {r48, r12}           |
		* |            v[14]            |           {r25, r14}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r12}
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r30;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r12}
		"xor.b32 r10, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r4, r10, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r12}
		"xor.b32 r48, r48, r4;\n\t"
		"xor.b32 r12, r12, r5;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r12}
		"shf.r.wrap.b32 r10, r48, r12, 60;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 60;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r10}
		"add.cc.u32 r16, r16, r48;\n\t"
		"addc.u32 r17, r17, r10;\n\t"
		// A = {r4, r5}    B = {r11, r30}    C = {r16, r17}    D = {r48, r10}
		"xor.b32 r11, r11, r16;\n\t"
		"xor.b32 r30, r30, r17;\n\t"
		"shf.r.wrap.b32 r12, r11, r30, 43;\n\t"
		"shf.r.wrap.b32 r11, r30, r11, 43;\n\t"
		// A = {r4, r5}    B = {r11, r12}    C = {r16, r17}    D = {r48, r10}
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r12;\n\t"
		// A = {r4, r5}    B = {r11, r12}    C = {r16, r17}    D = {r48, r10}
		"xor.b32 r30, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r11, r12}    C = {r16, r17}    D = {r48, r10}
		"xor.b32 r48, r48, r4;\n\t"
		"xor.b32 r10, r10, r5;\n\t"
		"shf.r.wrap.b32 r30, r48, r10, 5;\n\t"
		"shf.r.wrap.b32 r48, r10, r48, 5;\n\t"
		// A = {r4, r5}    B = {r11, r12}    C = {r16, r17}    D = {r30, r48}
		"add.cc.u32 r16, r16, r30;\n\t"
		"addc.u32 r17, r17, r48;\n\t"
		// A = {r4, r5}    B = {r11, r12}    C = {r16, r17}    D = {r30, r48}
		"xor.b32 r11, r11, r16;\n\t"
		"xor.b32 r12, r12, r17;\n\t"
		"shf.r.wrap.b32 r10, r11, r12, 18;\n\t"
		"shf.r.wrap.b32 r11, r12, r11, 18;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r16, r17}    D = {r30, r48}
		"lop3.b32 r12, r4, r10, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r11, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r10, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r11, r17, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r11, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r10, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r11, r17, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r11, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r10, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r11, r17, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r11, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r10, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r11, r17, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r10, r11}    C = {r16, r17}    D = {r30, r48}
		/*
		* |------------------------[ROUND 3.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r26, r24}           |
		* |            v[ 5]            |           {r13, r28}           |
		* |            v[ 6]            |           {r15,  r9}           |
		* |            v[ 7]            |           {r10, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31,  r8}           |
		* |            v[13]            |           {r30, r48}           |
		* |            v[14]            |           {r25, r14}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r14}
		"add.cc.u32 r6, r6, r26;\n\t"
		"addc.u32 r7, r7, r24;\n\t"
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r14}
		"xor.b32 r12, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r12, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r14}
		"xor.b32 r25, r25, r6;\n\t"
		"xor.b32 r14, r14, r7;\n\t"
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r14}
		"shf.r.wrap.b32 r12, r25, r14, 60;\n\t"
		"shf.r.wrap.b32 r25, r14, r25, 60;\n\t"
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r12}
		"add.cc.u32 r18, r18, r25;\n\t"
		"addc.u32 r19, r19, r12;\n\t"
		// A = {r6, r7}    B = {r26, r24}    C = {r18, r19}    D = {r25, r12}
		"xor.b32 r26, r26, r18;\n\t"
		"xor.b32 r24, r24, r19;\n\t"
		"shf.r.wrap.b32 r14, r26, r24, 43;\n\t"
		"shf.r.wrap.b32 r26, r24, r26, 43;\n\t"
		// A = {r6, r7}    B = {r26, r14}    C = {r18, r19}    D = {r25, r12}
		"add.cc.u32 r6, r6, r26;\n\t"
		"addc.u32 r7, r7, r14;\n\t"
		// A = {r6, r7}    B = {r26, r14}    C = {r18, r19}    D = {r25, r12}
		"xor.b32 r24, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r26, r14}    C = {r18, r19}    D = {r25, r12}
		"xor.b32 r25, r25, r6;\n\t"
		"xor.b32 r12, r12, r7;\n\t"
		"shf.r.wrap.b32 r24, r25, r12, 5;\n\t"
		"shf.r.wrap.b32 r25, r12, r25, 5;\n\t"
		// A = {r6, r7}    B = {r26, r14}    C = {r18, r19}    D = {r24, r25}
		"add.cc.u32 r18, r18, r24;\n\t"
		"addc.u32 r19, r19, r25;\n\t"
		// A = {r6, r7}    B = {r26, r14}    C = {r18, r19}    D = {r24, r25}
		"xor.b32 r26, r26, r18;\n\t"
		"xor.b32 r14, r14, r19;\n\t"
		"shf.r.wrap.b32 r12, r26, r14, 18;\n\t"
		"shf.r.wrap.b32 r26, r14, r26, 18;\n\t"
		// A = {r6, r7}    B = {r12, r26}    C = {r18, r19}    D = {r24, r25}
		"lop3.b32 r14, r6, r12, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r26, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r12, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r26, r19, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r26, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r12, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r26, r19, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r26, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r12, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r26, r19, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r26, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r12, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r26, r19, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r12, r26}    C = {r18, r19}    D = {r24, r25}
		/*
		* |------------------------[ROUND 4.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r26}           |
		* |            v[ 5]            |           {r13, r28}           |
		* |            v[ 6]            |           {r15,  r9}           |
		* |            v[ 7]            |           {r10, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31,  r8}           |
		* |            v[13]            |           {r30, r48}           |
		* |            v[14]            |           {r24, r25}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r8}
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r26;\n\t"
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r8}
		"xor.b32 r14, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r14, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r8}
		"xor.b32 r31, r31, r0;\n\t"
		"xor.b32 r8, r8, r1;\n\t"
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r8}
		"shf.r.wrap.b32 r14, r31, r8, 60;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 60;\n\t"
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r14}
		"add.cc.u32 r16, r16, r31;\n\t"
		"addc.u32 r17, r17, r14;\n\t"
		// A = {r0, r1}    B = {r12, r26}    C = {r16, r17}    D = {r31, r14}
		"xor.b32 r12, r12, r16;\n\t"
		"xor.b32 r26, r26, r17;\n\t"
		"shf.r.wrap.b32 r8, r12, r26, 43;\n\t"
		"shf.r.wrap.b32 r12, r26, r12, 43;\n\t"
		// A = {r0, r1}    B = {r12, r8}    C = {r16, r17}    D = {r31, r14}
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r8;\n\t"
		// A = {r0, r1}    B = {r12, r8}    C = {r16, r17}    D = {r31, r14}
		"xor.b32 r26, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r12, r8}    C = {r16, r17}    D = {r31, r14}
		"xor.b32 r31, r31, r0;\n\t"
		"xor.b32 r14, r14, r1;\n\t"
		"shf.r.wrap.b32 r26, r31, r14, 5;\n\t"
		"shf.r.wrap.b32 r31, r14, r31, 5;\n\t"
		// A = {r0, r1}    B = {r12, r8}    C = {r16, r17}    D = {r26, r31}
		"add.cc.u32 r16, r16, r26;\n\t"
		"addc.u32 r17, r17, r31;\n\t"
		// A = {r0, r1}    B = {r12, r8}    C = {r16, r17}    D = {r26, r31}
		"xor.b32 r12, r12, r16;\n\t"
		"xor.b32 r8, r8, r17;\n\t"
		"shf.r.wrap.b32 r14, r12, r8, 18;\n\t"
		"shf.r.wrap.b32 r12, r8, r12, 18;\n\t"
		// A = {r0, r1}    B = {r14, r12}    C = {r16, r17}    D = {r26, r31}
		"lop3.b32 r8, r0, r14, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r12, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r14, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r12, r17, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r12, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r14, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r12, r17, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r12, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r14, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r12, r17, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r12, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r14, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r12, r17, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r14, r12}    C = {r16, r17}    D = {r26, r31}
		/*
		* |------------------------[ROUND 4.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           {r13, r28}           |
		* |            v[ 6]            |           {r15,  r9}           |
		* |            v[ 7]            |           {r10, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r31}           |
		* |            v[13]            |           {r30, r48}           |
		* |            v[14]            |           {r24, r25}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r48}
		"add.cc.u32 r2, r2, r13;\n\t"
		"addc.u32 r3, r3, r28;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r48}
		"xor.b32 r8, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r8, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r48}
		"xor.b32 r30, r30, r2;\n\t"
		"xor.b32 r48, r48, r3;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r48}
		"shf.r.wrap.b32 r8, r30, r48, 60;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 60;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r8}
		"add.cc.u32 r18, r18, r30;\n\t"
		"addc.u32 r19, r19, r8;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r18, r19}    D = {r30, r8}
		"xor.b32 r13, r13, r18;\n\t"
		"xor.b32 r28, r28, r19;\n\t"
		"shf.r.wrap.b32 r48, r13, r28, 43;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 43;\n\t"
		// A = {r2, r3}    B = {r13, r48}    C = {r18, r19}    D = {r30, r8}
		"add.cc.u32 r2, r2, r13;\n\t"
		"addc.u32 r3, r3, r48;\n\t"
		// A = {r2, r3}    B = {r13, r48}    C = {r18, r19}    D = {r30, r8}
		"xor.b32 r28, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r13, r48}    C = {r18, r19}    D = {r30, r8}
		"xor.b32 r30, r30, r2;\n\t"
		"xor.b32 r8, r8, r3;\n\t"
		"shf.r.wrap.b32 r28, r30, r8, 5;\n\t"
		"shf.r.wrap.b32 r30, r8, r30, 5;\n\t"
		// A = {r2, r3}    B = {r13, r48}    C = {r18, r19}    D = {r28, r30}
		"add.cc.u32 r18, r18, r28;\n\t"
		"addc.u32 r19, r19, r30;\n\t"
		// A = {r2, r3}    B = {r13, r48}    C = {r18, r19}    D = {r28, r30}
		"xor.b32 r13, r13, r18;\n\t"
		"xor.b32 r48, r48, r19;\n\t"
		"shf.r.wrap.b32 r8, r13, r48, 18;\n\t"
		"shf.r.wrap.b32 r13, r48, r13, 18;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r18, r19}    D = {r28, r30}
		"lop3.b32 r48, r2, r8, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r13, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r8, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r13, r19, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r13, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r8, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r13, r19, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r13, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r8, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r13, r19, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r13, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r8, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r13, r19, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r18, r19}    D = {r28, r30}
		/*
		* |------------------------[ROUND 4.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           { r8, r13}           |
		* |            v[ 6]            |           {r15,  r9}           |
		* |            v[ 7]            |           {r10, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r31}           |
		* |            v[13]            |           {r28, r30}           |
		* |            v[14]            |           {r24, r25}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r25}
		"add.cc.u32 r4, r4, r15;\n\t"
		"addc.u32 r5, r5, r9;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r25}
		"xor.b32 r48, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r48, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r25}
		"xor.b32 r24, r24, r4;\n\t"
		"xor.b32 r25, r25, r5;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r25}
		"shf.r.wrap.b32 r48, r24, r25, 60;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 60;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r48}
		"add.cc.u32 r20, r20, r24;\n\t"
		"addc.u32 r21, r21, r48;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r20, r21}    D = {r24, r48}
		"xor.b32 r15, r15, r20;\n\t"
		"xor.b32 r9, r9, r21;\n\t"
		"shf.r.wrap.b32 r25, r15, r9, 43;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 43;\n\t"
		// A = {r4, r5}    B = {r15, r25}    C = {r20, r21}    D = {r24, r48}
		"add.cc.u32 r4, r4, r15;\n\t"
		"addc.u32 r5, r5, r25;\n\t"
		// A = {r4, r5}    B = {r15, r25}    C = {r20, r21}    D = {r24, r48}
		"xor.b32 r9, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r15, r25}    C = {r20, r21}    D = {r24, r48}
		"xor.b32 r24, r24, r4;\n\t"
		"xor.b32 r48, r48, r5;\n\t"
		"shf.r.wrap.b32 r9, r24, r48, 5;\n\t"
		"shf.r.wrap.b32 r24, r48, r24, 5;\n\t"
		// A = {r4, r5}    B = {r15, r25}    C = {r20, r21}    D = {r9, r24}
		"add.cc.u32 r20, r20, r9;\n\t"
		"addc.u32 r21, r21, r24;\n\t"
		// A = {r4, r5}    B = {r15, r25}    C = {r20, r21}    D = {r9, r24}
		"xor.b32 r15, r15, r20;\n\t"
		"xor.b32 r25, r25, r21;\n\t"
		"shf.r.wrap.b32 r48, r15, r25, 18;\n\t"
		"shf.r.wrap.b32 r15, r25, r15, 18;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r20, r21}    D = {r9, r24}
		"lop3.b32 r25, r4, r48, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r15, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r48, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r15, r21, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r15, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r48, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r15, r21, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r15, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r48, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r15, r21, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r15, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r48, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r15, r21, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r20, r21}    D = {r9, r24}
		/*
		* |------------------------[ROUND 4.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           { r8, r13}           |
		* |            v[ 6]            |           {r48, r15}           |
		* |            v[ 7]            |           {r10, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r31}           |
		* |            v[13]            |           {r28, r30}           |
		* |            v[14]            |           { r9, r24}           |
		* |            v[15]            |           {r29, r27}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r27}
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r11;\n\t"
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r27}
		"xor.b32 r25, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r25, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r27}
		"xor.b32 r29, r29, r6;\n\t"
		"xor.b32 r27, r27, r7;\n\t"
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r27}
		"shf.r.wrap.b32 r25, r29, r27, 60;\n\t"
		"shf.r.wrap.b32 r29, r27, r29, 60;\n\t"
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r25}
		"add.cc.u32 r22, r22, r29;\n\t"
		"addc.u32 r23, r23, r25;\n\t"
		// A = {r6, r7}    B = {r10, r11}    C = {r22, r23}    D = {r29, r25}
		"xor.b32 r10, r10, r22;\n\t"
		"xor.b32 r11, r11, r23;\n\t"
		"shf.r.wrap.b32 r27, r10, r11, 43;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 43;\n\t"
		// A = {r6, r7}    B = {r10, r27}    C = {r22, r23}    D = {r29, r25}
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r27;\n\t"
		// A = {r6, r7}    B = {r10, r27}    C = {r22, r23}    D = {r29, r25}
		"xor.b32 r11, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r10, r27}    C = {r22, r23}    D = {r29, r25}
		"xor.b32 r29, r29, r6;\n\t"
		"xor.b32 r25, r25, r7;\n\t"
		"shf.r.wrap.b32 r11, r29, r25, 5;\n\t"
		"shf.r.wrap.b32 r29, r25, r29, 5;\n\t"
		// A = {r6, r7}    B = {r10, r27}    C = {r22, r23}    D = {r11, r29}
		"add.cc.u32 r22, r22, r11;\n\t"
		"addc.u32 r23, r23, r29;\n\t"
		// A = {r6, r7}    B = {r10, r27}    C = {r22, r23}    D = {r11, r29}
		"xor.b32 r10, r10, r22;\n\t"
		"xor.b32 r27, r27, r23;\n\t"
		"shf.r.wrap.b32 r25, r10, r27, 18;\n\t"
		"shf.r.wrap.b32 r10, r27, r10, 18;\n\t"
		// A = {r6, r7}    B = {r25, r10}    C = {r22, r23}    D = {r11, r29}
		"lop3.b32 r27, r6, r25, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r10, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r25, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r10, r23, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r10, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r25, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r10, r23, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r10, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r25, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r10, r23, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r10, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r25, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r10, r23, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r25, r10}    C = {r22, r23}    D = {r11, r29}
		/*
		* |------------------------[ROUND 4.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           { r8, r13}           |
		* |            v[ 6]            |           {r48, r15}           |
		* |            v[ 7]            |           {r25, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r31}           |
		* |            v[13]            |           {r28, r30}           |
		* |            v[14]            |           { r9, r24}           |
		* |            v[15]            |           {r11, r29}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r29}
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r13;\n\t"
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r29}
		"xor.b32 r27, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r27, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r29}
		"xor.b32 r11, r11, r0;\n\t"
		"xor.b32 r29, r29, r1;\n\t"
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r29}
		"shf.r.wrap.b32 r27, r11, r29, 60;\n\t"
		"shf.r.wrap.b32 r11, r29, r11, 60;\n\t"
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r27}
		"add.cc.u32 r20, r20, r11;\n\t"
		"addc.u32 r21, r21, r27;\n\t"
		// A = {r0, r1}    B = {r8, r13}    C = {r20, r21}    D = {r11, r27}
		"xor.b32 r8, r8, r20;\n\t"
		"xor.b32 r13, r13, r21;\n\t"
		"shf.r.wrap.b32 r29, r8, r13, 43;\n\t"
		"shf.r.wrap.b32 r8, r13, r8, 43;\n\t"
		// A = {r0, r1}    B = {r8, r29}    C = {r20, r21}    D = {r11, r27}
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r29;\n\t"
		// A = {r0, r1}    B = {r8, r29}    C = {r20, r21}    D = {r11, r27}
		"xor.b32 r13, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r8, r29}    C = {r20, r21}    D = {r11, r27}
		"xor.b32 r11, r11, r0;\n\t"
		"xor.b32 r27, r27, r1;\n\t"
		"shf.r.wrap.b32 r13, r11, r27, 5;\n\t"
		"shf.r.wrap.b32 r11, r27, r11, 5;\n\t"
		// A = {r0, r1}    B = {r8, r29}    C = {r20, r21}    D = {r13, r11}
		"add.cc.u32 r20, r20, r13;\n\t"
		"addc.u32 r21, r21, r11;\n\t"
		// A = {r0, r1}    B = {r8, r29}    C = {r20, r21}    D = {r13, r11}
		"xor.b32 r8, r8, r20;\n\t"
		"xor.b32 r29, r29, r21;\n\t"
		"shf.r.wrap.b32 r27, r8, r29, 18;\n\t"
		"shf.r.wrap.b32 r8, r29, r8, 18;\n\t"
		// A = {r0, r1}    B = {r27, r8}    C = {r20, r21}    D = {r13, r11}
		"lop3.b32 r29, r0, r27, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r8, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r27, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r8, r21, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r8, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r27, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r8, r21, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r8, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r27, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r8, r21, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r8, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r27, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r8, r21, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r27, r8}    C = {r20, r21}    D = {r13, r11}
		/*
		* |------------------------[ROUND 4.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           {r27,  r8}           |
		* |            v[ 6]            |           {r48, r15}           |
		* |            v[ 7]            |           {r25, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r31}           |
		* |            v[13]            |           {r28, r30}           |
		* |            v[14]            |           { r9, r24}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r31}
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r15;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r31}
		"xor.b32 r29, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r2, r29, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r31}
		"xor.b32 r26, r26, r2;\n\t"
		"xor.b32 r31, r31, r3;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r31}
		"shf.r.wrap.b32 r29, r26, r31, 60;\n\t"
		"shf.r.wrap.b32 r26, r31, r26, 60;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r29}
		"add.cc.u32 r22, r22, r26;\n\t"
		"addc.u32 r23, r23, r29;\n\t"
		// A = {r2, r3}    B = {r48, r15}    C = {r22, r23}    D = {r26, r29}
		"xor.b32 r48, r48, r22;\n\t"
		"xor.b32 r15, r15, r23;\n\t"
		"shf.r.wrap.b32 r31, r48, r15, 43;\n\t"
		"shf.r.wrap.b32 r48, r15, r48, 43;\n\t"
		// A = {r2, r3}    B = {r48, r31}    C = {r22, r23}    D = {r26, r29}
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r31;\n\t"
		// A = {r2, r3}    B = {r48, r31}    C = {r22, r23}    D = {r26, r29}
		"xor.b32 r15, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r48, r31}    C = {r22, r23}    D = {r26, r29}
		"xor.b32 r26, r26, r2;\n\t"
		"xor.b32 r29, r29, r3;\n\t"
		"shf.r.wrap.b32 r15, r26, r29, 5;\n\t"
		"shf.r.wrap.b32 r26, r29, r26, 5;\n\t"
		// A = {r2, r3}    B = {r48, r31}    C = {r22, r23}    D = {r15, r26}
		"add.cc.u32 r22, r22, r15;\n\t"
		"addc.u32 r23, r23, r26;\n\t"
		// A = {r2, r3}    B = {r48, r31}    C = {r22, r23}    D = {r15, r26}
		"xor.b32 r48, r48, r22;\n\t"
		"xor.b32 r31, r31, r23;\n\t"
		"shf.r.wrap.b32 r29, r48, r31, 18;\n\t"
		"shf.r.wrap.b32 r48, r31, r48, 18;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r22, r23}    D = {r15, r26}
		"lop3.b32 r31, r2, r29, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r48, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r29, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r48, r23, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r48, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r29, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r48, r23, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r48, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r29, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r48, r23, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r48, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r29, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r48, r23, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r29, r48}    C = {r22, r23}    D = {r15, r26}
		/*
		* |------------------------[ROUND 4.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           {r27,  r8}           |
		* |            v[ 6]            |           {r29, r48}           |
		* |            v[ 7]            |           {r25, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15, r26}           |
		* |            v[13]            |           {r28, r30}           |
		* |            v[14]            |           { r9, r24}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r30}
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r10;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r30}
		"xor.b32 r31, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r4, r31, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r30}
		"xor.b32 r28, r28, r4;\n\t"
		"xor.b32 r30, r30, r5;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r30}
		"shf.r.wrap.b32 r31, r28, r30, 60;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 60;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r31}
		"add.cc.u32 r16, r16, r28;\n\t"
		"addc.u32 r17, r17, r31;\n\t"
		// A = {r4, r5}    B = {r25, r10}    C = {r16, r17}    D = {r28, r31}
		"xor.b32 r25, r25, r16;\n\t"
		"xor.b32 r10, r10, r17;\n\t"
		"shf.r.wrap.b32 r30, r25, r10, 43;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 43;\n\t"
		// A = {r4, r5}    B = {r25, r30}    C = {r16, r17}    D = {r28, r31}
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r30;\n\t"
		// A = {r4, r5}    B = {r25, r30}    C = {r16, r17}    D = {r28, r31}
		"xor.b32 r10, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r25, r30}    C = {r16, r17}    D = {r28, r31}
		"xor.b32 r28, r28, r4;\n\t"
		"xor.b32 r31, r31, r5;\n\t"
		"shf.r.wrap.b32 r10, r28, r31, 5;\n\t"
		"shf.r.wrap.b32 r28, r31, r28, 5;\n\t"
		// A = {r4, r5}    B = {r25, r30}    C = {r16, r17}    D = {r10, r28}
		"add.cc.u32 r16, r16, r10;\n\t"
		"addc.u32 r17, r17, r28;\n\t"
		// A = {r4, r5}    B = {r25, r30}    C = {r16, r17}    D = {r10, r28}
		"xor.b32 r25, r25, r16;\n\t"
		"xor.b32 r30, r30, r17;\n\t"
		"shf.r.wrap.b32 r31, r25, r30, 18;\n\t"
		"shf.r.wrap.b32 r25, r30, r25, 18;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r16, r17}    D = {r10, r28}
		"lop3.b32 r30, r4, r31, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r25, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r31, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r25, r17, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r25, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r31, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r25, r17, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r25, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r31, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r25, r17, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r25, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r31, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r25, r17, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r31, r25}    C = {r16, r17}    D = {r10, r28}
		/*
		* |------------------------[ROUND 4.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r14, r12}           |
		* |            v[ 5]            |           {r27,  r8}           |
		* |            v[ 6]            |           {r29, r48}           |
		* |            v[ 7]            |           {r31, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15, r26}           |
		* |            v[13]            |           {r10, r28}           |
		* |            v[14]            |           { r9, r24}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r24}
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r12;\n\t"
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r24}
		"xor.b32 r30, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r30, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r24}
		"xor.b32 r9, r9, r6;\n\t"
		"xor.b32 r24, r24, r7;\n\t"
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r24}
		"shf.r.wrap.b32 r30, r9, r24, 60;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 60;\n\t"
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r30}
		"add.cc.u32 r18, r18, r9;\n\t"
		"addc.u32 r19, r19, r30;\n\t"
		// A = {r6, r7}    B = {r14, r12}    C = {r18, r19}    D = {r9, r30}
		"xor.b32 r14, r14, r18;\n\t"
		"xor.b32 r12, r12, r19;\n\t"
		"shf.r.wrap.b32 r24, r14, r12, 43;\n\t"
		"shf.r.wrap.b32 r14, r12, r14, 43;\n\t"
		// A = {r6, r7}    B = {r14, r24}    C = {r18, r19}    D = {r9, r30}
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r24;\n\t"
		// A = {r6, r7}    B = {r14, r24}    C = {r18, r19}    D = {r9, r30}
		"xor.b32 r12, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r14, r24}    C = {r18, r19}    D = {r9, r30}
		"xor.b32 r9, r9, r6;\n\t"
		"xor.b32 r30, r30, r7;\n\t"
		"shf.r.wrap.b32 r12, r9, r30, 5;\n\t"
		"shf.r.wrap.b32 r9, r30, r9, 5;\n\t"
		// A = {r6, r7}    B = {r14, r24}    C = {r18, r19}    D = {r12, r9}
		"add.cc.u32 r18, r18, r12;\n\t"
		"addc.u32 r19, r19, r9;\n\t"
		// A = {r6, r7}    B = {r14, r24}    C = {r18, r19}    D = {r12, r9}
		"xor.b32 r14, r14, r18;\n\t"
		"xor.b32 r24, r24, r19;\n\t"
		"shf.r.wrap.b32 r30, r14, r24, 18;\n\t"
		"shf.r.wrap.b32 r14, r24, r14, 18;\n\t"
		// A = {r6, r7}    B = {r30, r14}    C = {r18, r19}    D = {r12, r9}
		"lop3.b32 r24, r6, r30, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r14, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r30, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r14, r19, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r14, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r30, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r14, r19, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r14, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r30, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r14, r19, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r14, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r30, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r14, r19, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r30, r14}    C = {r18, r19}    D = {r12, r9}
		/*
		* |------------------------[ROUND 5.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r14}           |
		* |            v[ 5]            |           {r27,  r8}           |
		* |            v[ 6]            |           {r29, r48}           |
		* |            v[ 7]            |           {r31, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15, r26}           |
		* |            v[13]            |           {r10, r28}           |
		* |            v[14]            |           {r12,  r9}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r26}
		"add.cc.u32 r0, r0, r30;\n\t"
		"addc.u32 r1, r1, r14;\n\t"
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r26}
		"xor.b32 r24, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r24, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r26}
		"xor.b32 r15, r15, r0;\n\t"
		"xor.b32 r26, r26, r1;\n\t"
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r26}
		"shf.r.wrap.b32 r24, r15, r26, 60;\n\t"
		"shf.r.wrap.b32 r15, r26, r15, 60;\n\t"
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r24}
		"add.cc.u32 r16, r16, r15;\n\t"
		"addc.u32 r17, r17, r24;\n\t"
		// A = {r0, r1}    B = {r30, r14}    C = {r16, r17}    D = {r15, r24}
		"xor.b32 r30, r30, r16;\n\t"
		"xor.b32 r14, r14, r17;\n\t"
		"shf.r.wrap.b32 r26, r30, r14, 43;\n\t"
		"shf.r.wrap.b32 r30, r14, r30, 43;\n\t"
		// A = {r0, r1}    B = {r30, r26}    C = {r16, r17}    D = {r15, r24}
		"add.cc.u32 r0, r0, r30;\n\t"
		"addc.u32 r1, r1, r26;\n\t"
		// A = {r0, r1}    B = {r30, r26}    C = {r16, r17}    D = {r15, r24}
		"xor.b32 r14, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r30, r26}    C = {r16, r17}    D = {r15, r24}
		"xor.b32 r15, r15, r0;\n\t"
		"xor.b32 r24, r24, r1;\n\t"
		"shf.r.wrap.b32 r14, r15, r24, 5;\n\t"
		"shf.r.wrap.b32 r15, r24, r15, 5;\n\t"
		// A = {r0, r1}    B = {r30, r26}    C = {r16, r17}    D = {r14, r15}
		"add.cc.u32 r16, r16, r14;\n\t"
		"addc.u32 r17, r17, r15;\n\t"
		// A = {r0, r1}    B = {r30, r26}    C = {r16, r17}    D = {r14, r15}
		"xor.b32 r30, r30, r16;\n\t"
		"xor.b32 r26, r26, r17;\n\t"
		"shf.r.wrap.b32 r24, r30, r26, 18;\n\t"
		"shf.r.wrap.b32 r30, r26, r30, 18;\n\t"
		// A = {r0, r1}    B = {r24, r30}    C = {r16, r17}    D = {r14, r15}
		"lop3.b32 r26, r0, r24, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r30, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r24, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r30, r17, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r30, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r24, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r30, r17, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r30, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r24, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r30, r17, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r30, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r24, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r30, r17, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r24, r30}    C = {r16, r17}    D = {r14, r15}
		/*
		* |------------------------[ROUND 5.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r27,  r8}           |
		* |            v[ 6]            |           {r29, r48}           |
		* |            v[ 7]            |           {r31, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r15}           |
		* |            v[13]            |           {r10, r28}           |
		* |            v[14]            |           {r12,  r9}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r28}
		"add.cc.u32 r2, r2, r27;\n\t"
		"addc.u32 r3, r3, r8;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r28}
		"xor.b32 r26, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r26, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r28}
		"xor.b32 r10, r10, r2;\n\t"
		"xor.b32 r28, r28, r3;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r28}
		"shf.r.wrap.b32 r26, r10, r28, 60;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 60;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r26}
		"add.cc.u32 r18, r18, r10;\n\t"
		"addc.u32 r19, r19, r26;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r18, r19}    D = {r10, r26}
		"xor.b32 r27, r27, r18;\n\t"
		"xor.b32 r8, r8, r19;\n\t"
		"shf.r.wrap.b32 r28, r27, r8, 43;\n\t"
		"shf.r.wrap.b32 r27, r8, r27, 43;\n\t"
		// A = {r2, r3}    B = {r27, r28}    C = {r18, r19}    D = {r10, r26}
		"add.cc.u32 r2, r2, r27;\n\t"
		"addc.u32 r3, r3, r28;\n\t"
		// A = {r2, r3}    B = {r27, r28}    C = {r18, r19}    D = {r10, r26}
		"xor.b32 r8, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r27, r28}    C = {r18, r19}    D = {r10, r26}
		"xor.b32 r10, r10, r2;\n\t"
		"xor.b32 r26, r26, r3;\n\t"
		"shf.r.wrap.b32 r8, r10, r26, 5;\n\t"
		"shf.r.wrap.b32 r10, r26, r10, 5;\n\t"
		// A = {r2, r3}    B = {r27, r28}    C = {r18, r19}    D = {r8, r10}
		"add.cc.u32 r18, r18, r8;\n\t"
		"addc.u32 r19, r19, r10;\n\t"
		// A = {r2, r3}    B = {r27, r28}    C = {r18, r19}    D = {r8, r10}
		"xor.b32 r27, r27, r18;\n\t"
		"xor.b32 r28, r28, r19;\n\t"
		"shf.r.wrap.b32 r26, r27, r28, 18;\n\t"
		"shf.r.wrap.b32 r27, r28, r27, 18;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r18, r19}    D = {r8, r10}
		"lop3.b32 r28, r2, r26, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r27, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r26, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r27, r19, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r27, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r26, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r27, r19, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r27, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r26, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r27, r19, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r27, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r26, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r27, r19, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r18, r19}    D = {r8, r10}
		/*
		* |------------------------[ROUND 5.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r26, r27}           |
		* |            v[ 6]            |           {r29, r48}           |
		* |            v[ 7]            |           {r31, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r15}           |
		* |            v[13]            |           { r8, r10}           |
		* |            v[14]            |           {r12,  r9}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r9}
		"add.cc.u32 r4, r4, r29;\n\t"
		"addc.u32 r5, r5, r48;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r9}
		"xor.b32 r28, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r4, r28, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r9}
		"xor.b32 r12, r12, r4;\n\t"
		"xor.b32 r9, r9, r5;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r9}
		"shf.r.wrap.b32 r28, r12, r9, 60;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 60;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r28}
		"add.cc.u32 r20, r20, r12;\n\t"
		"addc.u32 r21, r21, r28;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r20, r21}    D = {r12, r28}
		"xor.b32 r29, r29, r20;\n\t"
		"xor.b32 r48, r48, r21;\n\t"
		"shf.r.wrap.b32 r9, r29, r48, 43;\n\t"
		"shf.r.wrap.b32 r29, r48, r29, 43;\n\t"
		// A = {r4, r5}    B = {r29, r9}    C = {r20, r21}    D = {r12, r28}
		"add.cc.u32 r4, r4, r29;\n\t"
		"addc.u32 r5, r5, r9;\n\t"
		// A = {r4, r5}    B = {r29, r9}    C = {r20, r21}    D = {r12, r28}
		"xor.b32 r48, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r29, r9}    C = {r20, r21}    D = {r12, r28}
		"xor.b32 r12, r12, r4;\n\t"
		"xor.b32 r28, r28, r5;\n\t"
		"shf.r.wrap.b32 r48, r12, r28, 5;\n\t"
		"shf.r.wrap.b32 r12, r28, r12, 5;\n\t"
		// A = {r4, r5}    B = {r29, r9}    C = {r20, r21}    D = {r48, r12}
		"add.cc.u32 r20, r20, r48;\n\t"
		"addc.u32 r21, r21, r12;\n\t"
		// A = {r4, r5}    B = {r29, r9}    C = {r20, r21}    D = {r48, r12}
		"xor.b32 r29, r29, r20;\n\t"
		"xor.b32 r9, r9, r21;\n\t"
		"shf.r.wrap.b32 r28, r29, r9, 18;\n\t"
		"shf.r.wrap.b32 r29, r9, r29, 18;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r20, r21}    D = {r48, r12}
		"lop3.b32 r9, r4, r28, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r29, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r28, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r29, r21, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r29, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r28, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r29, r21, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r29, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r28, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r29, r21, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r29, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r28, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r29, r21, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r20, r21}    D = {r48, r12}
		/*
		* |------------------------[ROUND 5.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r26, r27}           |
		* |            v[ 6]            |           {r28, r29}           |
		* |            v[ 7]            |           {r31, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r15}           |
		* |            v[13]            |           { r8, r10}           |
		* |            v[14]            |           {r48, r12}           |
		* |            v[15]            |           {r13, r11}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r11}
		"add.cc.u32 r6, r6, r31;\n\t"
		"addc.u32 r7, r7, r25;\n\t"
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r11}
		"xor.b32 r9, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r9, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r11}
		"xor.b32 r13, r13, r6;\n\t"
		"xor.b32 r11, r11, r7;\n\t"
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r11}
		"shf.r.wrap.b32 r9, r13, r11, 60;\n\t"
		"shf.r.wrap.b32 r13, r11, r13, 60;\n\t"
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r9}
		"add.cc.u32 r22, r22, r13;\n\t"
		"addc.u32 r23, r23, r9;\n\t"
		// A = {r6, r7}    B = {r31, r25}    C = {r22, r23}    D = {r13, r9}
		"xor.b32 r31, r31, r22;\n\t"
		"xor.b32 r25, r25, r23;\n\t"
		"shf.r.wrap.b32 r11, r31, r25, 43;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 43;\n\t"
		// A = {r6, r7}    B = {r31, r11}    C = {r22, r23}    D = {r13, r9}
		"add.cc.u32 r6, r6, r31;\n\t"
		"addc.u32 r7, r7, r11;\n\t"
		// A = {r6, r7}    B = {r31, r11}    C = {r22, r23}    D = {r13, r9}
		"xor.b32 r25, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r31, r11}    C = {r22, r23}    D = {r13, r9}
		"xor.b32 r13, r13, r6;\n\t"
		"xor.b32 r9, r9, r7;\n\t"
		"shf.r.wrap.b32 r25, r13, r9, 5;\n\t"
		"shf.r.wrap.b32 r13, r9, r13, 5;\n\t"
		// A = {r6, r7}    B = {r31, r11}    C = {r22, r23}    D = {r25, r13}
		"add.cc.u32 r22, r22, r25;\n\t"
		"addc.u32 r23, r23, r13;\n\t"
		// A = {r6, r7}    B = {r31, r11}    C = {r22, r23}    D = {r25, r13}
		"xor.b32 r31, r31, r22;\n\t"
		"xor.b32 r11, r11, r23;\n\t"
		"shf.r.wrap.b32 r9, r31, r11, 18;\n\t"
		"shf.r.wrap.b32 r31, r11, r31, 18;\n\t"
		// A = {r6, r7}    B = {r9, r31}    C = {r22, r23}    D = {r25, r13}
		"lop3.b32 r11, r6, r9, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r31, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r9, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r31, r23, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r31, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r9, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r31, r23, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r31, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r9, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r31, r23, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r31, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r9, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r31, r23, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r9, r31}    C = {r22, r23}    D = {r25, r13}
		/*
		* |------------------------[ROUND 5.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r26, r27}           |
		* |            v[ 6]            |           {r28, r29}           |
		* |            v[ 7]            |           { r9, r31}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r15}           |
		* |            v[13]            |           { r8, r10}           |
		* |            v[14]            |           {r48, r12}           |
		* |            v[15]            |           {r25, r13}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r13}
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r27;\n\t"
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r13}
		"xor.b32 r11, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r0, r11, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r13}
		"xor.b32 r25, r25, r0;\n\t"
		"xor.b32 r13, r13, r1;\n\t"
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r13}
		"shf.r.wrap.b32 r11, r25, r13, 60;\n\t"
		"shf.r.wrap.b32 r25, r13, r25, 60;\n\t"
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r11}
		"add.cc.u32 r20, r20, r25;\n\t"
		"addc.u32 r21, r21, r11;\n\t"
		// A = {r0, r1}    B = {r26, r27}    C = {r20, r21}    D = {r25, r11}
		"xor.b32 r26, r26, r20;\n\t"
		"xor.b32 r27, r27, r21;\n\t"
		"shf.r.wrap.b32 r13, r26, r27, 43;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 43;\n\t"
		// A = {r0, r1}    B = {r26, r13}    C = {r20, r21}    D = {r25, r11}
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r13;\n\t"
		// A = {r0, r1}    B = {r26, r13}    C = {r20, r21}    D = {r25, r11}
		"xor.b32 r27, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r26, r13}    C = {r20, r21}    D = {r25, r11}
		"xor.b32 r25, r25, r0;\n\t"
		"xor.b32 r11, r11, r1;\n\t"
		"shf.r.wrap.b32 r27, r25, r11, 5;\n\t"
		"shf.r.wrap.b32 r25, r11, r25, 5;\n\t"
		// A = {r0, r1}    B = {r26, r13}    C = {r20, r21}    D = {r27, r25}
		"add.cc.u32 r20, r20, r27;\n\t"
		"addc.u32 r21, r21, r25;\n\t"
		// A = {r0, r1}    B = {r26, r13}    C = {r20, r21}    D = {r27, r25}
		"xor.b32 r26, r26, r20;\n\t"
		"xor.b32 r13, r13, r21;\n\t"
		"shf.r.wrap.b32 r11, r26, r13, 18;\n\t"
		"shf.r.wrap.b32 r26, r13, r26, 18;\n\t"
		// A = {r0, r1}    B = {r11, r26}    C = {r20, r21}    D = {r27, r25}
		"lop3.b32 r13, r0, r11, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r26, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r11, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r26, r21, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r26, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r11, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r26, r21, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r26, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r11, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r26, r21, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r26, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r11, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r26, r21, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r11, r26}    C = {r20, r21}    D = {r27, r25}
		/*
		* |------------------------[ROUND 5.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r11, r26}           |
		* |            v[ 6]            |           {r28, r29}           |
		* |            v[ 7]            |           { r9, r31}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r15}           |
		* |            v[13]            |           { r8, r10}           |
		* |            v[14]            |           {r48, r12}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r15}
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r29;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r15}
		"xor.b32 r13, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r13, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r15}
		"xor.b32 r14, r14, r2;\n\t"
		"xor.b32 r15, r15, r3;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r15}
		"shf.r.wrap.b32 r13, r14, r15, 60;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 60;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r13}
		"add.cc.u32 r22, r22, r14;\n\t"
		"addc.u32 r23, r23, r13;\n\t"
		// A = {r2, r3}    B = {r28, r29}    C = {r22, r23}    D = {r14, r13}
		"xor.b32 r28, r28, r22;\n\t"
		"xor.b32 r29, r29, r23;\n\t"
		"shf.r.wrap.b32 r15, r28, r29, 43;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 43;\n\t"
		// A = {r2, r3}    B = {r28, r15}    C = {r22, r23}    D = {r14, r13}
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r15;\n\t"
		// A = {r2, r3}    B = {r28, r15}    C = {r22, r23}    D = {r14, r13}
		"xor.b32 r29, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r28, r15}    C = {r22, r23}    D = {r14, r13}
		"xor.b32 r14, r14, r2;\n\t"
		"xor.b32 r13, r13, r3;\n\t"
		"shf.r.wrap.b32 r29, r14, r13, 5;\n\t"
		"shf.r.wrap.b32 r14, r13, r14, 5;\n\t"
		// A = {r2, r3}    B = {r28, r15}    C = {r22, r23}    D = {r29, r14}
		"add.cc.u32 r22, r22, r29;\n\t"
		"addc.u32 r23, r23, r14;\n\t"
		// A = {r2, r3}    B = {r28, r15}    C = {r22, r23}    D = {r29, r14}
		"xor.b32 r28, r28, r22;\n\t"
		"xor.b32 r15, r15, r23;\n\t"
		"shf.r.wrap.b32 r13, r28, r15, 18;\n\t"
		"shf.r.wrap.b32 r28, r15, r28, 18;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r22, r23}    D = {r29, r14}
		"lop3.b32 r15, r2, r13, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r28, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r13, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r28, r23, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r28, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r13, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r28, r23, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r28, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r13, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r28, r23, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r28, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r13, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r28, r23, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r13, r28}    C = {r22, r23}    D = {r29, r14}
		/*
		* |------------------------[ROUND 5.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r11, r26}           |
		* |            v[ 6]            |           {r13, r28}           |
		* |            v[ 7]            |           { r9, r31}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r14}           |
		* |            v[13]            |           { r8, r10}           |
		* |            v[14]            |           {r48, r12}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r10}
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r31;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r10}
		"xor.b32 r15, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r4, r15, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r10}
		"xor.b32 r8, r8, r4;\n\t"
		"xor.b32 r10, r10, r5;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r10}
		"shf.r.wrap.b32 r15, r8, r10, 60;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 60;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r15}
		"add.cc.u32 r16, r16, r8;\n\t"
		"addc.u32 r17, r17, r15;\n\t"
		// A = {r4, r5}    B = {r9, r31}    C = {r16, r17}    D = {r8, r15}
		"xor.b32 r9, r9, r16;\n\t"
		"xor.b32 r31, r31, r17;\n\t"
		"shf.r.wrap.b32 r10, r9, r31, 43;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 43;\n\t"
		// A = {r4, r5}    B = {r9, r10}    C = {r16, r17}    D = {r8, r15}
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r10;\n\t"
		// A = {r4, r5}    B = {r9, r10}    C = {r16, r17}    D = {r8, r15}
		"xor.b32 r31, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r9, r10}    C = {r16, r17}    D = {r8, r15}
		"xor.b32 r8, r8, r4;\n\t"
		"xor.b32 r15, r15, r5;\n\t"
		"shf.r.wrap.b32 r31, r8, r15, 5;\n\t"
		"shf.r.wrap.b32 r8, r15, r8, 5;\n\t"
		// A = {r4, r5}    B = {r9, r10}    C = {r16, r17}    D = {r31, r8}
		"add.cc.u32 r16, r16, r31;\n\t"
		"addc.u32 r17, r17, r8;\n\t"
		// A = {r4, r5}    B = {r9, r10}    C = {r16, r17}    D = {r31, r8}
		"xor.b32 r9, r9, r16;\n\t"
		"xor.b32 r10, r10, r17;\n\t"
		"shf.r.wrap.b32 r15, r9, r10, 18;\n\t"
		"shf.r.wrap.b32 r9, r10, r9, 18;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r16, r17}    D = {r31, r8}
		"lop3.b32 r10, r4, r15, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r9, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r15, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r9, r17, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r9, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r15, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r9, r17, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r9, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r15, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r9, r17, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r9, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r15, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r9, r17, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r15, r9}    C = {r16, r17}    D = {r31, r8}
		/*
		* |------------------------[ROUND 5.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r24, r30}           |
		* |            v[ 5]            |           {r11, r26}           |
		* |            v[ 6]            |           {r13, r28}           |
		* |            v[ 7]            |           {r15,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r14}           |
		* |            v[13]            |           {r31,  r8}           |
		* |            v[14]            |           {r48, r12}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r12}
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r30;\n\t"
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r12}
		"xor.b32 r10, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r6, r10, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r12}
		"xor.b32 r48, r48, r6;\n\t"
		"xor.b32 r12, r12, r7;\n\t"
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r12}
		"shf.r.wrap.b32 r10, r48, r12, 60;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 60;\n\t"
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r10}
		"add.cc.u32 r18, r18, r48;\n\t"
		"addc.u32 r19, r19, r10;\n\t"
		// A = {r6, r7}    B = {r24, r30}    C = {r18, r19}    D = {r48, r10}
		"xor.b32 r24, r24, r18;\n\t"
		"xor.b32 r30, r30, r19;\n\t"
		"shf.r.wrap.b32 r12, r24, r30, 43;\n\t"
		"shf.r.wrap.b32 r24, r30, r24, 43;\n\t"
		// A = {r6, r7}    B = {r24, r12}    C = {r18, r19}    D = {r48, r10}
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r12;\n\t"
		// A = {r6, r7}    B = {r24, r12}    C = {r18, r19}    D = {r48, r10}
		"xor.b32 r30, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r24, r12}    C = {r18, r19}    D = {r48, r10}
		"xor.b32 r48, r48, r6;\n\t"
		"xor.b32 r10, r10, r7;\n\t"
		"shf.r.wrap.b32 r30, r48, r10, 5;\n\t"
		"shf.r.wrap.b32 r48, r10, r48, 5;\n\t"
		// A = {r6, r7}    B = {r24, r12}    C = {r18, r19}    D = {r30, r48}
		"add.cc.u32 r18, r18, r30;\n\t"
		"addc.u32 r19, r19, r48;\n\t"
		// A = {r6, r7}    B = {r24, r12}    C = {r18, r19}    D = {r30, r48}
		"xor.b32 r24, r24, r18;\n\t"
		"xor.b32 r12, r12, r19;\n\t"
		"shf.r.wrap.b32 r10, r24, r12, 18;\n\t"
		"shf.r.wrap.b32 r24, r12, r24, 18;\n\t"
		// A = {r6, r7}    B = {r10, r24}    C = {r18, r19}    D = {r30, r48}
		"lop3.b32 r12, r6, r10, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r24, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r10, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r24, r19, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r24, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r10, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r24, r19, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r24, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r10, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r24, r19, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r24, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r10, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r24, r19, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r10, r24}    C = {r18, r19}    D = {r30, r48}
		/*
		* |------------------------[ROUND 6.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r24}           |
		* |            v[ 5]            |           {r11, r26}           |
		* |            v[ 6]            |           {r13, r28}           |
		* |            v[ 7]            |           {r15,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r14}           |
		* |            v[13]            |           {r31,  r8}           |
		* |            v[14]            |           {r30, r48}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r14}
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r24;\n\t"
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r14}
		"xor.b32 r12, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r0, r12, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r14}
		"xor.b32 r29, r29, r0;\n\t"
		"xor.b32 r14, r14, r1;\n\t"
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r14}
		"shf.r.wrap.b32 r12, r29, r14, 60;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 60;\n\t"
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r12}
		"add.cc.u32 r16, r16, r29;\n\t"
		"addc.u32 r17, r17, r12;\n\t"
		// A = {r0, r1}    B = {r10, r24}    C = {r16, r17}    D = {r29, r12}
		"xor.b32 r10, r10, r16;\n\t"
		"xor.b32 r24, r24, r17;\n\t"
		"shf.r.wrap.b32 r14, r10, r24, 43;\n\t"
		"shf.r.wrap.b32 r10, r24, r10, 43;\n\t"
		// A = {r0, r1}    B = {r10, r14}    C = {r16, r17}    D = {r29, r12}
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r14;\n\t"
		// A = {r0, r1}    B = {r10, r14}    C = {r16, r17}    D = {r29, r12}
		"xor.b32 r24, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r10, r14}    C = {r16, r17}    D = {r29, r12}
		"xor.b32 r29, r29, r0;\n\t"
		"xor.b32 r12, r12, r1;\n\t"
		"shf.r.wrap.b32 r24, r29, r12, 5;\n\t"
		"shf.r.wrap.b32 r29, r12, r29, 5;\n\t"
		// A = {r0, r1}    B = {r10, r14}    C = {r16, r17}    D = {r24, r29}
		"add.cc.u32 r16, r16, r24;\n\t"
		"addc.u32 r17, r17, r29;\n\t"
		// A = {r0, r1}    B = {r10, r14}    C = {r16, r17}    D = {r24, r29}
		"xor.b32 r10, r10, r16;\n\t"
		"xor.b32 r14, r14, r17;\n\t"
		"shf.r.wrap.b32 r12, r10, r14, 18;\n\t"
		"shf.r.wrap.b32 r10, r14, r10, 18;\n\t"
		// A = {r0, r1}    B = {r12, r10}    C = {r16, r17}    D = {r24, r29}
		"lop3.b32 r14, r0, r12, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r10, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r12, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r10, r17, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r10, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r12, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r10, r17, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r10, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r12, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r10, r17, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r10, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r12, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r10, r17, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r12, r10}    C = {r16, r17}    D = {r24, r29}
		/*
		* |------------------------[ROUND 6.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r11, r26}           |
		* |            v[ 6]            |           {r13, r28}           |
		* |            v[ 7]            |           {r15,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r29}           |
		* |            v[13]            |           {r31,  r8}           |
		* |            v[14]            |           {r30, r48}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r8}
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r26;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r8}
		"xor.b32 r14, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r2, r14, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r8}
		"xor.b32 r31, r31, r2;\n\t"
		"xor.b32 r8, r8, r3;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r8}
		"shf.r.wrap.b32 r14, r31, r8, 60;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 60;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r14}
		"add.cc.u32 r18, r18, r31;\n\t"
		"addc.u32 r19, r19, r14;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r18, r19}    D = {r31, r14}
		"xor.b32 r11, r11, r18;\n\t"
		"xor.b32 r26, r26, r19;\n\t"
		"shf.r.wrap.b32 r8, r11, r26, 43;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 43;\n\t"
		// A = {r2, r3}    B = {r11, r8}    C = {r18, r19}    D = {r31, r14}
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r8;\n\t"
		// A = {r2, r3}    B = {r11, r8}    C = {r18, r19}    D = {r31, r14}
		"xor.b32 r26, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r11, r8}    C = {r18, r19}    D = {r31, r14}
		"xor.b32 r31, r31, r2;\n\t"
		"xor.b32 r14, r14, r3;\n\t"
		"shf.r.wrap.b32 r26, r31, r14, 5;\n\t"
		"shf.r.wrap.b32 r31, r14, r31, 5;\n\t"
		// A = {r2, r3}    B = {r11, r8}    C = {r18, r19}    D = {r26, r31}
		"add.cc.u32 r18, r18, r26;\n\t"
		"addc.u32 r19, r19, r31;\n\t"
		// A = {r2, r3}    B = {r11, r8}    C = {r18, r19}    D = {r26, r31}
		"xor.b32 r11, r11, r18;\n\t"
		"xor.b32 r8, r8, r19;\n\t"
		"shf.r.wrap.b32 r14, r11, r8, 18;\n\t"
		"shf.r.wrap.b32 r11, r8, r11, 18;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r18, r19}    D = {r26, r31}
		"lop3.b32 r8, r2, r14, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r11, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r14, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r11, r19, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r11, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r14, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r11, r19, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r11, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r14, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r11, r19, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r11, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r14, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r11, r19, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r18, r19}    D = {r26, r31}
		/*
		* |------------------------[ROUND 6.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r14, r11}           |
		* |            v[ 6]            |           {r13, r28}           |
		* |            v[ 7]            |           {r15,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r29}           |
		* |            v[13]            |           {r26, r31}           |
		* |            v[14]            |           {r30, r48}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r48}
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r28;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r48}
		"xor.b32 r8, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r4, r8, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r48}
		"xor.b32 r30, r30, r4;\n\t"
		"xor.b32 r48, r48, r5;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r48}
		"shf.r.wrap.b32 r8, r30, r48, 60;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 60;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r8}
		"add.cc.u32 r20, r20, r30;\n\t"
		"addc.u32 r21, r21, r8;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r20, r21}    D = {r30, r8}
		"xor.b32 r13, r13, r20;\n\t"
		"xor.b32 r28, r28, r21;\n\t"
		"shf.r.wrap.b32 r48, r13, r28, 43;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 43;\n\t"
		// A = {r4, r5}    B = {r13, r48}    C = {r20, r21}    D = {r30, r8}
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r48;\n\t"
		// A = {r4, r5}    B = {r13, r48}    C = {r20, r21}    D = {r30, r8}
		"xor.b32 r28, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r13, r48}    C = {r20, r21}    D = {r30, r8}
		"xor.b32 r30, r30, r4;\n\t"
		"xor.b32 r8, r8, r5;\n\t"
		"shf.r.wrap.b32 r28, r30, r8, 5;\n\t"
		"shf.r.wrap.b32 r30, r8, r30, 5;\n\t"
		// A = {r4, r5}    B = {r13, r48}    C = {r20, r21}    D = {r28, r30}
		"add.cc.u32 r20, r20, r28;\n\t"
		"addc.u32 r21, r21, r30;\n\t"
		// A = {r4, r5}    B = {r13, r48}    C = {r20, r21}    D = {r28, r30}
		"xor.b32 r13, r13, r20;\n\t"
		"xor.b32 r48, r48, r21;\n\t"
		"shf.r.wrap.b32 r8, r13, r48, 18;\n\t"
		"shf.r.wrap.b32 r13, r48, r13, 18;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r20, r21}    D = {r28, r30}
		"lop3.b32 r48, r4, r8, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r13, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r8, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r13, r21, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r13, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r8, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r13, r21, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r13, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r8, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r13, r21, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r13, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r8, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r13, r21, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r20, r21}    D = {r28, r30}
		/*
		* |------------------------[ROUND 6.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r14, r11}           |
		* |            v[ 6]            |           { r8, r13}           |
		* |            v[ 7]            |           {r15,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r29}           |
		* |            v[13]            |           {r26, r31}           |
		* |            v[14]            |           {r28, r30}           |
		* |            v[15]            |           {r27, r25}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r25}
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r9;\n\t"
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r25}
		"xor.b32 r48, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r6, r48, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r25}
		"xor.b32 r27, r27, r6;\n\t"
		"xor.b32 r25, r25, r7;\n\t"
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r25}
		"shf.r.wrap.b32 r48, r27, r25, 60;\n\t"
		"shf.r.wrap.b32 r27, r25, r27, 60;\n\t"
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r48}
		"add.cc.u32 r22, r22, r27;\n\t"
		"addc.u32 r23, r23, r48;\n\t"
		// A = {r6, r7}    B = {r15, r9}    C = {r22, r23}    D = {r27, r48}
		"xor.b32 r15, r15, r22;\n\t"
		"xor.b32 r9, r9, r23;\n\t"
		"shf.r.wrap.b32 r25, r15, r9, 43;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 43;\n\t"
		// A = {r6, r7}    B = {r15, r25}    C = {r22, r23}    D = {r27, r48}
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r25;\n\t"
		// A = {r6, r7}    B = {r15, r25}    C = {r22, r23}    D = {r27, r48}
		"xor.b32 r9, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r6, r6, r9;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r15, r25}    C = {r22, r23}    D = {r27, r48}
		"xor.b32 r27, r27, r6;\n\t"
		"xor.b32 r48, r48, r7;\n\t"
		"shf.r.wrap.b32 r9, r27, r48, 5;\n\t"
		"shf.r.wrap.b32 r27, r48, r27, 5;\n\t"
		// A = {r6, r7}    B = {r15, r25}    C = {r22, r23}    D = {r9, r27}
		"add.cc.u32 r22, r22, r9;\n\t"
		"addc.u32 r23, r23, r27;\n\t"
		// A = {r6, r7}    B = {r15, r25}    C = {r22, r23}    D = {r9, r27}
		"xor.b32 r15, r15, r22;\n\t"
		"xor.b32 r25, r25, r23;\n\t"
		"shf.r.wrap.b32 r48, r15, r25, 18;\n\t"
		"shf.r.wrap.b32 r15, r25, r15, 18;\n\t"
		// A = {r6, r7}    B = {r48, r15}    C = {r22, r23}    D = {r9, r27}
		"lop3.b32 r25, r6, r48, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r15, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r48, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r15, r23, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r15, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r48, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r15, r23, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r15, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r48, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r15, r23, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r15, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r48, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r15, r23, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r48, r15}    C = {r22, r23}    D = {r9, r27}
		/*
		* |------------------------[ROUND 6.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r14, r11}           |
		* |            v[ 6]            |           { r8, r13}           |
		* |            v[ 7]            |           {r48, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r29}           |
		* |            v[13]            |           {r26, r31}           |
		* |            v[14]            |           {r28, r30}           |
		* |            v[15]            |           { r9, r27}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r27}
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r11;\n\t"
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r27}
		"xor.b32 r25, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r0, r25, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r27}
		"xor.b32 r9, r9, r0;\n\t"
		"xor.b32 r27, r27, r1;\n\t"
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r27}
		"shf.r.wrap.b32 r25, r9, r27, 60;\n\t"
		"shf.r.wrap.b32 r9, r27, r9, 60;\n\t"
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r25}
		"add.cc.u32 r20, r20, r9;\n\t"
		"addc.u32 r21, r21, r25;\n\t"
		// A = {r0, r1}    B = {r14, r11}    C = {r20, r21}    D = {r9, r25}
		"xor.b32 r14, r14, r20;\n\t"
		"xor.b32 r11, r11, r21;\n\t"
		"shf.r.wrap.b32 r27, r14, r11, 43;\n\t"
		"shf.r.wrap.b32 r14, r11, r14, 43;\n\t"
		// A = {r0, r1}    B = {r14, r27}    C = {r20, r21}    D = {r9, r25}
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r27;\n\t"
		// A = {r0, r1}    B = {r14, r27}    C = {r20, r21}    D = {r9, r25}
		"xor.b32 r11, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r0, r11;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r14, r27}    C = {r20, r21}    D = {r9, r25}
		"xor.b32 r9, r9, r0;\n\t"
		"xor.b32 r25, r25, r1;\n\t"
		"shf.r.wrap.b32 r11, r9, r25, 5;\n\t"
		"shf.r.wrap.b32 r9, r25, r9, 5;\n\t"
		// A = {r0, r1}    B = {r14, r27}    C = {r20, r21}    D = {r11, r9}
		"add.cc.u32 r20, r20, r11;\n\t"
		"addc.u32 r21, r21, r9;\n\t"
		// A = {r0, r1}    B = {r14, r27}    C = {r20, r21}    D = {r11, r9}
		"xor.b32 r14, r14, r20;\n\t"
		"xor.b32 r27, r27, r21;\n\t"
		"shf.r.wrap.b32 r25, r14, r27, 18;\n\t"
		"shf.r.wrap.b32 r14, r27, r14, 18;\n\t"
		// A = {r0, r1}    B = {r25, r14}    C = {r20, r21}    D = {r11, r9}
		"lop3.b32 r27, r0, r25, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r14, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r25, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r14, r21, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r14, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r25, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r14, r21, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r14, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r25, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r14, r21, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r14, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r25, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r14, r21, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r25, r14}    C = {r20, r21}    D = {r11, r9}
		/*
		* |------------------------[ROUND 6.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r25, r14}           |
		* |            v[ 6]            |           { r8, r13}           |
		* |            v[ 7]            |           {r48, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r24, r29}           |
		* |            v[13]            |           {r26, r31}           |
		* |            v[14]            |           {r28, r30}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r29}
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r13;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r29}
		"xor.b32 r27, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r27, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r29}
		"xor.b32 r24, r24, r2;\n\t"
		"xor.b32 r29, r29, r3;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r29}
		"shf.r.wrap.b32 r27, r24, r29, 60;\n\t"
		"shf.r.wrap.b32 r24, r29, r24, 60;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r27}
		"add.cc.u32 r22, r22, r24;\n\t"
		"addc.u32 r23, r23, r27;\n\t"
		// A = {r2, r3}    B = {r8, r13}    C = {r22, r23}    D = {r24, r27}
		"xor.b32 r8, r8, r22;\n\t"
		"xor.b32 r13, r13, r23;\n\t"
		"shf.r.wrap.b32 r29, r8, r13, 43;\n\t"
		"shf.r.wrap.b32 r8, r13, r8, 43;\n\t"
		// A = {r2, r3}    B = {r8, r29}    C = {r22, r23}    D = {r24, r27}
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r29;\n\t"
		// A = {r2, r3}    B = {r8, r29}    C = {r22, r23}    D = {r24, r27}
		"xor.b32 r13, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r2, r2, r13;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r8, r29}    C = {r22, r23}    D = {r24, r27}
		"xor.b32 r24, r24, r2;\n\t"
		"xor.b32 r27, r27, r3;\n\t"
		"shf.r.wrap.b32 r13, r24, r27, 5;\n\t"
		"shf.r.wrap.b32 r24, r27, r24, 5;\n\t"
		// A = {r2, r3}    B = {r8, r29}    C = {r22, r23}    D = {r13, r24}
		"add.cc.u32 r22, r22, r13;\n\t"
		"addc.u32 r23, r23, r24;\n\t"
		// A = {r2, r3}    B = {r8, r29}    C = {r22, r23}    D = {r13, r24}
		"xor.b32 r8, r8, r22;\n\t"
		"xor.b32 r29, r29, r23;\n\t"
		"shf.r.wrap.b32 r27, r8, r29, 18;\n\t"
		"shf.r.wrap.b32 r8, r29, r8, 18;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r22, r23}    D = {r13, r24}
		"lop3.b32 r29, r2, r27, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r8, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r27, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r8, r23, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r8, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r27, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r8, r23, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r8, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r27, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r8, r23, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r8, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r27, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r8, r23, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r27, r8}    C = {r22, r23}    D = {r13, r24}
		/*
		* |------------------------[ROUND 6.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r25, r14}           |
		* |            v[ 6]            |           {r27,  r8}           |
		* |            v[ 7]            |           {r48, r15}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r24}           |
		* |            v[13]            |           {r26, r31}           |
		* |            v[14]            |           {r28, r30}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r31}
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r15;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r31}
		"xor.b32 r29, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r4, r29, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r31}
		"xor.b32 r26, r26, r4;\n\t"
		"xor.b32 r31, r31, r5;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r31}
		"shf.r.wrap.b32 r29, r26, r31, 60;\n\t"
		"shf.r.wrap.b32 r26, r31, r26, 60;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r29}
		"add.cc.u32 r16, r16, r26;\n\t"
		"addc.u32 r17, r17, r29;\n\t"
		// A = {r4, r5}    B = {r48, r15}    C = {r16, r17}    D = {r26, r29}
		"xor.b32 r48, r48, r16;\n\t"
		"xor.b32 r15, r15, r17;\n\t"
		"shf.r.wrap.b32 r31, r48, r15, 43;\n\t"
		"shf.r.wrap.b32 r48, r15, r48, 43;\n\t"
		// A = {r4, r5}    B = {r48, r31}    C = {r16, r17}    D = {r26, r29}
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r31;\n\t"
		// A = {r4, r5}    B = {r48, r31}    C = {r16, r17}    D = {r26, r29}
		"xor.b32 r15, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r4, r4, r15;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r48, r31}    C = {r16, r17}    D = {r26, r29}
		"xor.b32 r26, r26, r4;\n\t"
		"xor.b32 r29, r29, r5;\n\t"
		"shf.r.wrap.b32 r15, r26, r29, 5;\n\t"
		"shf.r.wrap.b32 r26, r29, r26, 5;\n\t"
		// A = {r4, r5}    B = {r48, r31}    C = {r16, r17}    D = {r15, r26}
		"add.cc.u32 r16, r16, r15;\n\t"
		"addc.u32 r17, r17, r26;\n\t"
		// A = {r4, r5}    B = {r48, r31}    C = {r16, r17}    D = {r15, r26}
		"xor.b32 r48, r48, r16;\n\t"
		"xor.b32 r31, r31, r17;\n\t"
		"shf.r.wrap.b32 r29, r48, r31, 18;\n\t"
		"shf.r.wrap.b32 r48, r31, r48, 18;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r16, r17}    D = {r15, r26}
		"lop3.b32 r31, r4, r29, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r48, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r29, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r48, r17, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r48, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r29, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r48, r17, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r48, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r29, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r48, r17, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r48, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r29, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r48, r17, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r29, r48}    C = {r16, r17}    D = {r15, r26}
		/*
		* |------------------------[ROUND 6.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r12, r10}           |
		* |            v[ 5]            |           {r25, r14}           |
		* |            v[ 6]            |           {r27,  r8}           |
		* |            v[ 7]            |           {r29, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r24}           |
		* |            v[13]            |           {r15, r26}           |
		* |            v[14]            |           {r28, r30}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r30}
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r10;\n\t"
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r30}
		"xor.b32 r31, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r6, r31, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r30}
		"xor.b32 r28, r28, r6;\n\t"
		"xor.b32 r30, r30, r7;\n\t"
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r30}
		"shf.r.wrap.b32 r31, r28, r30, 60;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 60;\n\t"
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r31}
		"add.cc.u32 r18, r18, r28;\n\t"
		"addc.u32 r19, r19, r31;\n\t"
		// A = {r6, r7}    B = {r12, r10}    C = {r18, r19}    D = {r28, r31}
		"xor.b32 r12, r12, r18;\n\t"
		"xor.b32 r10, r10, r19;\n\t"
		"shf.r.wrap.b32 r30, r12, r10, 43;\n\t"
		"shf.r.wrap.b32 r12, r10, r12, 43;\n\t"
		// A = {r6, r7}    B = {r12, r30}    C = {r18, r19}    D = {r28, r31}
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r30;\n\t"
		// A = {r6, r7}    B = {r12, r30}    C = {r18, r19}    D = {r28, r31}
		"xor.b32 r10, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r12, r30}    C = {r18, r19}    D = {r28, r31}
		"xor.b32 r28, r28, r6;\n\t"
		"xor.b32 r31, r31, r7;\n\t"
		"shf.r.wrap.b32 r10, r28, r31, 5;\n\t"
		"shf.r.wrap.b32 r28, r31, r28, 5;\n\t"
		// A = {r6, r7}    B = {r12, r30}    C = {r18, r19}    D = {r10, r28}
		"add.cc.u32 r18, r18, r10;\n\t"
		"addc.u32 r19, r19, r28;\n\t"
		// A = {r6, r7}    B = {r12, r30}    C = {r18, r19}    D = {r10, r28}
		"xor.b32 r12, r12, r18;\n\t"
		"xor.b32 r30, r30, r19;\n\t"
		"shf.r.wrap.b32 r31, r12, r30, 18;\n\t"
		"shf.r.wrap.b32 r12, r30, r12, 18;\n\t"
		// A = {r6, r7}    B = {r31, r12}    C = {r18, r19}    D = {r10, r28}
		"lop3.b32 r30, r6, r31, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r12, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r31, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r12, r19, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r12, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r31, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r12, r19, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r12, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r31, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r12, r19, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r12, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r31, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r12, r19, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r31, r12}    C = {r18, r19}    D = {r10, r28}
		/*
		* |------------------------[ROUND 7.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r12}           |
		* |            v[ 5]            |           {r25, r14}           |
		* |            v[ 6]            |           {r27,  r8}           |
		* |            v[ 7]            |           {r29, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r24}           |
		* |            v[13]            |           {r15, r26}           |
		* |            v[14]            |           {r10, r28}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r24}
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r12;\n\t"
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r24}
		"xor.b32 r30, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r0, r30, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r24}
		"xor.b32 r13, r13, r0;\n\t"
		"xor.b32 r24, r24, r1;\n\t"
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r24}
		"shf.r.wrap.b32 r30, r13, r24, 60;\n\t"
		"shf.r.wrap.b32 r13, r24, r13, 60;\n\t"
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r30}
		"add.cc.u32 r16, r16, r13;\n\t"
		"addc.u32 r17, r17, r30;\n\t"
		// A = {r0, r1}    B = {r31, r12}    C = {r16, r17}    D = {r13, r30}
		"xor.b32 r31, r31, r16;\n\t"
		"xor.b32 r12, r12, r17;\n\t"
		"shf.r.wrap.b32 r24, r31, r12, 43;\n\t"
		"shf.r.wrap.b32 r31, r12, r31, 43;\n\t"
		// A = {r0, r1}    B = {r31, r24}    C = {r16, r17}    D = {r13, r30}
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r24;\n\t"
		// A = {r0, r1}    B = {r31, r24}    C = {r16, r17}    D = {r13, r30}
		"xor.b32 r12, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r31, r24}    C = {r16, r17}    D = {r13, r30}
		"xor.b32 r13, r13, r0;\n\t"
		"xor.b32 r30, r30, r1;\n\t"
		"shf.r.wrap.b32 r12, r13, r30, 5;\n\t"
		"shf.r.wrap.b32 r13, r30, r13, 5;\n\t"
		// A = {r0, r1}    B = {r31, r24}    C = {r16, r17}    D = {r12, r13}
		"add.cc.u32 r16, r16, r12;\n\t"
		"addc.u32 r17, r17, r13;\n\t"
		// A = {r0, r1}    B = {r31, r24}    C = {r16, r17}    D = {r12, r13}
		"xor.b32 r31, r31, r16;\n\t"
		"xor.b32 r24, r24, r17;\n\t"
		"shf.r.wrap.b32 r30, r31, r24, 18;\n\t"
		"shf.r.wrap.b32 r31, r24, r31, 18;\n\t"
		// A = {r0, r1}    B = {r30, r31}    C = {r16, r17}    D = {r12, r13}
		"lop3.b32 r24, r0, r30, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r31, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r30, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r31, r17, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r31, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r30, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r31, r17, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r31, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r30, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r31, r17, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r31, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r30, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r31, r17, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r30, r31}    C = {r16, r17}    D = {r12, r13}
		/*
		* |------------------------[ROUND 7.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           {r25, r14}           |
		* |            v[ 6]            |           {r27,  r8}           |
		* |            v[ 7]            |           {r29, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12, r13}           |
		* |            v[13]            |           {r15, r26}           |
		* |            v[14]            |           {r10, r28}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r26}
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r14;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r26}
		"xor.b32 r24, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r2, r24, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r26}
		"xor.b32 r15, r15, r2;\n\t"
		"xor.b32 r26, r26, r3;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r26}
		"shf.r.wrap.b32 r24, r15, r26, 60;\n\t"
		"shf.r.wrap.b32 r15, r26, r15, 60;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r24}
		"add.cc.u32 r18, r18, r15;\n\t"
		"addc.u32 r19, r19, r24;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r18, r19}    D = {r15, r24}
		"xor.b32 r25, r25, r18;\n\t"
		"xor.b32 r14, r14, r19;\n\t"
		"shf.r.wrap.b32 r26, r25, r14, 43;\n\t"
		"shf.r.wrap.b32 r25, r14, r25, 43;\n\t"
		// A = {r2, r3}    B = {r25, r26}    C = {r18, r19}    D = {r15, r24}
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r26;\n\t"
		// A = {r2, r3}    B = {r25, r26}    C = {r18, r19}    D = {r15, r24}
		"xor.b32 r14, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r25, r26}    C = {r18, r19}    D = {r15, r24}
		"xor.b32 r15, r15, r2;\n\t"
		"xor.b32 r24, r24, r3;\n\t"
		"shf.r.wrap.b32 r14, r15, r24, 5;\n\t"
		"shf.r.wrap.b32 r15, r24, r15, 5;\n\t"
		// A = {r2, r3}    B = {r25, r26}    C = {r18, r19}    D = {r14, r15}
		"add.cc.u32 r18, r18, r14;\n\t"
		"addc.u32 r19, r19, r15;\n\t"
		// A = {r2, r3}    B = {r25, r26}    C = {r18, r19}    D = {r14, r15}
		"xor.b32 r25, r25, r18;\n\t"
		"xor.b32 r26, r26, r19;\n\t"
		"shf.r.wrap.b32 r24, r25, r26, 18;\n\t"
		"shf.r.wrap.b32 r25, r26, r25, 18;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r18, r19}    D = {r14, r15}
		"lop3.b32 r26, r2, r24, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r25, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r24, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r25, r19, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r25, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r24, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r25, r19, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r25, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r24, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r25, r19, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r25, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r24, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r25, r19, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r18, r19}    D = {r14, r15}
		/*
		* |------------------------[ROUND 7.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           {r24, r25}           |
		* |            v[ 6]            |           {r27,  r8}           |
		* |            v[ 7]            |           {r29, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12, r13}           |
		* |            v[13]            |           {r14, r15}           |
		* |            v[14]            |           {r10, r28}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r28}
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r8;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r28}
		"xor.b32 r26, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r4, r26, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r28}
		"xor.b32 r10, r10, r4;\n\t"
		"xor.b32 r28, r28, r5;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r28}
		"shf.r.wrap.b32 r26, r10, r28, 60;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 60;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r26}
		"add.cc.u32 r20, r20, r10;\n\t"
		"addc.u32 r21, r21, r26;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r20, r21}    D = {r10, r26}
		"xor.b32 r27, r27, r20;\n\t"
		"xor.b32 r8, r8, r21;\n\t"
		"shf.r.wrap.b32 r28, r27, r8, 43;\n\t"
		"shf.r.wrap.b32 r27, r8, r27, 43;\n\t"
		// A = {r4, r5}    B = {r27, r28}    C = {r20, r21}    D = {r10, r26}
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r28;\n\t"
		// A = {r4, r5}    B = {r27, r28}    C = {r20, r21}    D = {r10, r26}
		"xor.b32 r8, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r27, r28}    C = {r20, r21}    D = {r10, r26}
		"xor.b32 r10, r10, r4;\n\t"
		"xor.b32 r26, r26, r5;\n\t"
		"shf.r.wrap.b32 r8, r10, r26, 5;\n\t"
		"shf.r.wrap.b32 r10, r26, r10, 5;\n\t"
		// A = {r4, r5}    B = {r27, r28}    C = {r20, r21}    D = {r8, r10}
		"add.cc.u32 r20, r20, r8;\n\t"
		"addc.u32 r21, r21, r10;\n\t"
		// A = {r4, r5}    B = {r27, r28}    C = {r20, r21}    D = {r8, r10}
		"xor.b32 r27, r27, r20;\n\t"
		"xor.b32 r28, r28, r21;\n\t"
		"shf.r.wrap.b32 r26, r27, r28, 18;\n\t"
		"shf.r.wrap.b32 r27, r28, r27, 18;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r20, r21}    D = {r8, r10}
		"lop3.b32 r28, r4, r26, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r27, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r26, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r27, r21, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r27, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r26, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r27, r21, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r27, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r26, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r27, r21, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r27, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r26, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r27, r21, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r20, r21}    D = {r8, r10}
		/*
		* |------------------------[ROUND 7.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           {r24, r25}           |
		* |            v[ 6]            |           {r26, r27}           |
		* |            v[ 7]            |           {r29, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12, r13}           |
		* |            v[13]            |           {r14, r15}           |
		* |            v[14]            |           { r8, r10}           |
		* |            v[15]            |           {r11,  r9}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r9}
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r48;\n\t"
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r9}
		"xor.b32 r28, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r6, r28, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r9}
		"xor.b32 r11, r11, r6;\n\t"
		"xor.b32 r9, r9, r7;\n\t"
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r9}
		"shf.r.wrap.b32 r28, r11, r9, 60;\n\t"
		"shf.r.wrap.b32 r11, r9, r11, 60;\n\t"
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r28}
		"add.cc.u32 r22, r22, r11;\n\t"
		"addc.u32 r23, r23, r28;\n\t"
		// A = {r6, r7}    B = {r29, r48}    C = {r22, r23}    D = {r11, r28}
		"xor.b32 r29, r29, r22;\n\t"
		"xor.b32 r48, r48, r23;\n\t"
		"shf.r.wrap.b32 r9, r29, r48, 43;\n\t"
		"shf.r.wrap.b32 r29, r48, r29, 43;\n\t"
		// A = {r6, r7}    B = {r29, r9}    C = {r22, r23}    D = {r11, r28}
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r9;\n\t"
		// A = {r6, r7}    B = {r29, r9}    C = {r22, r23}    D = {r11, r28}
		"xor.b32 r48, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r29, r9}    C = {r22, r23}    D = {r11, r28}
		"xor.b32 r11, r11, r6;\n\t"
		"xor.b32 r28, r28, r7;\n\t"
		"shf.r.wrap.b32 r48, r11, r28, 5;\n\t"
		"shf.r.wrap.b32 r11, r28, r11, 5;\n\t"
		// A = {r6, r7}    B = {r29, r9}    C = {r22, r23}    D = {r48, r11}
		"add.cc.u32 r22, r22, r48;\n\t"
		"addc.u32 r23, r23, r11;\n\t"
		// A = {r6, r7}    B = {r29, r9}    C = {r22, r23}    D = {r48, r11}
		"xor.b32 r29, r29, r22;\n\t"
		"xor.b32 r9, r9, r23;\n\t"
		"shf.r.wrap.b32 r28, r29, r9, 18;\n\t"
		"shf.r.wrap.b32 r29, r9, r29, 18;\n\t"
		// A = {r6, r7}    B = {r28, r29}    C = {r22, r23}    D = {r48, r11}
		"lop3.b32 r9, r6, r28, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r29, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r28, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r29, r23, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r29, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r28, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r29, r23, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r29, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r28, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r29, r23, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r29, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r28, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r29, r23, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r28, r29}    C = {r22, r23}    D = {r48, r11}
		/*
		* |------------------------[ROUND 7.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           {r24, r25}           |
		* |            v[ 6]            |           {r26, r27}           |
		* |            v[ 7]            |           {r28, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12, r13}           |
		* |            v[13]            |           {r14, r15}           |
		* |            v[14]            |           { r8, r10}           |
		* |            v[15]            |           {r48, r11}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r11}
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r25;\n\t"
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r11}
		"xor.b32 r9, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r9, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r11}
		"xor.b32 r48, r48, r0;\n\t"
		"xor.b32 r11, r11, r1;\n\t"
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r11}
		"shf.r.wrap.b32 r9, r48, r11, 60;\n\t"
		"shf.r.wrap.b32 r48, r11, r48, 60;\n\t"
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r9}
		"add.cc.u32 r20, r20, r48;\n\t"
		"addc.u32 r21, r21, r9;\n\t"
		// A = {r0, r1}    B = {r24, r25}    C = {r20, r21}    D = {r48, r9}
		"xor.b32 r24, r24, r20;\n\t"
		"xor.b32 r25, r25, r21;\n\t"
		"shf.r.wrap.b32 r11, r24, r25, 43;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 43;\n\t"
		// A = {r0, r1}    B = {r24, r11}    C = {r20, r21}    D = {r48, r9}
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r11;\n\t"
		// A = {r0, r1}    B = {r24, r11}    C = {r20, r21}    D = {r48, r9}
		"xor.b32 r25, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r0, r0, r25;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r24, r11}    C = {r20, r21}    D = {r48, r9}
		"xor.b32 r48, r48, r0;\n\t"
		"xor.b32 r9, r9, r1;\n\t"
		"shf.r.wrap.b32 r25, r48, r9, 5;\n\t"
		"shf.r.wrap.b32 r48, r9, r48, 5;\n\t"
		// A = {r0, r1}    B = {r24, r11}    C = {r20, r21}    D = {r25, r48}
		"add.cc.u32 r20, r20, r25;\n\t"
		"addc.u32 r21, r21, r48;\n\t"
		// A = {r0, r1}    B = {r24, r11}    C = {r20, r21}    D = {r25, r48}
		"xor.b32 r24, r24, r20;\n\t"
		"xor.b32 r11, r11, r21;\n\t"
		"shf.r.wrap.b32 r9, r24, r11, 18;\n\t"
		"shf.r.wrap.b32 r24, r11, r24, 18;\n\t"
		// A = {r0, r1}    B = {r9, r24}    C = {r20, r21}    D = {r25, r48}
		"lop3.b32 r11, r0, r9, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r24, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r9, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r24, r21, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r0, r9, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r24, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r9, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r24, r21, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r0, r9, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r24, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r9, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r24, r21, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r0, r9, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r24, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r9, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r24, r21, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r9, r24}    C = {r20, r21}    D = {r25, r48}
		/*
		* |------------------------[ROUND 7.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           { r9, r24}           |
		* |            v[ 6]            |           {r26, r27}           |
		* |            v[ 7]            |           {r28, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r12, r13}           |
		* |            v[13]            |           {r14, r15}           |
		* |            v[14]            |           { r8, r10}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r13}
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r27;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r13}
		"xor.b32 r11, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r2, r11, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r13}
		"xor.b32 r12, r12, r2;\n\t"
		"xor.b32 r13, r13, r3;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r13}
		"shf.r.wrap.b32 r11, r12, r13, 60;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 60;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r11}
		"add.cc.u32 r22, r22, r12;\n\t"
		"addc.u32 r23, r23, r11;\n\t"
		// A = {r2, r3}    B = {r26, r27}    C = {r22, r23}    D = {r12, r11}
		"xor.b32 r26, r26, r22;\n\t"
		"xor.b32 r27, r27, r23;\n\t"
		"shf.r.wrap.b32 r13, r26, r27, 43;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 43;\n\t"
		// A = {r2, r3}    B = {r26, r13}    C = {r22, r23}    D = {r12, r11}
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r13;\n\t"
		// A = {r2, r3}    B = {r26, r13}    C = {r22, r23}    D = {r12, r11}
		"xor.b32 r27, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r2, r2, r27;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r26, r13}    C = {r22, r23}    D = {r12, r11}
		"xor.b32 r12, r12, r2;\n\t"
		"xor.b32 r11, r11, r3;\n\t"
		"shf.r.wrap.b32 r27, r12, r11, 5;\n\t"
		"shf.r.wrap.b32 r12, r11, r12, 5;\n\t"
		// A = {r2, r3}    B = {r26, r13}    C = {r22, r23}    D = {r27, r12}
		"add.cc.u32 r22, r22, r27;\n\t"
		"addc.u32 r23, r23, r12;\n\t"
		// A = {r2, r3}    B = {r26, r13}    C = {r22, r23}    D = {r27, r12}
		"xor.b32 r26, r26, r22;\n\t"
		"xor.b32 r13, r13, r23;\n\t"
		"shf.r.wrap.b32 r11, r26, r13, 18;\n\t"
		"shf.r.wrap.b32 r26, r13, r26, 18;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r22, r23}    D = {r27, r12}
		"lop3.b32 r13, r2, r11, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r26, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r11, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r26, r23, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r2, r11, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r26, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r11, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r26, r23, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r2, r11, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r26, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r11, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r26, r23, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r2, r11, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r26, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r11, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r26, r23, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r11, r26}    C = {r22, r23}    D = {r27, r12}
		/*
		* |------------------------[ROUND 7.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           { r9, r24}           |
		* |            v[ 6]            |           {r11, r26}           |
		* |            v[ 7]            |           {r28, r29}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27, r12}           |
		* |            v[13]            |           {r14, r15}           |
		* |            v[14]            |           { r8, r10}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r15}
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r29;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r15}
		"xor.b32 r13, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r4, r13, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r15}
		"xor.b32 r14, r14, r4;\n\t"
		"xor.b32 r15, r15, r5;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r15}
		"shf.r.wrap.b32 r13, r14, r15, 60;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 60;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r13}
		"add.cc.u32 r16, r16, r14;\n\t"
		"addc.u32 r17, r17, r13;\n\t"
		// A = {r4, r5}    B = {r28, r29}    C = {r16, r17}    D = {r14, r13}
		"xor.b32 r28, r28, r16;\n\t"
		"xor.b32 r29, r29, r17;\n\t"
		"shf.r.wrap.b32 r15, r28, r29, 43;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 43;\n\t"
		// A = {r4, r5}    B = {r28, r15}    C = {r16, r17}    D = {r14, r13}
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r15;\n\t"
		// A = {r4, r5}    B = {r28, r15}    C = {r16, r17}    D = {r14, r13}
		"xor.b32 r29, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r4, r4, r29;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r28, r15}    C = {r16, r17}    D = {r14, r13}
		"xor.b32 r14, r14, r4;\n\t"
		"xor.b32 r13, r13, r5;\n\t"
		"shf.r.wrap.b32 r29, r14, r13, 5;\n\t"
		"shf.r.wrap.b32 r14, r13, r14, 5;\n\t"
		// A = {r4, r5}    B = {r28, r15}    C = {r16, r17}    D = {r29, r14}
		"add.cc.u32 r16, r16, r29;\n\t"
		"addc.u32 r17, r17, r14;\n\t"
		// A = {r4, r5}    B = {r28, r15}    C = {r16, r17}    D = {r29, r14}
		"xor.b32 r28, r28, r16;\n\t"
		"xor.b32 r15, r15, r17;\n\t"
		"shf.r.wrap.b32 r13, r28, r15, 18;\n\t"
		"shf.r.wrap.b32 r28, r15, r28, 18;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r16, r17}    D = {r29, r14}
		"lop3.b32 r15, r4, r13, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r28, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r13, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r28, r17, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r4, r13, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r28, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r13, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r28, r17, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r4, r13, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r28, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r13, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r28, r17, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r4, r13, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r28, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r13, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r28, r17, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r13, r28}    C = {r16, r17}    D = {r29, r14}
		/*
		* |------------------------[ROUND 7.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r30, r31}           |
		* |            v[ 5]            |           { r9, r24}           |
		* |            v[ 6]            |           {r11, r26}           |
		* |            v[ 7]            |           {r13, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27, r12}           |
		* |            v[13]            |           {r29, r14}           |
		* |            v[14]            |           { r8, r10}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r10}
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r31;\n\t"
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r10}
		"xor.b32 r15, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r6, r15, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r10}
		"xor.b32 r8, r8, r6;\n\t"
		"xor.b32 r10, r10, r7;\n\t"
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r10}
		"shf.r.wrap.b32 r15, r8, r10, 60;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 60;\n\t"
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r15}
		"add.cc.u32 r18, r18, r8;\n\t"
		"addc.u32 r19, r19, r15;\n\t"
		// A = {r6, r7}    B = {r30, r31}    C = {r18, r19}    D = {r8, r15}
		"xor.b32 r30, r30, r18;\n\t"
		"xor.b32 r31, r31, r19;\n\t"
		"shf.r.wrap.b32 r10, r30, r31, 43;\n\t"
		"shf.r.wrap.b32 r30, r31, r30, 43;\n\t"
		// A = {r6, r7}    B = {r30, r10}    C = {r18, r19}    D = {r8, r15}
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r10;\n\t"
		// A = {r6, r7}    B = {r30, r10}    C = {r18, r19}    D = {r8, r15}
		"xor.b32 r31, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r6, r6, r31;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r30, r10}    C = {r18, r19}    D = {r8, r15}
		"xor.b32 r8, r8, r6;\n\t"
		"xor.b32 r15, r15, r7;\n\t"
		"shf.r.wrap.b32 r31, r8, r15, 5;\n\t"
		"shf.r.wrap.b32 r8, r15, r8, 5;\n\t"
		// A = {r6, r7}    B = {r30, r10}    C = {r18, r19}    D = {r31, r8}
		"add.cc.u32 r18, r18, r31;\n\t"
		"addc.u32 r19, r19, r8;\n\t"
		// A = {r6, r7}    B = {r30, r10}    C = {r18, r19}    D = {r31, r8}
		"xor.b32 r30, r30, r18;\n\t"
		"xor.b32 r10, r10, r19;\n\t"
		"shf.r.wrap.b32 r15, r30, r10, 18;\n\t"
		"shf.r.wrap.b32 r30, r10, r30, 18;\n\t"
		// A = {r6, r7}    B = {r15, r30}    C = {r18, r19}    D = {r31, r8}
		"lop3.b32 r10, r6, r15, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r30, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r15, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r30, r19, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r6, r15, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r30, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r15, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r30, r19, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r6, r15, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r30, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r15, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r30, r19, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r6, r15, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r30, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r15, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r30, r19, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r15, r30}    C = {r18, r19}    D = {r31, r8}
		/*
		* |------------------------[ROUND 8.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r30}           |
		* |            v[ 5]            |           { r9, r24}           |
		* |            v[ 6]            |           {r11, r26}           |
		* |            v[ 7]            |           {r13, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27, r12}           |
		* |            v[13]            |           {r29, r14}           |
		* |            v[14]            |           {r31,  r8}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r12}
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r30;\n\t"
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r12}
		"xor.b32 r10, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r0, r10, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r12}
		"xor.b32 r27, r27, r0;\n\t"
		"xor.b32 r12, r12, r1;\n\t"
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r12}
		"shf.r.wrap.b32 r10, r27, r12, 60;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 60;\n\t"
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r10}
		"add.cc.u32 r16, r16, r27;\n\t"
		"addc.u32 r17, r17, r10;\n\t"
		// A = {r0, r1}    B = {r15, r30}    C = {r16, r17}    D = {r27, r10}
		"xor.b32 r15, r15, r16;\n\t"
		"xor.b32 r30, r30, r17;\n\t"
		"shf.r.wrap.b32 r12, r15, r30, 43;\n\t"
		"shf.r.wrap.b32 r15, r30, r15, 43;\n\t"
		// A = {r0, r1}    B = {r15, r12}    C = {r16, r17}    D = {r27, r10}
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r12;\n\t"
		// A = {r0, r1}    B = {r15, r12}    C = {r16, r17}    D = {r27, r10}
		"xor.b32 r30, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r0, r0, r30;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r15, r12}    C = {r16, r17}    D = {r27, r10}
		"xor.b32 r27, r27, r0;\n\t"
		"xor.b32 r10, r10, r1;\n\t"
		"shf.r.wrap.b32 r30, r27, r10, 5;\n\t"
		"shf.r.wrap.b32 r27, r10, r27, 5;\n\t"
		// A = {r0, r1}    B = {r15, r12}    C = {r16, r17}    D = {r30, r27}
		"add.cc.u32 r16, r16, r30;\n\t"
		"addc.u32 r17, r17, r27;\n\t"
		// A = {r0, r1}    B = {r15, r12}    C = {r16, r17}    D = {r30, r27}
		"xor.b32 r15, r15, r16;\n\t"
		"xor.b32 r12, r12, r17;\n\t"
		"shf.r.wrap.b32 r10, r15, r12, 18;\n\t"
		"shf.r.wrap.b32 r15, r12, r15, 18;\n\t"
		// A = {r0, r1}    B = {r10, r15}    C = {r16, r17}    D = {r30, r27}
		"lop3.b32 r12, r0, r10, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r15, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r10, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r15, r17, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r0, r10, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r15, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r10, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r15, r17, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r0, r10, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r15, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r10, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r15, r17, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r0, r10, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r15, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r10, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r15, r17, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r10, r15}    C = {r16, r17}    D = {r30, r27}
		/*
		* |------------------------[ROUND 8.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           { r9, r24}           |
		* |            v[ 6]            |           {r11, r26}           |
		* |            v[ 7]            |           {r13, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r27}           |
		* |            v[13]            |           {r29, r14}           |
		* |            v[14]            |           {r31,  r8}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r14}
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r24;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r14}
		"xor.b32 r12, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r2, r12, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r14}
		"xor.b32 r29, r29, r2;\n\t"
		"xor.b32 r14, r14, r3;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r14}
		"shf.r.wrap.b32 r12, r29, r14, 60;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 60;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r12}
		"add.cc.u32 r18, r18, r29;\n\t"
		"addc.u32 r19, r19, r12;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r18, r19}    D = {r29, r12}
		"xor.b32 r9, r9, r18;\n\t"
		"xor.b32 r24, r24, r19;\n\t"
		"shf.r.wrap.b32 r14, r9, r24, 43;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 43;\n\t"
		// A = {r2, r3}    B = {r9, r14}    C = {r18, r19}    D = {r29, r12}
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r14;\n\t"
		// A = {r2, r3}    B = {r9, r14}    C = {r18, r19}    D = {r29, r12}
		"xor.b32 r24, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r2, r2, r24;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r9, r14}    C = {r18, r19}    D = {r29, r12}
		"xor.b32 r29, r29, r2;\n\t"
		"xor.b32 r12, r12, r3;\n\t"
		"shf.r.wrap.b32 r24, r29, r12, 5;\n\t"
		"shf.r.wrap.b32 r29, r12, r29, 5;\n\t"
		// A = {r2, r3}    B = {r9, r14}    C = {r18, r19}    D = {r24, r29}
		"add.cc.u32 r18, r18, r24;\n\t"
		"addc.u32 r19, r19, r29;\n\t"
		// A = {r2, r3}    B = {r9, r14}    C = {r18, r19}    D = {r24, r29}
		"xor.b32 r9, r9, r18;\n\t"
		"xor.b32 r14, r14, r19;\n\t"
		"shf.r.wrap.b32 r12, r9, r14, 18;\n\t"
		"shf.r.wrap.b32 r9, r14, r9, 18;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r18, r19}    D = {r24, r29}
		"lop3.b32 r14, r2, r12, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r9, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r12, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r9, r19, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r2, r12, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r9, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r12, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r9, r19, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r2, r12, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r9, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r12, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r9, r19, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r2, r12, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r9, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r12, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r9, r19, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r18, r19}    D = {r24, r29}
		/*
		* |------------------------[ROUND 8.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r12,  r9}           |
		* |            v[ 6]            |           {r11, r26}           |
		* |            v[ 7]            |           {r13, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r27}           |
		* |            v[13]            |           {r24, r29}           |
		* |            v[14]            |           {r31,  r8}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r8}
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r26;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r8}
		"xor.b32 r14, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r4, r14, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r8}
		"xor.b32 r31, r31, r4;\n\t"
		"xor.b32 r8, r8, r5;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r8}
		"shf.r.wrap.b32 r14, r31, r8, 60;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 60;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r14}
		"add.cc.u32 r20, r20, r31;\n\t"
		"addc.u32 r21, r21, r14;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r20, r21}    D = {r31, r14}
		"xor.b32 r11, r11, r20;\n\t"
		"xor.b32 r26, r26, r21;\n\t"
		"shf.r.wrap.b32 r8, r11, r26, 43;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 43;\n\t"
		// A = {r4, r5}    B = {r11, r8}    C = {r20, r21}    D = {r31, r14}
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r8;\n\t"
		// A = {r4, r5}    B = {r11, r8}    C = {r20, r21}    D = {r31, r14}
		"xor.b32 r26, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r4, r4, r26;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r11, r8}    C = {r20, r21}    D = {r31, r14}
		"xor.b32 r31, r31, r4;\n\t"
		"xor.b32 r14, r14, r5;\n\t"
		"shf.r.wrap.b32 r26, r31, r14, 5;\n\t"
		"shf.r.wrap.b32 r31, r14, r31, 5;\n\t"
		// A = {r4, r5}    B = {r11, r8}    C = {r20, r21}    D = {r26, r31}
		"add.cc.u32 r20, r20, r26;\n\t"
		"addc.u32 r21, r21, r31;\n\t"
		// A = {r4, r5}    B = {r11, r8}    C = {r20, r21}    D = {r26, r31}
		"xor.b32 r11, r11, r20;\n\t"
		"xor.b32 r8, r8, r21;\n\t"
		"shf.r.wrap.b32 r14, r11, r8, 18;\n\t"
		"shf.r.wrap.b32 r11, r8, r11, 18;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r20, r21}    D = {r26, r31}
		"lop3.b32 r8, r4, r14, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r11, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r14, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r11, r21, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r4, r14, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r11, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r14, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r11, r21, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r4, r14, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r11, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r14, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r11, r21, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r4, r14, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r11, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r14, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r11, r21, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r20, r21}    D = {r26, r31}
		/*
		* |------------------------[ROUND 8.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r12,  r9}           |
		* |            v[ 6]            |           {r14, r11}           |
		* |            v[ 7]            |           {r13, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r27}           |
		* |            v[13]            |           {r24, r29}           |
		* |            v[14]            |           {r26, r31}           |
		* |            v[15]            |           {r25, r48}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r48}
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r28;\n\t"
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r48}
		"xor.b32 r8, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r8, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r48}
		"xor.b32 r25, r25, r6;\n\t"
		"xor.b32 r48, r48, r7;\n\t"
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r48}
		"shf.r.wrap.b32 r8, r25, r48, 60;\n\t"
		"shf.r.wrap.b32 r25, r48, r25, 60;\n\t"
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r8}
		"add.cc.u32 r22, r22, r25;\n\t"
		"addc.u32 r23, r23, r8;\n\t"
		// A = {r6, r7}    B = {r13, r28}    C = {r22, r23}    D = {r25, r8}
		"xor.b32 r13, r13, r22;\n\t"
		"xor.b32 r28, r28, r23;\n\t"
		"shf.r.wrap.b32 r48, r13, r28, 43;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 43;\n\t"
		// A = {r6, r7}    B = {r13, r48}    C = {r22, r23}    D = {r25, r8}
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r48;\n\t"
		// A = {r6, r7}    B = {r13, r48}    C = {r22, r23}    D = {r25, r8}
		"xor.b32 r28, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r6, r6, r28;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r13, r48}    C = {r22, r23}    D = {r25, r8}
		"xor.b32 r25, r25, r6;\n\t"
		"xor.b32 r8, r8, r7;\n\t"
		"shf.r.wrap.b32 r28, r25, r8, 5;\n\t"
		"shf.r.wrap.b32 r25, r8, r25, 5;\n\t"
		// A = {r6, r7}    B = {r13, r48}    C = {r22, r23}    D = {r28, r25}
		"add.cc.u32 r22, r22, r28;\n\t"
		"addc.u32 r23, r23, r25;\n\t"
		// A = {r6, r7}    B = {r13, r48}    C = {r22, r23}    D = {r28, r25}
		"xor.b32 r13, r13, r22;\n\t"
		"xor.b32 r48, r48, r23;\n\t"
		"shf.r.wrap.b32 r8, r13, r48, 18;\n\t"
		"shf.r.wrap.b32 r13, r48, r13, 18;\n\t"
		// A = {r6, r7}    B = {r8, r13}    C = {r22, r23}    D = {r28, r25}
		"lop3.b32 r48, r6, r8, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r13, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r8, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r13, r23, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r6, r8, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r13, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r8, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r13, r23, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r6, r8, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r13, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r8, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r13, r23, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r6, r8, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r13, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r8, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r13, r23, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r8, r13}    C = {r22, r23}    D = {r28, r25}
		/*
		* |------------------------[ROUND 8.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r12,  r9}           |
		* |            v[ 6]            |           {r14, r11}           |
		* |            v[ 7]            |           { r8, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r27}           |
		* |            v[13]            |           {r24, r29}           |
		* |            v[14]            |           {r26, r31}           |
		* |            v[15]            |           {r28, r25}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r25}
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r9;\n\t"
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r25}
		"xor.b32 r48, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r48, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r25}
		"xor.b32 r28, r28, r0;\n\t"
		"xor.b32 r25, r25, r1;\n\t"
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r25}
		"shf.r.wrap.b32 r48, r28, r25, 60;\n\t"
		"shf.r.wrap.b32 r28, r25, r28, 60;\n\t"
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r48}
		"add.cc.u32 r20, r20, r28;\n\t"
		"addc.u32 r21, r21, r48;\n\t"
		// A = {r0, r1}    B = {r12, r9}    C = {r20, r21}    D = {r28, r48}
		"xor.b32 r12, r12, r20;\n\t"
		"xor.b32 r9, r9, r21;\n\t"
		"shf.r.wrap.b32 r25, r12, r9, 43;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 43;\n\t"
		// A = {r0, r1}    B = {r12, r25}    C = {r20, r21}    D = {r28, r48}
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r25;\n\t"
		// A = {r0, r1}    B = {r12, r25}    C = {r20, r21}    D = {r28, r48}
		"xor.b32 r9, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r12, r25}    C = {r20, r21}    D = {r28, r48}
		"xor.b32 r28, r28, r0;\n\t"
		"xor.b32 r48, r48, r1;\n\t"
		"shf.r.wrap.b32 r9, r28, r48, 5;\n\t"
		"shf.r.wrap.b32 r28, r48, r28, 5;\n\t"
		// A = {r0, r1}    B = {r12, r25}    C = {r20, r21}    D = {r9, r28}
		"add.cc.u32 r20, r20, r9;\n\t"
		"addc.u32 r21, r21, r28;\n\t"
		// A = {r0, r1}    B = {r12, r25}    C = {r20, r21}    D = {r9, r28}
		"xor.b32 r12, r12, r20;\n\t"
		"xor.b32 r25, r25, r21;\n\t"
		"shf.r.wrap.b32 r48, r12, r25, 18;\n\t"
		"shf.r.wrap.b32 r12, r25, r12, 18;\n\t"
		// A = {r0, r1}    B = {r48, r12}    C = {r20, r21}    D = {r9, r28}
		"lop3.b32 r25, r0, r48, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r12, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r48, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r12, r21, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r12, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r48, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r12, r21, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r12, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r48, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r12, r21, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r0, r48, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r12, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r48, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r12, r21, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r48, r12}    C = {r20, r21}    D = {r9, r28}
		/*
		* |------------------------[ROUND 8.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r48, r12}           |
		* |            v[ 6]            |           {r14, r11}           |
		* |            v[ 7]            |           { r8, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r30, r27}           |
		* |            v[13]            |           {r24, r29}           |
		* |            v[14]            |           {r26, r31}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r27}
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r11;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r27}
		"xor.b32 r25, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r25, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r27}
		"xor.b32 r30, r30, r2;\n\t"
		"xor.b32 r27, r27, r3;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r27}
		"shf.r.wrap.b32 r25, r30, r27, 60;\n\t"
		"shf.r.wrap.b32 r30, r27, r30, 60;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r25}
		"add.cc.u32 r22, r22, r30;\n\t"
		"addc.u32 r23, r23, r25;\n\t"
		// A = {r2, r3}    B = {r14, r11}    C = {r22, r23}    D = {r30, r25}
		"xor.b32 r14, r14, r22;\n\t"
		"xor.b32 r11, r11, r23;\n\t"
		"shf.r.wrap.b32 r27, r14, r11, 43;\n\t"
		"shf.r.wrap.b32 r14, r11, r14, 43;\n\t"
		// A = {r2, r3}    B = {r14, r27}    C = {r22, r23}    D = {r30, r25}
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r27;\n\t"
		// A = {r2, r3}    B = {r14, r27}    C = {r22, r23}    D = {r30, r25}
		"xor.b32 r11, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r2, r2, r11;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r14, r27}    C = {r22, r23}    D = {r30, r25}
		"xor.b32 r30, r30, r2;\n\t"
		"xor.b32 r25, r25, r3;\n\t"
		"shf.r.wrap.b32 r11, r30, r25, 5;\n\t"
		"shf.r.wrap.b32 r30, r25, r30, 5;\n\t"
		// A = {r2, r3}    B = {r14, r27}    C = {r22, r23}    D = {r11, r30}
		"add.cc.u32 r22, r22, r11;\n\t"
		"addc.u32 r23, r23, r30;\n\t"
		// A = {r2, r3}    B = {r14, r27}    C = {r22, r23}    D = {r11, r30}
		"xor.b32 r14, r14, r22;\n\t"
		"xor.b32 r27, r27, r23;\n\t"
		"shf.r.wrap.b32 r25, r14, r27, 18;\n\t"
		"shf.r.wrap.b32 r14, r27, r14, 18;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r22, r23}    D = {r11, r30}
		"lop3.b32 r27, r2, r25, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r14, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r25, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r14, r23, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r14, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r25, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r14, r23, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r14, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r25, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r14, r23, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r2, r25, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r14, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r25, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r14, r23, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r25, r14}    C = {r22, r23}    D = {r11, r30}
		/*
		* |------------------------[ROUND 8.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r48, r12}           |
		* |            v[ 6]            |           {r25, r14}           |
		* |            v[ 7]            |           { r8, r13}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r30}           |
		* |            v[13]            |           {r24, r29}           |
		* |            v[14]            |           {r26, r31}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r29}
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r13;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r29}
		"xor.b32 r27, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r27, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r29}
		"xor.b32 r24, r24, r4;\n\t"
		"xor.b32 r29, r29, r5;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r29}
		"shf.r.wrap.b32 r27, r24, r29, 60;\n\t"
		"shf.r.wrap.b32 r24, r29, r24, 60;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r27}
		"add.cc.u32 r16, r16, r24;\n\t"
		"addc.u32 r17, r17, r27;\n\t"
		// A = {r4, r5}    B = {r8, r13}    C = {r16, r17}    D = {r24, r27}
		"xor.b32 r8, r8, r16;\n\t"
		"xor.b32 r13, r13, r17;\n\t"
		"shf.r.wrap.b32 r29, r8, r13, 43;\n\t"
		"shf.r.wrap.b32 r8, r13, r8, 43;\n\t"
		// A = {r4, r5}    B = {r8, r29}    C = {r16, r17}    D = {r24, r27}
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r29;\n\t"
		// A = {r4, r5}    B = {r8, r29}    C = {r16, r17}    D = {r24, r27}
		"xor.b32 r13, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r4, r4, r13;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r8, r29}    C = {r16, r17}    D = {r24, r27}
		"xor.b32 r24, r24, r4;\n\t"
		"xor.b32 r27, r27, r5;\n\t"
		"shf.r.wrap.b32 r13, r24, r27, 5;\n\t"
		"shf.r.wrap.b32 r24, r27, r24, 5;\n\t"
		// A = {r4, r5}    B = {r8, r29}    C = {r16, r17}    D = {r13, r24}
		"add.cc.u32 r16, r16, r13;\n\t"
		"addc.u32 r17, r17, r24;\n\t"
		// A = {r4, r5}    B = {r8, r29}    C = {r16, r17}    D = {r13, r24}
		"xor.b32 r8, r8, r16;\n\t"
		"xor.b32 r29, r29, r17;\n\t"
		"shf.r.wrap.b32 r27, r8, r29, 18;\n\t"
		"shf.r.wrap.b32 r8, r29, r8, 18;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r16, r17}    D = {r13, r24}
		"lop3.b32 r29, r4, r27, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r8, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r27, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r8, r17, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r8, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r27, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r8, r17, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r8, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r27, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r8, r17, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r4, r27, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r8, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r27, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r8, r17, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r27, r8}    C = {r16, r17}    D = {r13, r24}
		/*
		* |------------------------[ROUND 8.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r10, r15}           |
		* |            v[ 5]            |           {r48, r12}           |
		* |            v[ 6]            |           {r25, r14}           |
		* |            v[ 7]            |           {r27,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r30}           |
		* |            v[13]            |           {r13, r24}           |
		* |            v[14]            |           {r26, r31}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r31}
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r15;\n\t"
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r31}
		"xor.b32 r29, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r6, r29, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r31}
		"xor.b32 r26, r26, r6;\n\t"
		"xor.b32 r31, r31, r7;\n\t"
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r31}
		"shf.r.wrap.b32 r29, r26, r31, 60;\n\t"
		"shf.r.wrap.b32 r26, r31, r26, 60;\n\t"
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r29}
		"add.cc.u32 r18, r18, r26;\n\t"
		"addc.u32 r19, r19, r29;\n\t"
		// A = {r6, r7}    B = {r10, r15}    C = {r18, r19}    D = {r26, r29}
		"xor.b32 r10, r10, r18;\n\t"
		"xor.b32 r15, r15, r19;\n\t"
		"shf.r.wrap.b32 r31, r10, r15, 43;\n\t"
		"shf.r.wrap.b32 r10, r15, r10, 43;\n\t"
		// A = {r6, r7}    B = {r10, r31}    C = {r18, r19}    D = {r26, r29}
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r31;\n\t"
		// A = {r6, r7}    B = {r10, r31}    C = {r18, r19}    D = {r26, r29}
		"xor.b32 r15, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r10, r31}    C = {r18, r19}    D = {r26, r29}
		"xor.b32 r26, r26, r6;\n\t"
		"xor.b32 r29, r29, r7;\n\t"
		"shf.r.wrap.b32 r15, r26, r29, 5;\n\t"
		"shf.r.wrap.b32 r26, r29, r26, 5;\n\t"
		// A = {r6, r7}    B = {r10, r31}    C = {r18, r19}    D = {r15, r26}
		"add.cc.u32 r18, r18, r15;\n\t"
		"addc.u32 r19, r19, r26;\n\t"
		// A = {r6, r7}    B = {r10, r31}    C = {r18, r19}    D = {r15, r26}
		"xor.b32 r10, r10, r18;\n\t"
		"xor.b32 r31, r31, r19;\n\t"
		"shf.r.wrap.b32 r29, r10, r31, 18;\n\t"
		"shf.r.wrap.b32 r10, r31, r10, 18;\n\t"
		// A = {r6, r7}    B = {r29, r10}    C = {r18, r19}    D = {r15, r26}
		"lop3.b32 r31, r6, r29, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r10, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r29, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r10, r19, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r10, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r29, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r10, r19, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r10, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r29, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r10, r19, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r6, r29, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r10, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r29, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r10, r19, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r29, r10}    C = {r18, r19}    D = {r15, r26}
		/*
		* |------------------------[ROUND 9.0]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r10}           |
		* |            v[ 5]            |           {r48, r12}           |
		* |            v[ 6]            |           {r25, r14}           |
		* |            v[ 7]            |           {r27,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r30}           |
		* |            v[13]            |           {r13, r24}           |
		* |            v[14]            |           {r15, r26}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r30}
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r10;\n\t"
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r30}
		"xor.b32 r31, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r31, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r30}
		"xor.b32 r11, r11, r0;\n\t"
		"xor.b32 r30, r30, r1;\n\t"
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r30}
		"shf.r.wrap.b32 r31, r11, r30, 60;\n\t"
		"shf.r.wrap.b32 r11, r30, r11, 60;\n\t"
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r31}
		"add.cc.u32 r16, r16, r11;\n\t"
		"addc.u32 r17, r17, r31;\n\t"
		// A = {r0, r1}    B = {r29, r10}    C = {r16, r17}    D = {r11, r31}
		"xor.b32 r29, r29, r16;\n\t"
		"xor.b32 r10, r10, r17;\n\t"
		"shf.r.wrap.b32 r30, r29, r10, 43;\n\t"
		"shf.r.wrap.b32 r29, r10, r29, 43;\n\t"
		// A = {r0, r1}    B = {r29, r30}    C = {r16, r17}    D = {r11, r31}
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r30;\n\t"
		// A = {r0, r1}    B = {r29, r30}    C = {r16, r17}    D = {r11, r31}
		"xor.b32 r10, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r29, r30}    C = {r16, r17}    D = {r11, r31}
		"xor.b32 r11, r11, r0;\n\t"
		"xor.b32 r31, r31, r1;\n\t"
		"shf.r.wrap.b32 r10, r11, r31, 5;\n\t"
		"shf.r.wrap.b32 r11, r31, r11, 5;\n\t"
		// A = {r0, r1}    B = {r29, r30}    C = {r16, r17}    D = {r10, r11}
		"add.cc.u32 r16, r16, r10;\n\t"
		"addc.u32 r17, r17, r11;\n\t"
		// A = {r0, r1}    B = {r29, r30}    C = {r16, r17}    D = {r10, r11}
		"xor.b32 r29, r29, r16;\n\t"
		"xor.b32 r30, r30, r17;\n\t"
		"shf.r.wrap.b32 r31, r29, r30, 18;\n\t"
		"shf.r.wrap.b32 r29, r30, r29, 18;\n\t"
		// A = {r0, r1}    B = {r31, r29}    C = {r16, r17}    D = {r10, r11}
		"lop3.b32 r30, r0, r31, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r29, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r31, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r29, r17, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r29, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r31, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r29, r17, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r29, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r31, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r29, r17, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r0, r31, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r29, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r31, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r29, r17, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r31, r29}    C = {r16, r17}    D = {r10, r11}
		/*
		* |------------------------[ROUND 9.1]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r48, r12}           |
		* |            v[ 6]            |           {r25, r14}           |
		* |            v[ 7]            |           {r27,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r11}           |
		* |            v[13]            |           {r13, r24}           |
		* |            v[14]            |           {r15, r26}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r24}
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r12;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r24}
		"xor.b32 r30, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r2, r30, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r24}
		"xor.b32 r13, r13, r2;\n\t"
		"xor.b32 r24, r24, r3;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r24}
		"shf.r.wrap.b32 r30, r13, r24, 60;\n\t"
		"shf.r.wrap.b32 r13, r24, r13, 60;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r30}
		"add.cc.u32 r18, r18, r13;\n\t"
		"addc.u32 r19, r19, r30;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r18, r19}    D = {r13, r30}
		"xor.b32 r48, r48, r18;\n\t"
		"xor.b32 r12, r12, r19;\n\t"
		"shf.r.wrap.b32 r24, r48, r12, 43;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 43;\n\t"
		// A = {r2, r3}    B = {r48, r24}    C = {r18, r19}    D = {r13, r30}
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r24;\n\t"
		// A = {r2, r3}    B = {r48, r24}    C = {r18, r19}    D = {r13, r30}
		"xor.b32 r12, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r48, r24}    C = {r18, r19}    D = {r13, r30}
		"xor.b32 r13, r13, r2;\n\t"
		"xor.b32 r30, r30, r3;\n\t"
		"shf.r.wrap.b32 r12, r13, r30, 5;\n\t"
		"shf.r.wrap.b32 r13, r30, r13, 5;\n\t"
		// A = {r2, r3}    B = {r48, r24}    C = {r18, r19}    D = {r12, r13}
		"add.cc.u32 r18, r18, r12;\n\t"
		"addc.u32 r19, r19, r13;\n\t"
		// A = {r2, r3}    B = {r48, r24}    C = {r18, r19}    D = {r12, r13}
		"xor.b32 r48, r48, r18;\n\t"
		"xor.b32 r24, r24, r19;\n\t"
		"shf.r.wrap.b32 r30, r48, r24, 18;\n\t"
		"shf.r.wrap.b32 r48, r24, r48, 18;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r18, r19}    D = {r12, r13}
		"lop3.b32 r24, r2, r30, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r48, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r30, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r48, r19, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r48, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r30, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r48, r19, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r48, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r30, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r48, r19, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r2, r30, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r48, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r30, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r48, r19, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r18, r19}    D = {r12, r13}
		/*
		* |------------------------[ROUND 9.2]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r30, r48}           |
		* |            v[ 6]            |           {r25, r14}           |
		* |            v[ 7]            |           {r27,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r11}           |
		* |            v[13]            |           {r12, r13}           |
		* |            v[14]            |           {r15, r26}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r26}
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r14;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r26}
		"xor.b32 r24, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r4, r24, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r26}
		"xor.b32 r15, r15, r4;\n\t"
		"xor.b32 r26, r26, r5;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r26}
		"shf.r.wrap.b32 r24, r15, r26, 60;\n\t"
		"shf.r.wrap.b32 r15, r26, r15, 60;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r24}
		"add.cc.u32 r20, r20, r15;\n\t"
		"addc.u32 r21, r21, r24;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r20, r21}    D = {r15, r24}
		"xor.b32 r25, r25, r20;\n\t"
		"xor.b32 r14, r14, r21;\n\t"
		"shf.r.wrap.b32 r26, r25, r14, 43;\n\t"
		"shf.r.wrap.b32 r25, r14, r25, 43;\n\t"
		// A = {r4, r5}    B = {r25, r26}    C = {r20, r21}    D = {r15, r24}
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r26;\n\t"
		// A = {r4, r5}    B = {r25, r26}    C = {r20, r21}    D = {r15, r24}
		"xor.b32 r14, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r25, r26}    C = {r20, r21}    D = {r15, r24}
		"xor.b32 r15, r15, r4;\n\t"
		"xor.b32 r24, r24, r5;\n\t"
		"shf.r.wrap.b32 r14, r15, r24, 5;\n\t"
		"shf.r.wrap.b32 r15, r24, r15, 5;\n\t"
		// A = {r4, r5}    B = {r25, r26}    C = {r20, r21}    D = {r14, r15}
		"add.cc.u32 r20, r20, r14;\n\t"
		"addc.u32 r21, r21, r15;\n\t"
		// A = {r4, r5}    B = {r25, r26}    C = {r20, r21}    D = {r14, r15}
		"xor.b32 r25, r25, r20;\n\t"
		"xor.b32 r26, r26, r21;\n\t"
		"shf.r.wrap.b32 r24, r25, r26, 18;\n\t"
		"shf.r.wrap.b32 r25, r26, r25, 18;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r20, r21}    D = {r14, r15}
		"lop3.b32 r26, r4, r24, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r25, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r24, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r25, r21, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r25, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r24, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r25, r21, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r25, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r24, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r25, r21, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r4, r24, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r25, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r24, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r25, r21, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r20, r21}    D = {r14, r15}
		/*
		* |------------------------[ROUND 9.3]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r30, r48}           |
		* |            v[ 6]            |           {r24, r25}           |
		* |            v[ 7]            |           {r27,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r11}           |
		* |            v[13]            |           {r12, r13}           |
		* |            v[14]            |           {r14, r15}           |
		* |            v[15]            |           { r9, r28}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r28}
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r8;\n\t"
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r28}
		"xor.b32 r26, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r6, r26, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r28}
		"xor.b32 r9, r9, r6;\n\t"
		"xor.b32 r28, r28, r7;\n\t"
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r28}
		"shf.r.wrap.b32 r26, r9, r28, 60;\n\t"
		"shf.r.wrap.b32 r9, r28, r9, 60;\n\t"
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r26}
		"add.cc.u32 r22, r22, r9;\n\t"
		"addc.u32 r23, r23, r26;\n\t"
		// A = {r6, r7}    B = {r27, r8}    C = {r22, r23}    D = {r9, r26}
		"xor.b32 r27, r27, r22;\n\t"
		"xor.b32 r8, r8, r23;\n\t"
		"shf.r.wrap.b32 r28, r27, r8, 43;\n\t"
		"shf.r.wrap.b32 r27, r8, r27, 43;\n\t"
		// A = {r6, r7}    B = {r27, r28}    C = {r22, r23}    D = {r9, r26}
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r28;\n\t"
		// A = {r6, r7}    B = {r27, r28}    C = {r22, r23}    D = {r9, r26}
		"xor.b32 r8, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r27, r28}    C = {r22, r23}    D = {r9, r26}
		"xor.b32 r9, r9, r6;\n\t"
		"xor.b32 r26, r26, r7;\n\t"
		"shf.r.wrap.b32 r8, r9, r26, 5;\n\t"
		"shf.r.wrap.b32 r9, r26, r9, 5;\n\t"
		// A = {r6, r7}    B = {r27, r28}    C = {r22, r23}    D = {r8, r9}
		"add.cc.u32 r22, r22, r8;\n\t"
		"addc.u32 r23, r23, r9;\n\t"
		// A = {r6, r7}    B = {r27, r28}    C = {r22, r23}    D = {r8, r9}
		"xor.b32 r27, r27, r22;\n\t"
		"xor.b32 r28, r28, r23;\n\t"
		"shf.r.wrap.b32 r26, r27, r28, 18;\n\t"
		"shf.r.wrap.b32 r27, r28, r27, 18;\n\t"
		// A = {r6, r7}    B = {r26, r27}    C = {r22, r23}    D = {r8, r9}
		"lop3.b32 r28, r6, r26, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r27, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r26, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r27, r23, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r27, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r26, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r27, r23, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r27, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r26, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r27, r23, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r6, r26, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r27, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r26, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r27, r23, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r26, r27}    C = {r22, r23}    D = {r8, r9}
		/*
		* |------------------------[ROUND 9.4]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r30, r48}           |
		* |            v[ 6]            |           {r24, r25}           |
		* |            v[ 7]            |           {r26, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r11}           |
		* |            v[13]            |           {r12, r13}           |
		* |            v[14]            |           {r14, r15}           |
		* |            v[15]            |           { r8,  r9}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r9}
		"add.cc.u32 r0, r0, r30;\n\t"
		"addc.u32 r1, r1, r48;\n\t"
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r9}
		"xor.b32 r28, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r0, r28, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r9}
		"xor.b32 r8, r8, r0;\n\t"
		"xor.b32 r9, r9, r1;\n\t"
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r9}
		"shf.r.wrap.b32 r28, r8, r9, 60;\n\t"
		"shf.r.wrap.b32 r8, r9, r8, 60;\n\t"
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r28}
		"add.cc.u32 r20, r20, r8;\n\t"
		"addc.u32 r21, r21, r28;\n\t"
		// A = {r0, r1}    B = {r30, r48}    C = {r20, r21}    D = {r8, r28}
		"xor.b32 r30, r30, r20;\n\t"
		"xor.b32 r48, r48, r21;\n\t"
		"shf.r.wrap.b32 r9, r30, r48, 43;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 43;\n\t"
		// A = {r0, r1}    B = {r30, r9}    C = {r20, r21}    D = {r8, r28}
		"add.cc.u32 r0, r0, r30;\n\t"
		"addc.u32 r1, r1, r9;\n\t"
		// A = {r0, r1}    B = {r30, r9}    C = {r20, r21}    D = {r8, r28}
		"xor.b32 r48, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r30, r9}    C = {r20, r21}    D = {r8, r28}
		"xor.b32 r8, r8, r0;\n\t"
		"xor.b32 r28, r28, r1;\n\t"
		"shf.r.wrap.b32 r48, r8, r28, 5;\n\t"
		"shf.r.wrap.b32 r8, r28, r8, 5;\n\t"
		// A = {r0, r1}    B = {r30, r9}    C = {r20, r21}    D = {r48, r8}
		"add.cc.u32 r20, r20, r48;\n\t"
		"addc.u32 r21, r21, r8;\n\t"
		// A = {r0, r1}    B = {r30, r9}    C = {r20, r21}    D = {r48, r8}
		"xor.b32 r30, r30, r20;\n\t"
		"xor.b32 r9, r9, r21;\n\t"
		"shf.r.wrap.b32 r28, r30, r9, 18;\n\t"
		"shf.r.wrap.b32 r30, r9, r30, 18;\n\t"
		// A = {r0, r1}    B = {r28, r30}    C = {r20, r21}    D = {r48, r8}
		"lop3.b32 r9, r0, r28, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r30, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r28, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r30, r21, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r30, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r28, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r30, r21, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r30, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r28, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r30, r21, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r0, r28, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r30, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r28, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r30, r21, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r28, r30}    C = {r20, r21}    D = {r48, r8}
		/*
		* |------------------------[ROUND 9.5]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r28, r30}           |
		* |            v[ 6]            |           {r24, r25}           |
		* |            v[ 7]            |           {r26, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r10, r11}           |
		* |            v[13]            |           {r12, r13}           |
		* |            v[14]            |           {r14, r15}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r11}
		"add.cc.u32 r2, r2, r24;\n\t"
		"addc.u32 r3, r3, r25;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r11}
		"xor.b32 r9, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r2, r9, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r11}
		"xor.b32 r10, r10, r2;\n\t"
		"xor.b32 r11, r11, r3;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r11}
		"shf.r.wrap.b32 r9, r10, r11, 60;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 60;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r9}
		"add.cc.u32 r22, r22, r10;\n\t"
		"addc.u32 r23, r23, r9;\n\t"
		// A = {r2, r3}    B = {r24, r25}    C = {r22, r23}    D = {r10, r9}
		"xor.b32 r24, r24, r22;\n\t"
		"xor.b32 r25, r25, r23;\n\t"
		"shf.r.wrap.b32 r11, r24, r25, 43;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 43;\n\t"
		// A = {r2, r3}    B = {r24, r11}    C = {r22, r23}    D = {r10, r9}
		"add.cc.u32 r2, r2, r24;\n\t"
		"addc.u32 r3, r3, r11;\n\t"
		// A = {r2, r3}    B = {r24, r11}    C = {r22, r23}    D = {r10, r9}
		"xor.b32 r25, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r2, r2, r25;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r24, r11}    C = {r22, r23}    D = {r10, r9}
		"xor.b32 r10, r10, r2;\n\t"
		"xor.b32 r9, r9, r3;\n\t"
		"shf.r.wrap.b32 r25, r10, r9, 5;\n\t"
		"shf.r.wrap.b32 r10, r9, r10, 5;\n\t"
		// A = {r2, r3}    B = {r24, r11}    C = {r22, r23}    D = {r25, r10}
		"add.cc.u32 r22, r22, r25;\n\t"
		"addc.u32 r23, r23, r10;\n\t"
		// A = {r2, r3}    B = {r24, r11}    C = {r22, r23}    D = {r25, r10}
		"xor.b32 r24, r24, r22;\n\t"
		"xor.b32 r11, r11, r23;\n\t"
		"shf.r.wrap.b32 r9, r24, r11, 18;\n\t"
		"shf.r.wrap.b32 r24, r11, r24, 18;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r22, r23}    D = {r25, r10}
		"lop3.b32 r11, r2, r9, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r24, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r9, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r24, r23, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r24, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r9, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r24, r23, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r24, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r9, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r24, r23, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r2, r9, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r24, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r9, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r24, r23, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r9, r24}    C = {r22, r23}    D = {r25, r10}
		/*
		* |------------------------[ROUND 9.6]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r28, r30}           |
		* |            v[ 6]            |           { r9, r24}           |
		* |            v[ 7]            |           {r26, r27}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r25, r10}           |
		* |            v[13]            |           {r12, r13}           |
		* |            v[14]            |           {r14, r15}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r13}
		"add.cc.u32 r4, r4, r26;\n\t"
		"addc.u32 r5, r5, r27;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r13}
		"xor.b32 r11, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r11, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r13}
		"xor.b32 r12, r12, r4;\n\t"
		"xor.b32 r13, r13, r5;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r13}
		"shf.r.wrap.b32 r11, r12, r13, 60;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 60;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r11}
		"add.cc.u32 r16, r16, r12;\n\t"
		"addc.u32 r17, r17, r11;\n\t"
		// A = {r4, r5}    B = {r26, r27}    C = {r16, r17}    D = {r12, r11}
		"xor.b32 r26, r26, r16;\n\t"
		"xor.b32 r27, r27, r17;\n\t"
		"shf.r.wrap.b32 r13, r26, r27, 43;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 43;\n\t"
		// A = {r4, r5}    B = {r26, r13}    C = {r16, r17}    D = {r12, r11}
		"add.cc.u32 r4, r4, r26;\n\t"
		"addc.u32 r5, r5, r13;\n\t"
		// A = {r4, r5}    B = {r26, r13}    C = {r16, r17}    D = {r12, r11}
		"xor.b32 r27, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r4, r4, r27;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r26, r13}    C = {r16, r17}    D = {r12, r11}
		"xor.b32 r12, r12, r4;\n\t"
		"xor.b32 r11, r11, r5;\n\t"
		"shf.r.wrap.b32 r27, r12, r11, 5;\n\t"
		"shf.r.wrap.b32 r12, r11, r12, 5;\n\t"
		// A = {r4, r5}    B = {r26, r13}    C = {r16, r17}    D = {r27, r12}
		"add.cc.u32 r16, r16, r27;\n\t"
		"addc.u32 r17, r17, r12;\n\t"
		// A = {r4, r5}    B = {r26, r13}    C = {r16, r17}    D = {r27, r12}
		"xor.b32 r26, r26, r16;\n\t"
		"xor.b32 r13, r13, r17;\n\t"
		"shf.r.wrap.b32 r11, r26, r13, 18;\n\t"
		"shf.r.wrap.b32 r26, r13, r26, 18;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r16, r17}    D = {r27, r12}
		"lop3.b32 r13, r4, r11, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r26, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r11, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r26, r17, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r26, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r11, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r26, r17, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r26, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r11, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r26, r17, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r4, r11, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r26, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r11, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r26, r17, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r11, r26}    C = {r16, r17}    D = {r27, r12}
		/*
		* |------------------------[ROUND 9.7]---------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r31, r29}           |
		* |            v[ 5]            |           {r28, r30}           |
		* |            v[ 6]            |           { r9, r24}           |
		* |            v[ 7]            |           {r11, r26}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r25, r10}           |
		* |            v[13]            |           {r27, r12}           |
		* |            v[14]            |           {r14, r15}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r15}
		"add.cc.u32 r6, r6, r31;\n\t"
		"addc.u32 r7, r7, r29;\n\t"
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r15}
		"xor.b32 r13, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r6, r13, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r15}
		"xor.b32 r14, r14, r6;\n\t"
		"xor.b32 r15, r15, r7;\n\t"
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r15}
		"shf.r.wrap.b32 r13, r14, r15, 60;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 60;\n\t"
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r13}
		"add.cc.u32 r18, r18, r14;\n\t"
		"addc.u32 r19, r19, r13;\n\t"
		// A = {r6, r7}    B = {r31, r29}    C = {r18, r19}    D = {r14, r13}
		"xor.b32 r31, r31, r18;\n\t"
		"xor.b32 r29, r29, r19;\n\t"
		"shf.r.wrap.b32 r15, r31, r29, 43;\n\t"
		"shf.r.wrap.b32 r31, r29, r31, 43;\n\t"
		// A = {r6, r7}    B = {r31, r15}    C = {r18, r19}    D = {r14, r13}
		"add.cc.u32 r6, r6, r31;\n\t"
		"addc.u32 r7, r7, r15;\n\t"
		// A = {r6, r7}    B = {r31, r15}    C = {r18, r19}    D = {r14, r13}
		"xor.b32 r29, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r31, r15}    C = {r18, r19}    D = {r14, r13}
		"xor.b32 r14, r14, r6;\n\t"
		"xor.b32 r13, r13, r7;\n\t"
		"shf.r.wrap.b32 r29, r14, r13, 5;\n\t"
		"shf.r.wrap.b32 r14, r13, r14, 5;\n\t"
		// A = {r6, r7}    B = {r31, r15}    C = {r18, r19}    D = {r29, r14}
		"add.cc.u32 r18, r18, r29;\n\t"
		"addc.u32 r19, r19, r14;\n\t"
		// A = {r6, r7}    B = {r31, r15}    C = {r18, r19}    D = {r29, r14}
		"xor.b32 r31, r31, r18;\n\t"
		"xor.b32 r15, r15, r19;\n\t"
		"shf.r.wrap.b32 r13, r31, r15, 18;\n\t"
		"shf.r.wrap.b32 r31, r15, r31, 18;\n\t"
		// A = {r6, r7}    B = {r13, r31}    C = {r18, r19}    D = {r29, r14}
		"lop3.b32 r15, r6, r13, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r31, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r13, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r31, r19, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r31, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r13, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r31, r19, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r31, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r13, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r31, r19, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r6, r13, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r31, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r13, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r31, r19, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r13, r31}    C = {r18, r19}    D = {r29, r14}
		/*
		* |------------------------[ROUND 10.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r31}           |
		* |            v[ 5]            |           {r28, r30}           |
		* |            v[ 6]            |           { r9, r24}           |
		* |            v[ 7]            |           {r11, r26}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r25, r10}           |
		* |            v[13]            |           {r27, r12}           |
		* |            v[14]            |           {r29, r14}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r10}
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r31;\n\t"
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r10}
		"xor.b32 r15, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r15, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r10}
		"xor.b32 r25, r25, r0;\n\t"
		"xor.b32 r10, r10, r1;\n\t"
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r10}
		"shf.r.wrap.b32 r15, r25, r10, 60;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 60;\n\t"
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r15}
		"add.cc.u32 r16, r16, r25;\n\t"
		"addc.u32 r17, r17, r15;\n\t"
		// A = {r0, r1}    B = {r13, r31}    C = {r16, r17}    D = {r25, r15}
		"xor.b32 r13, r13, r16;\n\t"
		"xor.b32 r31, r31, r17;\n\t"
		"shf.r.wrap.b32 r10, r13, r31, 43;\n\t"
		"shf.r.wrap.b32 r13, r31, r13, 43;\n\t"
		// A = {r0, r1}    B = {r13, r10}    C = {r16, r17}    D = {r25, r15}
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r10;\n\t"
		// A = {r0, r1}    B = {r13, r10}    C = {r16, r17}    D = {r25, r15}
		"xor.b32 r31, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r13, r10}    C = {r16, r17}    D = {r25, r15}
		"xor.b32 r25, r25, r0;\n\t"
		"xor.b32 r15, r15, r1;\n\t"
		"shf.r.wrap.b32 r31, r25, r15, 5;\n\t"
		"shf.r.wrap.b32 r25, r15, r25, 5;\n\t"
		// A = {r0, r1}    B = {r13, r10}    C = {r16, r17}    D = {r31, r25}
		"add.cc.u32 r16, r16, r31;\n\t"
		"addc.u32 r17, r17, r25;\n\t"
		// A = {r0, r1}    B = {r13, r10}    C = {r16, r17}    D = {r31, r25}
		"xor.b32 r13, r13, r16;\n\t"
		"xor.b32 r10, r10, r17;\n\t"
		"shf.r.wrap.b32 r15, r13, r10, 18;\n\t"
		"shf.r.wrap.b32 r13, r10, r13, 18;\n\t"
		// A = {r0, r1}    B = {r15, r13}    C = {r16, r17}    D = {r31, r25}
		"lop3.b32 r10, r0, r15, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r13, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r15, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r13, r17, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r13, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r15, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r13, r17, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r13, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r15, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r13, r17, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r0, r15, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r13, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r15, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r13, r17, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r15, r13}    C = {r16, r17}    D = {r31, r25}
		/*
		* |------------------------[ROUND 10.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           {r28, r30}           |
		* |            v[ 6]            |           { r9, r24}           |
		* |            v[ 7]            |           {r11, r26}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31, r25}           |
		* |            v[13]            |           {r27, r12}           |
		* |            v[14]            |           {r29, r14}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r12}
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r30;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r12}
		"xor.b32 r10, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r10, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r12}
		"xor.b32 r27, r27, r2;\n\t"
		"xor.b32 r12, r12, r3;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r12}
		"shf.r.wrap.b32 r10, r27, r12, 60;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 60;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r10}
		"add.cc.u32 r18, r18, r27;\n\t"
		"addc.u32 r19, r19, r10;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r18, r19}    D = {r27, r10}
		"xor.b32 r28, r28, r18;\n\t"
		"xor.b32 r30, r30, r19;\n\t"
		"shf.r.wrap.b32 r12, r28, r30, 43;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 43;\n\t"
		// A = {r2, r3}    B = {r28, r12}    C = {r18, r19}    D = {r27, r10}
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r12;\n\t"
		// A = {r2, r3}    B = {r28, r12}    C = {r18, r19}    D = {r27, r10}
		"xor.b32 r30, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r2, r2, r30;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r28, r12}    C = {r18, r19}    D = {r27, r10}
		"xor.b32 r27, r27, r2;\n\t"
		"xor.b32 r10, r10, r3;\n\t"
		"shf.r.wrap.b32 r30, r27, r10, 5;\n\t"
		"shf.r.wrap.b32 r27, r10, r27, 5;\n\t"
		// A = {r2, r3}    B = {r28, r12}    C = {r18, r19}    D = {r30, r27}
		"add.cc.u32 r18, r18, r30;\n\t"
		"addc.u32 r19, r19, r27;\n\t"
		// A = {r2, r3}    B = {r28, r12}    C = {r18, r19}    D = {r30, r27}
		"xor.b32 r28, r28, r18;\n\t"
		"xor.b32 r12, r12, r19;\n\t"
		"shf.r.wrap.b32 r10, r28, r12, 18;\n\t"
		"shf.r.wrap.b32 r28, r12, r28, 18;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r18, r19}    D = {r30, r27}
		"lop3.b32 r12, r2, r10, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r28, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r10, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r28, r19, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r28, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r10, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r28, r19, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r28, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r10, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r28, r19, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r2, r10, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r28, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r10, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r28, r19, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r18, r19}    D = {r30, r27}
		/*
		* |------------------------[ROUND 10.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           {r10, r28}           |
		* |            v[ 6]            |           { r9, r24}           |
		* |            v[ 7]            |           {r11, r26}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31, r25}           |
		* |            v[13]            |           {r30, r27}           |
		* |            v[14]            |           {r29, r14}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r14}
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r24;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r14}
		"xor.b32 r12, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r4, r12, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r14}
		"xor.b32 r29, r29, r4;\n\t"
		"xor.b32 r14, r14, r5;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r14}
		"shf.r.wrap.b32 r12, r29, r14, 60;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 60;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r12}
		"add.cc.u32 r20, r20, r29;\n\t"
		"addc.u32 r21, r21, r12;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r20, r21}    D = {r29, r12}
		"xor.b32 r9, r9, r20;\n\t"
		"xor.b32 r24, r24, r21;\n\t"
		"shf.r.wrap.b32 r14, r9, r24, 43;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 43;\n\t"
		// A = {r4, r5}    B = {r9, r14}    C = {r20, r21}    D = {r29, r12}
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r14;\n\t"
		// A = {r4, r5}    B = {r9, r14}    C = {r20, r21}    D = {r29, r12}
		"xor.b32 r24, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r4, r24;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r9, r14}    C = {r20, r21}    D = {r29, r12}
		"xor.b32 r29, r29, r4;\n\t"
		"xor.b32 r12, r12, r5;\n\t"
		"shf.r.wrap.b32 r24, r29, r12, 5;\n\t"
		"shf.r.wrap.b32 r29, r12, r29, 5;\n\t"
		// A = {r4, r5}    B = {r9, r14}    C = {r20, r21}    D = {r24, r29}
		"add.cc.u32 r20, r20, r24;\n\t"
		"addc.u32 r21, r21, r29;\n\t"
		// A = {r4, r5}    B = {r9, r14}    C = {r20, r21}    D = {r24, r29}
		"xor.b32 r9, r9, r20;\n\t"
		"xor.b32 r14, r14, r21;\n\t"
		"shf.r.wrap.b32 r12, r9, r14, 18;\n\t"
		"shf.r.wrap.b32 r9, r14, r9, 18;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r20, r21}    D = {r24, r29}
		"lop3.b32 r14, r4, r12, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r9, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r12, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r9, r21, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r9, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r12, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r9, r21, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r9, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r12, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r9, r21, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r4, r12, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r9, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r12, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r9, r21, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r20, r21}    D = {r24, r29}
		/*
		* |------------------------[ROUND 10.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           {r10, r28}           |
		* |            v[ 6]            |           {r12,  r9}           |
		* |            v[ 7]            |           {r11, r26}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31, r25}           |
		* |            v[13]            |           {r30, r27}           |
		* |            v[14]            |           {r24, r29}           |
		* |            v[15]            |           {r48,  r8}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r8}
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r26;\n\t"
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r8}
		"xor.b32 r14, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r6, r14, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r8}
		"xor.b32 r48, r48, r6;\n\t"
		"xor.b32 r8, r8, r7;\n\t"
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r8}
		"shf.r.wrap.b32 r14, r48, r8, 60;\n\t"
		"shf.r.wrap.b32 r48, r8, r48, 60;\n\t"
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r14}
		"add.cc.u32 r22, r22, r48;\n\t"
		"addc.u32 r23, r23, r14;\n\t"
		// A = {r6, r7}    B = {r11, r26}    C = {r22, r23}    D = {r48, r14}
		"xor.b32 r11, r11, r22;\n\t"
		"xor.b32 r26, r26, r23;\n\t"
		"shf.r.wrap.b32 r8, r11, r26, 43;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 43;\n\t"
		// A = {r6, r7}    B = {r11, r8}    C = {r22, r23}    D = {r48, r14}
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r8;\n\t"
		// A = {r6, r7}    B = {r11, r8}    C = {r22, r23}    D = {r48, r14}
		"xor.b32 r26, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r6, r6, r26;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r11, r8}    C = {r22, r23}    D = {r48, r14}
		"xor.b32 r48, r48, r6;\n\t"
		"xor.b32 r14, r14, r7;\n\t"
		"shf.r.wrap.b32 r26, r48, r14, 5;\n\t"
		"shf.r.wrap.b32 r48, r14, r48, 5;\n\t"
		// A = {r6, r7}    B = {r11, r8}    C = {r22, r23}    D = {r26, r48}
		"add.cc.u32 r22, r22, r26;\n\t"
		"addc.u32 r23, r23, r48;\n\t"
		// A = {r6, r7}    B = {r11, r8}    C = {r22, r23}    D = {r26, r48}
		"xor.b32 r11, r11, r22;\n\t"
		"xor.b32 r8, r8, r23;\n\t"
		"shf.r.wrap.b32 r14, r11, r8, 18;\n\t"
		"shf.r.wrap.b32 r11, r8, r11, 18;\n\t"
		// A = {r6, r7}    B = {r14, r11}    C = {r22, r23}    D = {r26, r48}
		"lop3.b32 r8, r6, r14, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r11, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r14, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r11, r23, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r11, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r14, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r11, r23, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r11, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r14, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r11, r23, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r6, r14, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r11, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r14, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r11, r23, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r14, r11}    C = {r22, r23}    D = {r26, r48}
		/*
		* |------------------------[ROUND 10.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           {r10, r28}           |
		* |            v[ 6]            |           {r12,  r9}           |
		* |            v[ 7]            |           {r14, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31, r25}           |
		* |            v[13]            |           {r30, r27}           |
		* |            v[14]            |           {r24, r29}           |
		* |            v[15]            |           {r26, r48}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r48}
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r28;\n\t"
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r48}
		"xor.b32 r8, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r8, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r48}
		"xor.b32 r26, r26, r0;\n\t"
		"xor.b32 r48, r48, r1;\n\t"
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r48}
		"shf.r.wrap.b32 r8, r26, r48, 60;\n\t"
		"shf.r.wrap.b32 r26, r48, r26, 60;\n\t"
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r8}
		"add.cc.u32 r20, r20, r26;\n\t"
		"addc.u32 r21, r21, r8;\n\t"
		// A = {r0, r1}    B = {r10, r28}    C = {r20, r21}    D = {r26, r8}
		"xor.b32 r10, r10, r20;\n\t"
		"xor.b32 r28, r28, r21;\n\t"
		"shf.r.wrap.b32 r48, r10, r28, 43;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 43;\n\t"
		// A = {r0, r1}    B = {r10, r48}    C = {r20, r21}    D = {r26, r8}
		"add.cc.u32 r0, r0, r10;\n\t"
		"addc.u32 r1, r1, r48;\n\t"
		// A = {r0, r1}    B = {r10, r48}    C = {r20, r21}    D = {r26, r8}
		"xor.b32 r28, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r0, r0, r28;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r10, r48}    C = {r20, r21}    D = {r26, r8}
		"xor.b32 r26, r26, r0;\n\t"
		"xor.b32 r8, r8, r1;\n\t"
		"shf.r.wrap.b32 r28, r26, r8, 5;\n\t"
		"shf.r.wrap.b32 r26, r8, r26, 5;\n\t"
		// A = {r0, r1}    B = {r10, r48}    C = {r20, r21}    D = {r28, r26}
		"add.cc.u32 r20, r20, r28;\n\t"
		"addc.u32 r21, r21, r26;\n\t"
		// A = {r0, r1}    B = {r10, r48}    C = {r20, r21}    D = {r28, r26}
		"xor.b32 r10, r10, r20;\n\t"
		"xor.b32 r48, r48, r21;\n\t"
		"shf.r.wrap.b32 r8, r10, r48, 18;\n\t"
		"shf.r.wrap.b32 r10, r48, r10, 18;\n\t"
		// A = {r0, r1}    B = {r8, r10}    C = {r20, r21}    D = {r28, r26}
		"lop3.b32 r48, r0, r8, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r10, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r8, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r10, r21, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r10, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r8, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r10, r21, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r10, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r8, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r10, r21, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r0, r8, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r10, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r8, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r10, r21, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r8, r10}    C = {r20, r21}    D = {r28, r26}
		/*
		* |------------------------[ROUND 10.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           { r8, r10}           |
		* |            v[ 6]            |           {r12,  r9}           |
		* |            v[ 7]            |           {r14, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r31, r25}           |
		* |            v[13]            |           {r30, r27}           |
		* |            v[14]            |           {r24, r29}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r25}
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r9;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r25}
		"xor.b32 r48, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r2, r48, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r25}
		"xor.b32 r31, r31, r2;\n\t"
		"xor.b32 r25, r25, r3;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r25}
		"shf.r.wrap.b32 r48, r31, r25, 60;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 60;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r48}
		"add.cc.u32 r22, r22, r31;\n\t"
		"addc.u32 r23, r23, r48;\n\t"
		// A = {r2, r3}    B = {r12, r9}    C = {r22, r23}    D = {r31, r48}
		"xor.b32 r12, r12, r22;\n\t"
		"xor.b32 r9, r9, r23;\n\t"
		"shf.r.wrap.b32 r25, r12, r9, 43;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 43;\n\t"
		// A = {r2, r3}    B = {r12, r25}    C = {r22, r23}    D = {r31, r48}
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r25;\n\t"
		// A = {r2, r3}    B = {r12, r25}    C = {r22, r23}    D = {r31, r48}
		"xor.b32 r9, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r2, r9;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r12, r25}    C = {r22, r23}    D = {r31, r48}
		"xor.b32 r31, r31, r2;\n\t"
		"xor.b32 r48, r48, r3;\n\t"
		"shf.r.wrap.b32 r9, r31, r48, 5;\n\t"
		"shf.r.wrap.b32 r31, r48, r31, 5;\n\t"
		// A = {r2, r3}    B = {r12, r25}    C = {r22, r23}    D = {r9, r31}
		"add.cc.u32 r22, r22, r9;\n\t"
		"addc.u32 r23, r23, r31;\n\t"
		// A = {r2, r3}    B = {r12, r25}    C = {r22, r23}    D = {r9, r31}
		"xor.b32 r12, r12, r22;\n\t"
		"xor.b32 r25, r25, r23;\n\t"
		"shf.r.wrap.b32 r48, r12, r25, 18;\n\t"
		"shf.r.wrap.b32 r12, r25, r12, 18;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r22, r23}    D = {r9, r31}
		"lop3.b32 r25, r2, r48, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r12, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r48, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r12, r23, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r12, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r48, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r12, r23, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r12, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r48, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r12, r23, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r2, r48, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r12, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r48, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r12, r23, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r48, r12}    C = {r22, r23}    D = {r9, r31}
		/*
		* |------------------------[ROUND 10.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           { r8, r10}           |
		* |            v[ 6]            |           {r48, r12}           |
		* |            v[ 7]            |           {r14, r11}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r31}           |
		* |            v[13]            |           {r30, r27}           |
		* |            v[14]            |           {r24, r29}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r27}
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r11;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r27}
		"xor.b32 r25, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r4, r25, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r27}
		"xor.b32 r30, r30, r4;\n\t"
		"xor.b32 r27, r27, r5;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r27}
		"shf.r.wrap.b32 r25, r30, r27, 60;\n\t"
		"shf.r.wrap.b32 r30, r27, r30, 60;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r25}
		"add.cc.u32 r16, r16, r30;\n\t"
		"addc.u32 r17, r17, r25;\n\t"
		// A = {r4, r5}    B = {r14, r11}    C = {r16, r17}    D = {r30, r25}
		"xor.b32 r14, r14, r16;\n\t"
		"xor.b32 r11, r11, r17;\n\t"
		"shf.r.wrap.b32 r27, r14, r11, 43;\n\t"
		"shf.r.wrap.b32 r14, r11, r14, 43;\n\t"
		// A = {r4, r5}    B = {r14, r27}    C = {r16, r17}    D = {r30, r25}
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r27;\n\t"
		// A = {r4, r5}    B = {r14, r27}    C = {r16, r17}    D = {r30, r25}
		"xor.b32 r11, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r4, r11;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r14, r27}    C = {r16, r17}    D = {r30, r25}
		"xor.b32 r30, r30, r4;\n\t"
		"xor.b32 r25, r25, r5;\n\t"
		"shf.r.wrap.b32 r11, r30, r25, 5;\n\t"
		"shf.r.wrap.b32 r30, r25, r30, 5;\n\t"
		// A = {r4, r5}    B = {r14, r27}    C = {r16, r17}    D = {r11, r30}
		"add.cc.u32 r16, r16, r11;\n\t"
		"addc.u32 r17, r17, r30;\n\t"
		// A = {r4, r5}    B = {r14, r27}    C = {r16, r17}    D = {r11, r30}
		"xor.b32 r14, r14, r16;\n\t"
		"xor.b32 r27, r27, r17;\n\t"
		"shf.r.wrap.b32 r25, r14, r27, 18;\n\t"
		"shf.r.wrap.b32 r14, r27, r14, 18;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r16, r17}    D = {r11, r30}
		"lop3.b32 r27, r4, r25, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r14, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r25, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r14, r17, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r14, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r25, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r14, r17, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r14, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r25, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r14, r17, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r4, r25, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r14, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r25, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r14, r17, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r25, r14}    C = {r16, r17}    D = {r11, r30}
		/*
		* |------------------------[ROUND 10.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r15, r13}           |
		* |            v[ 5]            |           { r8, r10}           |
		* |            v[ 6]            |           {r48, r12}           |
		* |            v[ 7]            |           {r25, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r31}           |
		* |            v[13]            |           {r11, r30}           |
		* |            v[14]            |           {r24, r29}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r29}
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r13;\n\t"
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r29}
		"xor.b32 r27, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r27, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r29}
		"xor.b32 r24, r24, r6;\n\t"
		"xor.b32 r29, r29, r7;\n\t"
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r29}
		"shf.r.wrap.b32 r27, r24, r29, 60;\n\t"
		"shf.r.wrap.b32 r24, r29, r24, 60;\n\t"
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r27}
		"add.cc.u32 r18, r18, r24;\n\t"
		"addc.u32 r19, r19, r27;\n\t"
		// A = {r6, r7}    B = {r15, r13}    C = {r18, r19}    D = {r24, r27}
		"xor.b32 r15, r15, r18;\n\t"
		"xor.b32 r13, r13, r19;\n\t"
		"shf.r.wrap.b32 r29, r15, r13, 43;\n\t"
		"shf.r.wrap.b32 r15, r13, r15, 43;\n\t"
		// A = {r6, r7}    B = {r15, r29}    C = {r18, r19}    D = {r24, r27}
		"add.cc.u32 r6, r6, r15;\n\t"
		"addc.u32 r7, r7, r29;\n\t"
		// A = {r6, r7}    B = {r15, r29}    C = {r18, r19}    D = {r24, r27}
		"xor.b32 r13, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r15, r29}    C = {r18, r19}    D = {r24, r27}
		"xor.b32 r24, r24, r6;\n\t"
		"xor.b32 r27, r27, r7;\n\t"
		"shf.r.wrap.b32 r13, r24, r27, 5;\n\t"
		"shf.r.wrap.b32 r24, r27, r24, 5;\n\t"
		// A = {r6, r7}    B = {r15, r29}    C = {r18, r19}    D = {r13, r24}
		"add.cc.u32 r18, r18, r13;\n\t"
		"addc.u32 r19, r19, r24;\n\t"
		// A = {r6, r7}    B = {r15, r29}    C = {r18, r19}    D = {r13, r24}
		"xor.b32 r15, r15, r18;\n\t"
		"xor.b32 r29, r29, r19;\n\t"
		"shf.r.wrap.b32 r27, r15, r29, 18;\n\t"
		"shf.r.wrap.b32 r15, r29, r15, 18;\n\t"
		// A = {r6, r7}    B = {r27, r15}    C = {r18, r19}    D = {r13, r24}
		"lop3.b32 r29, r6, r27, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r15, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r27, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r15, r19, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r15, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r27, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r15, r19, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r15, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r27, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r15, r19, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r6, r27, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r15, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r27, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r15, r19, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r27, r15}    C = {r18, r19}    D = {r13, r24}
		/*
		* |------------------------[ROUND 11.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r15}           |
		* |            v[ 5]            |           { r8, r10}           |
		* |            v[ 6]            |           {r48, r12}           |
		* |            v[ 7]            |           {r25, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r9, r31}           |
		* |            v[13]            |           {r11, r30}           |
		* |            v[14]            |           {r13, r24}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r31}
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r15;\n\t"
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r31}
		"xor.b32 r29, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r0, r29, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r31}
		"xor.b32 r9, r9, r0;\n\t"
		"xor.b32 r31, r31, r1;\n\t"
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r31}
		"shf.r.wrap.b32 r29, r9, r31, 60;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 60;\n\t"
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r29}
		"add.cc.u32 r16, r16, r9;\n\t"
		"addc.u32 r17, r17, r29;\n\t"
		// A = {r0, r1}    B = {r27, r15}    C = {r16, r17}    D = {r9, r29}
		"xor.b32 r27, r27, r16;\n\t"
		"xor.b32 r15, r15, r17;\n\t"
		"shf.r.wrap.b32 r31, r27, r15, 43;\n\t"
		"shf.r.wrap.b32 r27, r15, r27, 43;\n\t"
		// A = {r0, r1}    B = {r27, r31}    C = {r16, r17}    D = {r9, r29}
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r31;\n\t"
		// A = {r0, r1}    B = {r27, r31}    C = {r16, r17}    D = {r9, r29}
		"xor.b32 r15, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r27, r31}    C = {r16, r17}    D = {r9, r29}
		"xor.b32 r9, r9, r0;\n\t"
		"xor.b32 r29, r29, r1;\n\t"
		"shf.r.wrap.b32 r15, r9, r29, 5;\n\t"
		"shf.r.wrap.b32 r9, r29, r9, 5;\n\t"
		// A = {r0, r1}    B = {r27, r31}    C = {r16, r17}    D = {r15, r9}
		"add.cc.u32 r16, r16, r15;\n\t"
		"addc.u32 r17, r17, r9;\n\t"
		// A = {r0, r1}    B = {r27, r31}    C = {r16, r17}    D = {r15, r9}
		"xor.b32 r27, r27, r16;\n\t"
		"xor.b32 r31, r31, r17;\n\t"
		"shf.r.wrap.b32 r29, r27, r31, 18;\n\t"
		"shf.r.wrap.b32 r27, r31, r27, 18;\n\t"
		// A = {r0, r1}    B = {r29, r27}    C = {r16, r17}    D = {r15, r9}
		"lop3.b32 r31, r0, r29, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r27, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r29, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r27, r17, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r27, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r29, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r27, r17, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r27, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r29, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r27, r17, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r0, r29, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r27, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r29, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r27, r17, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r29, r27}    C = {r16, r17}    D = {r15, r9}
		/*
		* |------------------------[ROUND 11.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           { r8, r10}           |
		* |            v[ 6]            |           {r48, r12}           |
		* |            v[ 7]            |           {r25, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15,  r9}           |
		* |            v[13]            |           {r11, r30}           |
		* |            v[14]            |           {r13, r24}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r30}
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r10;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r30}
		"xor.b32 r31, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r2, r31, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r30}
		"xor.b32 r11, r11, r2;\n\t"
		"xor.b32 r30, r30, r3;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r30}
		"shf.r.wrap.b32 r31, r11, r30, 60;\n\t"
		"shf.r.wrap.b32 r11, r30, r11, 60;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r31}
		"add.cc.u32 r18, r18, r11;\n\t"
		"addc.u32 r19, r19, r31;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r18, r19}    D = {r11, r31}
		"xor.b32 r8, r8, r18;\n\t"
		"xor.b32 r10, r10, r19;\n\t"
		"shf.r.wrap.b32 r30, r8, r10, 43;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 43;\n\t"
		// A = {r2, r3}    B = {r8, r30}    C = {r18, r19}    D = {r11, r31}
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r30;\n\t"
		// A = {r2, r3}    B = {r8, r30}    C = {r18, r19}    D = {r11, r31}
		"xor.b32 r10, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r8, r30}    C = {r18, r19}    D = {r11, r31}
		"xor.b32 r11, r11, r2;\n\t"
		"xor.b32 r31, r31, r3;\n\t"
		"shf.r.wrap.b32 r10, r11, r31, 5;\n\t"
		"shf.r.wrap.b32 r11, r31, r11, 5;\n\t"
		// A = {r2, r3}    B = {r8, r30}    C = {r18, r19}    D = {r10, r11}
		"add.cc.u32 r18, r18, r10;\n\t"
		"addc.u32 r19, r19, r11;\n\t"
		// A = {r2, r3}    B = {r8, r30}    C = {r18, r19}    D = {r10, r11}
		"xor.b32 r8, r8, r18;\n\t"
		"xor.b32 r30, r30, r19;\n\t"
		"shf.r.wrap.b32 r31, r8, r30, 18;\n\t"
		"shf.r.wrap.b32 r8, r30, r8, 18;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r18, r19}    D = {r10, r11}
		"lop3.b32 r30, r2, r31, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r8, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r31, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r8, r19, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r8, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r31, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r8, r19, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r8, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r31, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r8, r19, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r2, r31, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r8, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r31, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r8, r19, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r18, r19}    D = {r10, r11}
		/*
		* |------------------------[ROUND 11.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r31,  r8}           |
		* |            v[ 6]            |           {r48, r12}           |
		* |            v[ 7]            |           {r25, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15,  r9}           |
		* |            v[13]            |           {r10, r11}           |
		* |            v[14]            |           {r13, r24}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r24}
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r12;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r24}
		"xor.b32 r30, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r4, r30, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r24}
		"xor.b32 r13, r13, r4;\n\t"
		"xor.b32 r24, r24, r5;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r24}
		"shf.r.wrap.b32 r30, r13, r24, 60;\n\t"
		"shf.r.wrap.b32 r13, r24, r13, 60;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r30}
		"add.cc.u32 r20, r20, r13;\n\t"
		"addc.u32 r21, r21, r30;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r20, r21}    D = {r13, r30}
		"xor.b32 r48, r48, r20;\n\t"
		"xor.b32 r12, r12, r21;\n\t"
		"shf.r.wrap.b32 r24, r48, r12, 43;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 43;\n\t"
		// A = {r4, r5}    B = {r48, r24}    C = {r20, r21}    D = {r13, r30}
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r24;\n\t"
		// A = {r4, r5}    B = {r48, r24}    C = {r20, r21}    D = {r13, r30}
		"xor.b32 r12, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r48, r24}    C = {r20, r21}    D = {r13, r30}
		"xor.b32 r13, r13, r4;\n\t"
		"xor.b32 r30, r30, r5;\n\t"
		"shf.r.wrap.b32 r12, r13, r30, 5;\n\t"
		"shf.r.wrap.b32 r13, r30, r13, 5;\n\t"
		// A = {r4, r5}    B = {r48, r24}    C = {r20, r21}    D = {r12, r13}
		"add.cc.u32 r20, r20, r12;\n\t"
		"addc.u32 r21, r21, r13;\n\t"
		// A = {r4, r5}    B = {r48, r24}    C = {r20, r21}    D = {r12, r13}
		"xor.b32 r48, r48, r20;\n\t"
		"xor.b32 r24, r24, r21;\n\t"
		"shf.r.wrap.b32 r30, r48, r24, 18;\n\t"
		"shf.r.wrap.b32 r48, r24, r48, 18;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r20, r21}    D = {r12, r13}
		"lop3.b32 r24, r4, r30, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r48, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r30, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r48, r21, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r48, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r30, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r48, r21, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r48, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r30, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r48, r21, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r4, r30, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r48, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r30, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r48, r21, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r20, r21}    D = {r12, r13}
		/*
		* |------------------------[ROUND 11.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r31,  r8}           |
		* |            v[ 6]            |           {r30, r48}           |
		* |            v[ 7]            |           {r25, r14}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15,  r9}           |
		* |            v[13]            |           {r10, r11}           |
		* |            v[14]            |           {r12, r13}           |
		* |            v[15]            |           {r28, r26}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r26}
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r14;\n\t"
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r26}
		"xor.b32 r24, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r6, r24, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r26}
		"xor.b32 r28, r28, r6;\n\t"
		"xor.b32 r26, r26, r7;\n\t"
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r26}
		"shf.r.wrap.b32 r24, r28, r26, 60;\n\t"
		"shf.r.wrap.b32 r28, r26, r28, 60;\n\t"
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r24}
		"add.cc.u32 r22, r22, r28;\n\t"
		"addc.u32 r23, r23, r24;\n\t"
		// A = {r6, r7}    B = {r25, r14}    C = {r22, r23}    D = {r28, r24}
		"xor.b32 r25, r25, r22;\n\t"
		"xor.b32 r14, r14, r23;\n\t"
		"shf.r.wrap.b32 r26, r25, r14, 43;\n\t"
		"shf.r.wrap.b32 r25, r14, r25, 43;\n\t"
		// A = {r6, r7}    B = {r25, r26}    C = {r22, r23}    D = {r28, r24}
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r26;\n\t"
		// A = {r6, r7}    B = {r25, r26}    C = {r22, r23}    D = {r28, r24}
		"xor.b32 r14, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r6, r14;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r25, r26}    C = {r22, r23}    D = {r28, r24}
		"xor.b32 r28, r28, r6;\n\t"
		"xor.b32 r24, r24, r7;\n\t"
		"shf.r.wrap.b32 r14, r28, r24, 5;\n\t"
		"shf.r.wrap.b32 r28, r24, r28, 5;\n\t"
		// A = {r6, r7}    B = {r25, r26}    C = {r22, r23}    D = {r14, r28}
		"add.cc.u32 r22, r22, r14;\n\t"
		"addc.u32 r23, r23, r28;\n\t"
		// A = {r6, r7}    B = {r25, r26}    C = {r22, r23}    D = {r14, r28}
		"xor.b32 r25, r25, r22;\n\t"
		"xor.b32 r26, r26, r23;\n\t"
		"shf.r.wrap.b32 r24, r25, r26, 18;\n\t"
		"shf.r.wrap.b32 r25, r26, r25, 18;\n\t"
		// A = {r6, r7}    B = {r24, r25}    C = {r22, r23}    D = {r14, r28}
		"lop3.b32 r26, r6, r24, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r25, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r24, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r25, r23, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r25, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r24, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r25, r23, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r25, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r24, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r25, r23, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r6, r24, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r25, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r24, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r25, r23, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r24, r25}    C = {r22, r23}    D = {r14, r28}
		/*
		* |------------------------[ROUND 11.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r31,  r8}           |
		* |            v[ 6]            |           {r30, r48}           |
		* |            v[ 7]            |           {r24, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15,  r9}           |
		* |            v[13]            |           {r10, r11}           |
		* |            v[14]            |           {r12, r13}           |
		* |            v[15]            |           {r14, r28}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r28}
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r8;\n\t"
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r28}
		"xor.b32 r26, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r26, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r28}
		"xor.b32 r14, r14, r0;\n\t"
		"xor.b32 r28, r28, r1;\n\t"
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r28}
		"shf.r.wrap.b32 r26, r14, r28, 60;\n\t"
		"shf.r.wrap.b32 r14, r28, r14, 60;\n\t"
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r26}
		"add.cc.u32 r20, r20, r14;\n\t"
		"addc.u32 r21, r21, r26;\n\t"
		// A = {r0, r1}    B = {r31, r8}    C = {r20, r21}    D = {r14, r26}
		"xor.b32 r31, r31, r20;\n\t"
		"xor.b32 r8, r8, r21;\n\t"
		"shf.r.wrap.b32 r28, r31, r8, 43;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 43;\n\t"
		// A = {r0, r1}    B = {r31, r28}    C = {r20, r21}    D = {r14, r26}
		"add.cc.u32 r0, r0, r31;\n\t"
		"addc.u32 r1, r1, r28;\n\t"
		// A = {r0, r1}    B = {r31, r28}    C = {r20, r21}    D = {r14, r26}
		"xor.b32 r8, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r0, r8;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r31, r28}    C = {r20, r21}    D = {r14, r26}
		"xor.b32 r14, r14, r0;\n\t"
		"xor.b32 r26, r26, r1;\n\t"
		"shf.r.wrap.b32 r8, r14, r26, 5;\n\t"
		"shf.r.wrap.b32 r14, r26, r14, 5;\n\t"
		// A = {r0, r1}    B = {r31, r28}    C = {r20, r21}    D = {r8, r14}
		"add.cc.u32 r20, r20, r8;\n\t"
		"addc.u32 r21, r21, r14;\n\t"
		// A = {r0, r1}    B = {r31, r28}    C = {r20, r21}    D = {r8, r14}
		"xor.b32 r31, r31, r20;\n\t"
		"xor.b32 r28, r28, r21;\n\t"
		"shf.r.wrap.b32 r26, r31, r28, 18;\n\t"
		"shf.r.wrap.b32 r31, r28, r31, 18;\n\t"
		// A = {r0, r1}    B = {r26, r31}    C = {r20, r21}    D = {r8, r14}
		"lop3.b32 r28, r0, r26, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r31, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r26, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r31, r21, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r31, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r26, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r31, r21, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r31, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r26, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r31, r21, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r0, r26, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r31, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r26, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r31, r21, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r26, r31}    C = {r20, r21}    D = {r8, r14}
		/*
		* |------------------------[ROUND 11.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r26, r31}           |
		* |            v[ 6]            |           {r30, r48}           |
		* |            v[ 7]            |           {r24, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r15,  r9}           |
		* |            v[13]            |           {r10, r11}           |
		* |            v[14]            |           {r12, r13}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r9}
		"add.cc.u32 r2, r2, r30;\n\t"
		"addc.u32 r3, r3, r48;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r9}
		"xor.b32 r28, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r2, r28, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r9}
		"xor.b32 r15, r15, r2;\n\t"
		"xor.b32 r9, r9, r3;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r9}
		"shf.r.wrap.b32 r28, r15, r9, 60;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 60;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r28}
		"add.cc.u32 r22, r22, r15;\n\t"
		"addc.u32 r23, r23, r28;\n\t"
		// A = {r2, r3}    B = {r30, r48}    C = {r22, r23}    D = {r15, r28}
		"xor.b32 r30, r30, r22;\n\t"
		"xor.b32 r48, r48, r23;\n\t"
		"shf.r.wrap.b32 r9, r30, r48, 43;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 43;\n\t"
		// A = {r2, r3}    B = {r30, r9}    C = {r22, r23}    D = {r15, r28}
		"add.cc.u32 r2, r2, r30;\n\t"
		"addc.u32 r3, r3, r9;\n\t"
		// A = {r2, r3}    B = {r30, r9}    C = {r22, r23}    D = {r15, r28}
		"xor.b32 r48, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r2, r2, r48;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r30, r9}    C = {r22, r23}    D = {r15, r28}
		"xor.b32 r15, r15, r2;\n\t"
		"xor.b32 r28, r28, r3;\n\t"
		"shf.r.wrap.b32 r48, r15, r28, 5;\n\t"
		"shf.r.wrap.b32 r15, r28, r15, 5;\n\t"
		// A = {r2, r3}    B = {r30, r9}    C = {r22, r23}    D = {r48, r15}
		"add.cc.u32 r22, r22, r48;\n\t"
		"addc.u32 r23, r23, r15;\n\t"
		// A = {r2, r3}    B = {r30, r9}    C = {r22, r23}    D = {r48, r15}
		"xor.b32 r30, r30, r22;\n\t"
		"xor.b32 r9, r9, r23;\n\t"
		"shf.r.wrap.b32 r28, r30, r9, 18;\n\t"
		"shf.r.wrap.b32 r30, r9, r30, 18;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r22, r23}    D = {r48, r15}
		"lop3.b32 r9, r2, r28, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r30, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r28, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r30, r23, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r30, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r28, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r30, r23, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r30, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r28, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r30, r23, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r2, r28, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r30, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r28, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r30, r23, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r28, r30}    C = {r22, r23}    D = {r48, r15}
		/*
		* |------------------------[ROUND 11.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r26, r31}           |
		* |            v[ 6]            |           {r28, r30}           |
		* |            v[ 7]            |           {r24, r25}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r15}           |
		* |            v[13]            |           {r10, r11}           |
		* |            v[14]            |           {r12, r13}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r11}
		"add.cc.u32 r4, r4, r24;\n\t"
		"addc.u32 r5, r5, r25;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r11}
		"xor.b32 r9, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r4, r9, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r11}
		"xor.b32 r10, r10, r4;\n\t"
		"xor.b32 r11, r11, r5;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r11}
		"shf.r.wrap.b32 r9, r10, r11, 60;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 60;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r9}
		"add.cc.u32 r16, r16, r10;\n\t"
		"addc.u32 r17, r17, r9;\n\t"
		// A = {r4, r5}    B = {r24, r25}    C = {r16, r17}    D = {r10, r9}
		"xor.b32 r24, r24, r16;\n\t"
		"xor.b32 r25, r25, r17;\n\t"
		"shf.r.wrap.b32 r11, r24, r25, 43;\n\t"
		"shf.r.wrap.b32 r24, r25, r24, 43;\n\t"
		// A = {r4, r5}    B = {r24, r11}    C = {r16, r17}    D = {r10, r9}
		"add.cc.u32 r4, r4, r24;\n\t"
		"addc.u32 r5, r5, r11;\n\t"
		// A = {r4, r5}    B = {r24, r11}    C = {r16, r17}    D = {r10, r9}
		"xor.b32 r25, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r4, r4, r25;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r24, r11}    C = {r16, r17}    D = {r10, r9}
		"xor.b32 r10, r10, r4;\n\t"
		"xor.b32 r9, r9, r5;\n\t"
		"shf.r.wrap.b32 r25, r10, r9, 5;\n\t"
		"shf.r.wrap.b32 r10, r9, r10, 5;\n\t"
		// A = {r4, r5}    B = {r24, r11}    C = {r16, r17}    D = {r25, r10}
		"add.cc.u32 r16, r16, r25;\n\t"
		"addc.u32 r17, r17, r10;\n\t"
		// A = {r4, r5}    B = {r24, r11}    C = {r16, r17}    D = {r25, r10}
		"xor.b32 r24, r24, r16;\n\t"
		"xor.b32 r11, r11, r17;\n\t"
		"shf.r.wrap.b32 r9, r24, r11, 18;\n\t"
		"shf.r.wrap.b32 r24, r11, r24, 18;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r16, r17}    D = {r25, r10}
		"lop3.b32 r11, r4, r9, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r24, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r9, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r24, r17, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r24, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r9, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r24, r17, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r24, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r9, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r24, r17, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r4, r9, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r24, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r9, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r24, r17, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r9, r24}    C = {r16, r17}    D = {r25, r10}
		/*
		* |------------------------[ROUND 11.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r29, r27}           |
		* |            v[ 5]            |           {r26, r31}           |
		* |            v[ 6]            |           {r28, r30}           |
		* |            v[ 7]            |           { r9, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r15}           |
		* |            v[13]            |           {r25, r10}           |
		* |            v[14]            |           {r12, r13}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r13}
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r27;\n\t"
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r13}
		"xor.b32 r11, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r11, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r13}
		"xor.b32 r12, r12, r6;\n\t"
		"xor.b32 r13, r13, r7;\n\t"
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r13}
		"shf.r.wrap.b32 r11, r12, r13, 60;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 60;\n\t"
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r11}
		"add.cc.u32 r18, r18, r12;\n\t"
		"addc.u32 r19, r19, r11;\n\t"
		// A = {r6, r7}    B = {r29, r27}    C = {r18, r19}    D = {r12, r11}
		"xor.b32 r29, r29, r18;\n\t"
		"xor.b32 r27, r27, r19;\n\t"
		"shf.r.wrap.b32 r13, r29, r27, 43;\n\t"
		"shf.r.wrap.b32 r29, r27, r29, 43;\n\t"
		// A = {r6, r7}    B = {r29, r13}    C = {r18, r19}    D = {r12, r11}
		"add.cc.u32 r6, r6, r29;\n\t"
		"addc.u32 r7, r7, r13;\n\t"
		// A = {r6, r7}    B = {r29, r13}    C = {r18, r19}    D = {r12, r11}
		"xor.b32 r27, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r29, r13}    C = {r18, r19}    D = {r12, r11}
		"xor.b32 r12, r12, r6;\n\t"
		"xor.b32 r11, r11, r7;\n\t"
		"shf.r.wrap.b32 r27, r12, r11, 5;\n\t"
		"shf.r.wrap.b32 r12, r11, r12, 5;\n\t"
		// A = {r6, r7}    B = {r29, r13}    C = {r18, r19}    D = {r27, r12}
		"add.cc.u32 r18, r18, r27;\n\t"
		"addc.u32 r19, r19, r12;\n\t"
		// A = {r6, r7}    B = {r29, r13}    C = {r18, r19}    D = {r27, r12}
		"xor.b32 r29, r29, r18;\n\t"
		"xor.b32 r13, r13, r19;\n\t"
		"shf.r.wrap.b32 r11, r29, r13, 18;\n\t"
		"shf.r.wrap.b32 r29, r13, r29, 18;\n\t"
		// A = {r6, r7}    B = {r11, r29}    C = {r18, r19}    D = {r27, r12}
		"lop3.b32 r13, r6, r11, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r29, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r11, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r29, r19, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r29, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r11, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r29, r19, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r29, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r11, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r29, r19, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r6, r11, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r29, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r11, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r29, r19, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r11, r29}    C = {r18, r19}    D = {r27, r12}
		/*
		* |------------------------[ROUND 12.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11, r29}           |
		* |            v[ 5]            |           {r26, r31}           |
		* |            v[ 6]            |           {r28, r30}           |
		* |            v[ 7]            |           { r9, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r48, r15}           |
		* |            v[13]            |           {r25, r10}           |
		* |            v[14]            |           {r27, r12}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r15}
		"add.cc.u32 r0, r0, r11;\n\t"
		"addc.u32 r1, r1, r29;\n\t"
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r15}
		"xor.b32 r13, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r0, r13, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r15}
		"xor.b32 r48, r48, r0;\n\t"
		"xor.b32 r15, r15, r1;\n\t"
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r15}
		"shf.r.wrap.b32 r13, r48, r15, 60;\n\t"
		"shf.r.wrap.b32 r48, r15, r48, 60;\n\t"
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r13}
		"add.cc.u32 r16, r16, r48;\n\t"
		"addc.u32 r17, r17, r13;\n\t"
		// A = {r0, r1}    B = {r11, r29}    C = {r16, r17}    D = {r48, r13}
		"xor.b32 r11, r11, r16;\n\t"
		"xor.b32 r29, r29, r17;\n\t"
		"shf.r.wrap.b32 r15, r11, r29, 43;\n\t"
		"shf.r.wrap.b32 r11, r29, r11, 43;\n\t"
		// A = {r0, r1}    B = {r11, r15}    C = {r16, r17}    D = {r48, r13}
		"add.cc.u32 r0, r0, r11;\n\t"
		"addc.u32 r1, r1, r15;\n\t"
		// A = {r0, r1}    B = {r11, r15}    C = {r16, r17}    D = {r48, r13}
		"xor.b32 r29, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r11, r15}    C = {r16, r17}    D = {r48, r13}
		"xor.b32 r48, r48, r0;\n\t"
		"xor.b32 r13, r13, r1;\n\t"
		"shf.r.wrap.b32 r29, r48, r13, 5;\n\t"
		"shf.r.wrap.b32 r48, r13, r48, 5;\n\t"
		// A = {r0, r1}    B = {r11, r15}    C = {r16, r17}    D = {r29, r48}
		"add.cc.u32 r16, r16, r29;\n\t"
		"addc.u32 r17, r17, r48;\n\t"
		// A = {r0, r1}    B = {r11, r15}    C = {r16, r17}    D = {r29, r48}
		"xor.b32 r11, r11, r16;\n\t"
		"xor.b32 r15, r15, r17;\n\t"
		"shf.r.wrap.b32 r13, r11, r15, 18;\n\t"
		"shf.r.wrap.b32 r11, r15, r11, 18;\n\t"
		// A = {r0, r1}    B = {r13, r11}    C = {r16, r17}    D = {r29, r48}
		"lop3.b32 r15, r0, r13, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r11, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r13, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r11, r17, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r11, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r13, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r11, r17, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r11, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r13, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r11, r17, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r0, r13, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r11, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r13, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r11, r17, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r13, r11}    C = {r16, r17}    D = {r29, r48}
		/*
		* |------------------------[ROUND 12.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r26, r31}           |
		* |            v[ 6]            |           {r28, r30}           |
		* |            v[ 7]            |           { r9, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r48}           |
		* |            v[13]            |           {r25, r10}           |
		* |            v[14]            |           {r27, r12}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r10}
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r31;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r10}
		"xor.b32 r15, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r2, r15, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r10}
		"xor.b32 r25, r25, r2;\n\t"
		"xor.b32 r10, r10, r3;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r10}
		"shf.r.wrap.b32 r15, r25, r10, 60;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 60;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r15}
		"add.cc.u32 r18, r18, r25;\n\t"
		"addc.u32 r19, r19, r15;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r18, r19}    D = {r25, r15}
		"xor.b32 r26, r26, r18;\n\t"
		"xor.b32 r31, r31, r19;\n\t"
		"shf.r.wrap.b32 r10, r26, r31, 43;\n\t"
		"shf.r.wrap.b32 r26, r31, r26, 43;\n\t"
		// A = {r2, r3}    B = {r26, r10}    C = {r18, r19}    D = {r25, r15}
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r10;\n\t"
		// A = {r2, r3}    B = {r26, r10}    C = {r18, r19}    D = {r25, r15}
		"xor.b32 r31, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r26, r10}    C = {r18, r19}    D = {r25, r15}
		"xor.b32 r25, r25, r2;\n\t"
		"xor.b32 r15, r15, r3;\n\t"
		"shf.r.wrap.b32 r31, r25, r15, 5;\n\t"
		"shf.r.wrap.b32 r25, r15, r25, 5;\n\t"
		// A = {r2, r3}    B = {r26, r10}    C = {r18, r19}    D = {r31, r25}
		"add.cc.u32 r18, r18, r31;\n\t"
		"addc.u32 r19, r19, r25;\n\t"
		// A = {r2, r3}    B = {r26, r10}    C = {r18, r19}    D = {r31, r25}
		"xor.b32 r26, r26, r18;\n\t"
		"xor.b32 r10, r10, r19;\n\t"
		"shf.r.wrap.b32 r15, r26, r10, 18;\n\t"
		"shf.r.wrap.b32 r26, r10, r26, 18;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r18, r19}    D = {r31, r25}
		"lop3.b32 r10, r2, r15, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r26, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r15, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r26, r19, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r26, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r15, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r26, r19, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r26, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r15, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r26, r19, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r2, r15, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r26, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r15, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r26, r19, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r18, r19}    D = {r31, r25}
		/*
		* |------------------------[ROUND 12.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r15, r26}           |
		* |            v[ 6]            |           {r28, r30}           |
		* |            v[ 7]            |           { r9, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r48}           |
		* |            v[13]            |           {r31, r25}           |
		* |            v[14]            |           {r27, r12}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r12}
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r30;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r12}
		"xor.b32 r10, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r4, r10, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r12}
		"xor.b32 r27, r27, r4;\n\t"
		"xor.b32 r12, r12, r5;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r12}
		"shf.r.wrap.b32 r10, r27, r12, 60;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 60;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r10}
		"add.cc.u32 r20, r20, r27;\n\t"
		"addc.u32 r21, r21, r10;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r20, r21}    D = {r27, r10}
		"xor.b32 r28, r28, r20;\n\t"
		"xor.b32 r30, r30, r21;\n\t"
		"shf.r.wrap.b32 r12, r28, r30, 43;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 43;\n\t"
		// A = {r4, r5}    B = {r28, r12}    C = {r20, r21}    D = {r27, r10}
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r12;\n\t"
		// A = {r4, r5}    B = {r28, r12}    C = {r20, r21}    D = {r27, r10}
		"xor.b32 r30, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r28, r12}    C = {r20, r21}    D = {r27, r10}
		"xor.b32 r27, r27, r4;\n\t"
		"xor.b32 r10, r10, r5;\n\t"
		"shf.r.wrap.b32 r30, r27, r10, 5;\n\t"
		"shf.r.wrap.b32 r27, r10, r27, 5;\n\t"
		// A = {r4, r5}    B = {r28, r12}    C = {r20, r21}    D = {r30, r27}
		"add.cc.u32 r20, r20, r30;\n\t"
		"addc.u32 r21, r21, r27;\n\t"
		// A = {r4, r5}    B = {r28, r12}    C = {r20, r21}    D = {r30, r27}
		"xor.b32 r28, r28, r20;\n\t"
		"xor.b32 r12, r12, r21;\n\t"
		"shf.r.wrap.b32 r10, r28, r12, 18;\n\t"
		"shf.r.wrap.b32 r28, r12, r28, 18;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r20, r21}    D = {r30, r27}
		"lop3.b32 r12, r4, r10, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r28, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r10, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r28, r21, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r28, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r10, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r28, r21, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r28, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r10, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r28, r21, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r4, r10, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r28, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r10, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r28, r21, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r20, r21}    D = {r30, r27}
		/*
		* |------------------------[ROUND 12.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r15, r26}           |
		* |            v[ 6]            |           {r10, r28}           |
		* |            v[ 7]            |           { r9, r24}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r48}           |
		* |            v[13]            |           {r31, r25}           |
		* |            v[14]            |           {r30, r27}           |
		* |            v[15]            |           { r8, r14}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r14}
		"add.cc.u32 r6, r6, r9;\n\t"
		"addc.u32 r7, r7, r24;\n\t"
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r14}
		"xor.b32 r12, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r12, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r14}
		"xor.b32 r8, r8, r6;\n\t"
		"xor.b32 r14, r14, r7;\n\t"
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r14}
		"shf.r.wrap.b32 r12, r8, r14, 60;\n\t"
		"shf.r.wrap.b32 r8, r14, r8, 60;\n\t"
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r12}
		"add.cc.u32 r22, r22, r8;\n\t"
		"addc.u32 r23, r23, r12;\n\t"
		// A = {r6, r7}    B = {r9, r24}    C = {r22, r23}    D = {r8, r12}
		"xor.b32 r9, r9, r22;\n\t"
		"xor.b32 r24, r24, r23;\n\t"
		"shf.r.wrap.b32 r14, r9, r24, 43;\n\t"
		"shf.r.wrap.b32 r9, r24, r9, 43;\n\t"
		// A = {r6, r7}    B = {r9, r14}    C = {r22, r23}    D = {r8, r12}
		"add.cc.u32 r6, r6, r9;\n\t"
		"addc.u32 r7, r7, r14;\n\t"
		// A = {r6, r7}    B = {r9, r14}    C = {r22, r23}    D = {r8, r12}
		"xor.b32 r24, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r6, r24;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r9, r14}    C = {r22, r23}    D = {r8, r12}
		"xor.b32 r8, r8, r6;\n\t"
		"xor.b32 r12, r12, r7;\n\t"
		"shf.r.wrap.b32 r24, r8, r12, 5;\n\t"
		"shf.r.wrap.b32 r8, r12, r8, 5;\n\t"
		// A = {r6, r7}    B = {r9, r14}    C = {r22, r23}    D = {r24, r8}
		"add.cc.u32 r22, r22, r24;\n\t"
		"addc.u32 r23, r23, r8;\n\t"
		// A = {r6, r7}    B = {r9, r14}    C = {r22, r23}    D = {r24, r8}
		"xor.b32 r9, r9, r22;\n\t"
		"xor.b32 r14, r14, r23;\n\t"
		"shf.r.wrap.b32 r12, r9, r14, 18;\n\t"
		"shf.r.wrap.b32 r9, r14, r9, 18;\n\t"
		// A = {r6, r7}    B = {r12, r9}    C = {r22, r23}    D = {r24, r8}
		"lop3.b32 r14, r6, r12, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r9, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r12, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r9, r23, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r9, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r12, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r9, r23, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r9, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r12, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r9, r23, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r6, r12, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r9, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r12, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r9, r23, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r12, r9}    C = {r22, r23}    D = {r24, r8}
		/*
		* |------------------------[ROUND 12.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r15, r26}           |
		* |            v[ 6]            |           {r10, r28}           |
		* |            v[ 7]            |           {r12,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r48}           |
		* |            v[13]            |           {r31, r25}           |
		* |            v[14]            |           {r30, r27}           |
		* |            v[15]            |           {r24,  r8}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r8}
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r26;\n\t"
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r8}
		"xor.b32 r14, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r14, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r8}
		"xor.b32 r24, r24, r0;\n\t"
		"xor.b32 r8, r8, r1;\n\t"
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r8}
		"shf.r.wrap.b32 r14, r24, r8, 60;\n\t"
		"shf.r.wrap.b32 r24, r8, r24, 60;\n\t"
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r14}
		"add.cc.u32 r20, r20, r24;\n\t"
		"addc.u32 r21, r21, r14;\n\t"
		// A = {r0, r1}    B = {r15, r26}    C = {r20, r21}    D = {r24, r14}
		"xor.b32 r15, r15, r20;\n\t"
		"xor.b32 r26, r26, r21;\n\t"
		"shf.r.wrap.b32 r8, r15, r26, 43;\n\t"
		"shf.r.wrap.b32 r15, r26, r15, 43;\n\t"
		// A = {r0, r1}    B = {r15, r8}    C = {r20, r21}    D = {r24, r14}
		"add.cc.u32 r0, r0, r15;\n\t"
		"addc.u32 r1, r1, r8;\n\t"
		// A = {r0, r1}    B = {r15, r8}    C = {r20, r21}    D = {r24, r14}
		"xor.b32 r26, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r0, r0, r26;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r15, r8}    C = {r20, r21}    D = {r24, r14}
		"xor.b32 r24, r24, r0;\n\t"
		"xor.b32 r14, r14, r1;\n\t"
		"shf.r.wrap.b32 r26, r24, r14, 5;\n\t"
		"shf.r.wrap.b32 r24, r14, r24, 5;\n\t"
		// A = {r0, r1}    B = {r15, r8}    C = {r20, r21}    D = {r26, r24}
		"add.cc.u32 r20, r20, r26;\n\t"
		"addc.u32 r21, r21, r24;\n\t"
		// A = {r0, r1}    B = {r15, r8}    C = {r20, r21}    D = {r26, r24}
		"xor.b32 r15, r15, r20;\n\t"
		"xor.b32 r8, r8, r21;\n\t"
		"shf.r.wrap.b32 r14, r15, r8, 18;\n\t"
		"shf.r.wrap.b32 r15, r8, r15, 18;\n\t"
		// A = {r0, r1}    B = {r14, r15}    C = {r20, r21}    D = {r26, r24}
		"lop3.b32 r8, r0, r14, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r15, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r14, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r15, r21, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r15, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r14, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r15, r21, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r15, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r14, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r15, r21, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r0, r14, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r15, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r14, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r15, r21, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r14, r15}    C = {r20, r21}    D = {r26, r24}
		/*
		* |------------------------[ROUND 12.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r14, r15}           |
		* |            v[ 6]            |           {r10, r28}           |
		* |            v[ 7]            |           {r12,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r29, r48}           |
		* |            v[13]            |           {r31, r25}           |
		* |            v[14]            |           {r30, r27}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r48}
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r28;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r48}
		"xor.b32 r8, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r2, r8, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r48}
		"xor.b32 r29, r29, r2;\n\t"
		"xor.b32 r48, r48, r3;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r48}
		"shf.r.wrap.b32 r8, r29, r48, 60;\n\t"
		"shf.r.wrap.b32 r29, r48, r29, 60;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r8}
		"add.cc.u32 r22, r22, r29;\n\t"
		"addc.u32 r23, r23, r8;\n\t"
		// A = {r2, r3}    B = {r10, r28}    C = {r22, r23}    D = {r29, r8}
		"xor.b32 r10, r10, r22;\n\t"
		"xor.b32 r28, r28, r23;\n\t"
		"shf.r.wrap.b32 r48, r10, r28, 43;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 43;\n\t"
		// A = {r2, r3}    B = {r10, r48}    C = {r22, r23}    D = {r29, r8}
		"add.cc.u32 r2, r2, r10;\n\t"
		"addc.u32 r3, r3, r48;\n\t"
		// A = {r2, r3}    B = {r10, r48}    C = {r22, r23}    D = {r29, r8}
		"xor.b32 r28, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r2, r28;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r10, r48}    C = {r22, r23}    D = {r29, r8}
		"xor.b32 r29, r29, r2;\n\t"
		"xor.b32 r8, r8, r3;\n\t"
		"shf.r.wrap.b32 r28, r29, r8, 5;\n\t"
		"shf.r.wrap.b32 r29, r8, r29, 5;\n\t"
		// A = {r2, r3}    B = {r10, r48}    C = {r22, r23}    D = {r28, r29}
		"add.cc.u32 r22, r22, r28;\n\t"
		"addc.u32 r23, r23, r29;\n\t"
		// A = {r2, r3}    B = {r10, r48}    C = {r22, r23}    D = {r28, r29}
		"xor.b32 r10, r10, r22;\n\t"
		"xor.b32 r48, r48, r23;\n\t"
		"shf.r.wrap.b32 r8, r10, r48, 18;\n\t"
		"shf.r.wrap.b32 r10, r48, r10, 18;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r22, r23}    D = {r28, r29}
		"lop3.b32 r48, r2, r8, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r10, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r8, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r10, r23, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r10, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r8, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r10, r23, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r10, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r8, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r10, r23, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r2, r8, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r10, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r8, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r10, r23, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r8, r10}    C = {r22, r23}    D = {r28, r29}
		/*
		* |------------------------[ROUND 12.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r14, r15}           |
		* |            v[ 6]            |           { r8, r10}           |
		* |            v[ 7]            |           {r12,  r9}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r29}           |
		* |            v[13]            |           {r31, r25}           |
		* |            v[14]            |           {r30, r27}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r25}
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r9;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r25}
		"xor.b32 r48, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r4, r48, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r25}
		"xor.b32 r31, r31, r4;\n\t"
		"xor.b32 r25, r25, r5;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r25}
		"shf.r.wrap.b32 r48, r31, r25, 60;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 60;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r48}
		"add.cc.u32 r16, r16, r31;\n\t"
		"addc.u32 r17, r17, r48;\n\t"
		// A = {r4, r5}    B = {r12, r9}    C = {r16, r17}    D = {r31, r48}
		"xor.b32 r12, r12, r16;\n\t"
		"xor.b32 r9, r9, r17;\n\t"
		"shf.r.wrap.b32 r25, r12, r9, 43;\n\t"
		"shf.r.wrap.b32 r12, r9, r12, 43;\n\t"
		// A = {r4, r5}    B = {r12, r25}    C = {r16, r17}    D = {r31, r48}
		"add.cc.u32 r4, r4, r12;\n\t"
		"addc.u32 r5, r5, r25;\n\t"
		// A = {r4, r5}    B = {r12, r25}    C = {r16, r17}    D = {r31, r48}
		"xor.b32 r9, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r4, r4, r9;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r12, r25}    C = {r16, r17}    D = {r31, r48}
		"xor.b32 r31, r31, r4;\n\t"
		"xor.b32 r48, r48, r5;\n\t"
		"shf.r.wrap.b32 r9, r31, r48, 5;\n\t"
		"shf.r.wrap.b32 r31, r48, r31, 5;\n\t"
		// A = {r4, r5}    B = {r12, r25}    C = {r16, r17}    D = {r9, r31}
		"add.cc.u32 r16, r16, r9;\n\t"
		"addc.u32 r17, r17, r31;\n\t"
		// A = {r4, r5}    B = {r12, r25}    C = {r16, r17}    D = {r9, r31}
		"xor.b32 r12, r12, r16;\n\t"
		"xor.b32 r25, r25, r17;\n\t"
		"shf.r.wrap.b32 r48, r12, r25, 18;\n\t"
		"shf.r.wrap.b32 r12, r25, r12, 18;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r16, r17}    D = {r9, r31}
		"lop3.b32 r25, r4, r48, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r12, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r48, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r12, r17, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r12, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r48, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r12, r17, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r12, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r48, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r12, r17, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r4, r48, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r12, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r48, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r12, r17, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r48, r12}    C = {r16, r17}    D = {r9, r31}
		/*
		* |------------------------[ROUND 12.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r13, r11}           |
		* |            v[ 5]            |           {r14, r15}           |
		* |            v[ 6]            |           { r8, r10}           |
		* |            v[ 7]            |           {r48, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r29}           |
		* |            v[13]            |           { r9, r31}           |
		* |            v[14]            |           {r30, r27}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r27}
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r11;\n\t"
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r27}
		"xor.b32 r25, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r6, r25, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r27}
		"xor.b32 r30, r30, r6;\n\t"
		"xor.b32 r27, r27, r7;\n\t"
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r27}
		"shf.r.wrap.b32 r25, r30, r27, 60;\n\t"
		"shf.r.wrap.b32 r30, r27, r30, 60;\n\t"
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r25}
		"add.cc.u32 r18, r18, r30;\n\t"
		"addc.u32 r19, r19, r25;\n\t"
		// A = {r6, r7}    B = {r13, r11}    C = {r18, r19}    D = {r30, r25}
		"xor.b32 r13, r13, r18;\n\t"
		"xor.b32 r11, r11, r19;\n\t"
		"shf.r.wrap.b32 r27, r13, r11, 43;\n\t"
		"shf.r.wrap.b32 r13, r11, r13, 43;\n\t"
		// A = {r6, r7}    B = {r13, r27}    C = {r18, r19}    D = {r30, r25}
		"add.cc.u32 r6, r6, r13;\n\t"
		"addc.u32 r7, r7, r27;\n\t"
		// A = {r6, r7}    B = {r13, r27}    C = {r18, r19}    D = {r30, r25}
		"xor.b32 r11, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r13, r27}    C = {r18, r19}    D = {r30, r25}
		"xor.b32 r30, r30, r6;\n\t"
		"xor.b32 r25, r25, r7;\n\t"
		"shf.r.wrap.b32 r11, r30, r25, 5;\n\t"
		"shf.r.wrap.b32 r30, r25, r30, 5;\n\t"
		// A = {r6, r7}    B = {r13, r27}    C = {r18, r19}    D = {r11, r30}
		"add.cc.u32 r18, r18, r11;\n\t"
		"addc.u32 r19, r19, r30;\n\t"
		// A = {r6, r7}    B = {r13, r27}    C = {r18, r19}    D = {r11, r30}
		"xor.b32 r13, r13, r18;\n\t"
		"xor.b32 r27, r27, r19;\n\t"
		"shf.r.wrap.b32 r25, r13, r27, 18;\n\t"
		"shf.r.wrap.b32 r13, r27, r13, 18;\n\t"
		// A = {r6, r7}    B = {r25, r13}    C = {r18, r19}    D = {r11, r30}
		"lop3.b32 r27, r6, r25, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r13, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r25, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r13, r19, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r13, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r25, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r13, r19, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r13, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r25, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r13, r19, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r6, r25, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r13, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r25, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r13, r19, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r25, r13}    C = {r18, r19}    D = {r11, r30}
		/*
		* |------------------------[ROUND 13.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r13}           |
		* |            v[ 5]            |           {r14, r15}           |
		* |            v[ 6]            |           { r8, r10}           |
		* |            v[ 7]            |           {r48, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r28, r29}           |
		* |            v[13]            |           { r9, r31}           |
		* |            v[14]            |           {r11, r30}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r29}
		"add.cc.u32 r0, r0, r25;\n\t"
		"addc.u32 r1, r1, r13;\n\t"
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r29}
		"xor.b32 r27, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r27, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r29}
		"xor.b32 r28, r28, r0;\n\t"
		"xor.b32 r29, r29, r1;\n\t"
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r29}
		"shf.r.wrap.b32 r27, r28, r29, 60;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 60;\n\t"
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r27}
		"add.cc.u32 r16, r16, r28;\n\t"
		"addc.u32 r17, r17, r27;\n\t"
		// A = {r0, r1}    B = {r25, r13}    C = {r16, r17}    D = {r28, r27}
		"xor.b32 r25, r25, r16;\n\t"
		"xor.b32 r13, r13, r17;\n\t"
		"shf.r.wrap.b32 r29, r25, r13, 43;\n\t"
		"shf.r.wrap.b32 r25, r13, r25, 43;\n\t"
		// A = {r0, r1}    B = {r25, r29}    C = {r16, r17}    D = {r28, r27}
		"add.cc.u32 r0, r0, r25;\n\t"
		"addc.u32 r1, r1, r29;\n\t"
		// A = {r0, r1}    B = {r25, r29}    C = {r16, r17}    D = {r28, r27}
		"xor.b32 r13, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r25, r29}    C = {r16, r17}    D = {r28, r27}
		"xor.b32 r28, r28, r0;\n\t"
		"xor.b32 r27, r27, r1;\n\t"
		"shf.r.wrap.b32 r13, r28, r27, 5;\n\t"
		"shf.r.wrap.b32 r28, r27, r28, 5;\n\t"
		// A = {r0, r1}    B = {r25, r29}    C = {r16, r17}    D = {r13, r28}
		"add.cc.u32 r16, r16, r13;\n\t"
		"addc.u32 r17, r17, r28;\n\t"
		// A = {r0, r1}    B = {r25, r29}    C = {r16, r17}    D = {r13, r28}
		"xor.b32 r25, r25, r16;\n\t"
		"xor.b32 r29, r29, r17;\n\t"
		"shf.r.wrap.b32 r27, r25, r29, 18;\n\t"
		"shf.r.wrap.b32 r25, r29, r25, 18;\n\t"
		// A = {r0, r1}    B = {r27, r25}    C = {r16, r17}    D = {r13, r28}
		"lop3.b32 r29, r0, r27, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r25, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r27, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r25, r17, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r25, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r27, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r25, r17, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r25, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r27, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r25, r17, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r0, r27, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r25, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r27, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r25, r17, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r27, r25}    C = {r16, r17}    D = {r13, r28}
		/*
		* |------------------------[ROUND 13.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r14, r15}           |
		* |            v[ 6]            |           { r8, r10}           |
		* |            v[ 7]            |           {r48, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r28}           |
		* |            v[13]            |           { r9, r31}           |
		* |            v[14]            |           {r11, r30}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r31}
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r15;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r31}
		"xor.b32 r29, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r2, r29, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r31}
		"xor.b32 r9, r9, r2;\n\t"
		"xor.b32 r31, r31, r3;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r31}
		"shf.r.wrap.b32 r29, r9, r31, 60;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 60;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r29}
		"add.cc.u32 r18, r18, r9;\n\t"
		"addc.u32 r19, r19, r29;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r18, r19}    D = {r9, r29}
		"xor.b32 r14, r14, r18;\n\t"
		"xor.b32 r15, r15, r19;\n\t"
		"shf.r.wrap.b32 r31, r14, r15, 43;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 43;\n\t"
		// A = {r2, r3}    B = {r14, r31}    C = {r18, r19}    D = {r9, r29}
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r31;\n\t"
		// A = {r2, r3}    B = {r14, r31}    C = {r18, r19}    D = {r9, r29}
		"xor.b32 r15, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r14, r31}    C = {r18, r19}    D = {r9, r29}
		"xor.b32 r9, r9, r2;\n\t"
		"xor.b32 r29, r29, r3;\n\t"
		"shf.r.wrap.b32 r15, r9, r29, 5;\n\t"
		"shf.r.wrap.b32 r9, r29, r9, 5;\n\t"
		// A = {r2, r3}    B = {r14, r31}    C = {r18, r19}    D = {r15, r9}
		"add.cc.u32 r18, r18, r15;\n\t"
		"addc.u32 r19, r19, r9;\n\t"
		// A = {r2, r3}    B = {r14, r31}    C = {r18, r19}    D = {r15, r9}
		"xor.b32 r14, r14, r18;\n\t"
		"xor.b32 r31, r31, r19;\n\t"
		"shf.r.wrap.b32 r29, r14, r31, 18;\n\t"
		"shf.r.wrap.b32 r14, r31, r14, 18;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r18, r19}    D = {r15, r9}
		"lop3.b32 r31, r2, r29, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r14, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r29, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r14, r19, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r14, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r29, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r14, r19, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r14, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r29, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r14, r19, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r2, r29, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r14, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r29, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r14, r19, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r18, r19}    D = {r15, r9}
		/*
		* |------------------------[ROUND 13.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r29, r14}           |
		* |            v[ 6]            |           { r8, r10}           |
		* |            v[ 7]            |           {r48, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r28}           |
		* |            v[13]            |           {r15,  r9}           |
		* |            v[14]            |           {r11, r30}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r30}
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r10;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r30}
		"xor.b32 r31, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r4, r31, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r30}
		"xor.b32 r11, r11, r4;\n\t"
		"xor.b32 r30, r30, r5;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r30}
		"shf.r.wrap.b32 r31, r11, r30, 60;\n\t"
		"shf.r.wrap.b32 r11, r30, r11, 60;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r31}
		"add.cc.u32 r20, r20, r11;\n\t"
		"addc.u32 r21, r21, r31;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r20, r21}    D = {r11, r31}
		"xor.b32 r8, r8, r20;\n\t"
		"xor.b32 r10, r10, r21;\n\t"
		"shf.r.wrap.b32 r30, r8, r10, 43;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 43;\n\t"
		// A = {r4, r5}    B = {r8, r30}    C = {r20, r21}    D = {r11, r31}
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r30;\n\t"
		// A = {r4, r5}    B = {r8, r30}    C = {r20, r21}    D = {r11, r31}
		"xor.b32 r10, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r8, r30}    C = {r20, r21}    D = {r11, r31}
		"xor.b32 r11, r11, r4;\n\t"
		"xor.b32 r31, r31, r5;\n\t"
		"shf.r.wrap.b32 r10, r11, r31, 5;\n\t"
		"shf.r.wrap.b32 r11, r31, r11, 5;\n\t"
		// A = {r4, r5}    B = {r8, r30}    C = {r20, r21}    D = {r10, r11}
		"add.cc.u32 r20, r20, r10;\n\t"
		"addc.u32 r21, r21, r11;\n\t"
		// A = {r4, r5}    B = {r8, r30}    C = {r20, r21}    D = {r10, r11}
		"xor.b32 r8, r8, r20;\n\t"
		"xor.b32 r30, r30, r21;\n\t"
		"shf.r.wrap.b32 r31, r8, r30, 18;\n\t"
		"shf.r.wrap.b32 r8, r30, r8, 18;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r20, r21}    D = {r10, r11}
		"lop3.b32 r30, r4, r31, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r8, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r31, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r8, r21, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r8, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r31, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r8, r21, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r8, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r31, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r8, r21, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r4, r31, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r8, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r31, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r8, r21, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r20, r21}    D = {r10, r11}
		/*
		* |------------------------[ROUND 13.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r29, r14}           |
		* |            v[ 6]            |           {r31,  r8}           |
		* |            v[ 7]            |           {r48, r12}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r28}           |
		* |            v[13]            |           {r15,  r9}           |
		* |            v[14]            |           {r10, r11}           |
		* |            v[15]            |           {r26, r24}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r24}
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r12;\n\t"
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r24}
		"xor.b32 r30, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r6, r30, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r24}
		"xor.b32 r26, r26, r6;\n\t"
		"xor.b32 r24, r24, r7;\n\t"
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r24}
		"shf.r.wrap.b32 r30, r26, r24, 60;\n\t"
		"shf.r.wrap.b32 r26, r24, r26, 60;\n\t"
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r30}
		"add.cc.u32 r22, r22, r26;\n\t"
		"addc.u32 r23, r23, r30;\n\t"
		// A = {r6, r7}    B = {r48, r12}    C = {r22, r23}    D = {r26, r30}
		"xor.b32 r48, r48, r22;\n\t"
		"xor.b32 r12, r12, r23;\n\t"
		"shf.r.wrap.b32 r24, r48, r12, 43;\n\t"
		"shf.r.wrap.b32 r48, r12, r48, 43;\n\t"
		// A = {r6, r7}    B = {r48, r24}    C = {r22, r23}    D = {r26, r30}
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r24;\n\t"
		// A = {r6, r7}    B = {r48, r24}    C = {r22, r23}    D = {r26, r30}
		"xor.b32 r12, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r6, r6, r12;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r48, r24}    C = {r22, r23}    D = {r26, r30}
		"xor.b32 r26, r26, r6;\n\t"
		"xor.b32 r30, r30, r7;\n\t"
		"shf.r.wrap.b32 r12, r26, r30, 5;\n\t"
		"shf.r.wrap.b32 r26, r30, r26, 5;\n\t"
		// A = {r6, r7}    B = {r48, r24}    C = {r22, r23}    D = {r12, r26}
		"add.cc.u32 r22, r22, r12;\n\t"
		"addc.u32 r23, r23, r26;\n\t"
		// A = {r6, r7}    B = {r48, r24}    C = {r22, r23}    D = {r12, r26}
		"xor.b32 r48, r48, r22;\n\t"
		"xor.b32 r24, r24, r23;\n\t"
		"shf.r.wrap.b32 r30, r48, r24, 18;\n\t"
		"shf.r.wrap.b32 r48, r24, r48, 18;\n\t"
		// A = {r6, r7}    B = {r30, r48}    C = {r22, r23}    D = {r12, r26}
		"lop3.b32 r24, r6, r30, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r48, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r30, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r48, r23, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r48, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r30, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r48, r23, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r48, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r30, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r48, r23, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r6, r30, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r48, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r30, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r48, r23, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r30, r48}    C = {r22, r23}    D = {r12, r26}
		/*
		* |------------------------[ROUND 13.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r29, r14}           |
		* |            v[ 6]            |           {r31,  r8}           |
		* |            v[ 7]            |           {r30, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r28}           |
		* |            v[13]            |           {r15,  r9}           |
		* |            v[14]            |           {r10, r11}           |
		* |            v[15]            |           {r12, r26}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r26}
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r14;\n\t"
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r26}
		"xor.b32 r24, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r0, r24, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r26}
		"xor.b32 r12, r12, r0;\n\t"
		"xor.b32 r26, r26, r1;\n\t"
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r26}
		"shf.r.wrap.b32 r24, r12, r26, 60;\n\t"
		"shf.r.wrap.b32 r12, r26, r12, 60;\n\t"
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r24}
		"add.cc.u32 r20, r20, r12;\n\t"
		"addc.u32 r21, r21, r24;\n\t"
		// A = {r0, r1}    B = {r29, r14}    C = {r20, r21}    D = {r12, r24}
		"xor.b32 r29, r29, r20;\n\t"
		"xor.b32 r14, r14, r21;\n\t"
		"shf.r.wrap.b32 r26, r29, r14, 43;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 43;\n\t"
		// A = {r0, r1}    B = {r29, r26}    C = {r20, r21}    D = {r12, r24}
		"add.cc.u32 r0, r0, r29;\n\t"
		"addc.u32 r1, r1, r26;\n\t"
		// A = {r0, r1}    B = {r29, r26}    C = {r20, r21}    D = {r12, r24}
		"xor.b32 r14, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r0, r14;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r29, r26}    C = {r20, r21}    D = {r12, r24}
		"xor.b32 r12, r12, r0;\n\t"
		"xor.b32 r24, r24, r1;\n\t"
		"shf.r.wrap.b32 r14, r12, r24, 5;\n\t"
		"shf.r.wrap.b32 r12, r24, r12, 5;\n\t"
		// A = {r0, r1}    B = {r29, r26}    C = {r20, r21}    D = {r14, r12}
		"add.cc.u32 r20, r20, r14;\n\t"
		"addc.u32 r21, r21, r12;\n\t"
		// A = {r0, r1}    B = {r29, r26}    C = {r20, r21}    D = {r14, r12}
		"xor.b32 r29, r29, r20;\n\t"
		"xor.b32 r26, r26, r21;\n\t"
		"shf.r.wrap.b32 r24, r29, r26, 18;\n\t"
		"shf.r.wrap.b32 r29, r26, r29, 18;\n\t"
		// A = {r0, r1}    B = {r24, r29}    C = {r20, r21}    D = {r14, r12}
		"lop3.b32 r26, r0, r24, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r29, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r24, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r29, r21, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r29, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r24, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r29, r21, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r29, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r24, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r29, r21, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r0, r24, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r29, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r24, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r29, r21, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r12, r12, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r24, r29}    C = {r20, r21}    D = {r14, r12}
		/*
		* |------------------------[ROUND 13.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r24, r29}           |
		* |            v[ 6]            |           {r31,  r8}           |
		* |            v[ 7]            |           {r30, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r13, r28}           |
		* |            v[13]            |           {r15,  r9}           |
		* |            v[14]            |           {r10, r11}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r28}
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r8;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r28}
		"xor.b32 r26, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r26, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r28}
		"xor.b32 r13, r13, r2;\n\t"
		"xor.b32 r28, r28, r3;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r28}
		"shf.r.wrap.b32 r26, r13, r28, 60;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 60;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r26}
		"add.cc.u32 r22, r22, r13;\n\t"
		"addc.u32 r23, r23, r26;\n\t"
		// A = {r2, r3}    B = {r31, r8}    C = {r22, r23}    D = {r13, r26}
		"xor.b32 r31, r31, r22;\n\t"
		"xor.b32 r8, r8, r23;\n\t"
		"shf.r.wrap.b32 r28, r31, r8, 43;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 43;\n\t"
		// A = {r2, r3}    B = {r31, r28}    C = {r22, r23}    D = {r13, r26}
		"add.cc.u32 r2, r2, r31;\n\t"
		"addc.u32 r3, r3, r28;\n\t"
		// A = {r2, r3}    B = {r31, r28}    C = {r22, r23}    D = {r13, r26}
		"xor.b32 r8, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r2, r8;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r31, r28}    C = {r22, r23}    D = {r13, r26}
		"xor.b32 r13, r13, r2;\n\t"
		"xor.b32 r26, r26, r3;\n\t"
		"shf.r.wrap.b32 r8, r13, r26, 5;\n\t"
		"shf.r.wrap.b32 r13, r26, r13, 5;\n\t"
		// A = {r2, r3}    B = {r31, r28}    C = {r22, r23}    D = {r8, r13}
		"add.cc.u32 r22, r22, r8;\n\t"
		"addc.u32 r23, r23, r13;\n\t"
		// A = {r2, r3}    B = {r31, r28}    C = {r22, r23}    D = {r8, r13}
		"xor.b32 r31, r31, r22;\n\t"
		"xor.b32 r28, r28, r23;\n\t"
		"shf.r.wrap.b32 r26, r31, r28, 18;\n\t"
		"shf.r.wrap.b32 r31, r28, r31, 18;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r22, r23}    D = {r8, r13}
		"lop3.b32 r28, r2, r26, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r31, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r26, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r31, r23, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r31, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r26, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r31, r23, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r31, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r26, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r31, r23, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r2, r26, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r31, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r26, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r31, r23, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r26, r31}    C = {r22, r23}    D = {r8, r13}
		/*
		* |------------------------[ROUND 13.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r24, r29}           |
		* |            v[ 6]            |           {r26, r31}           |
		* |            v[ 7]            |           {r30, r48}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r13}           |
		* |            v[13]            |           {r15,  r9}           |
		* |            v[14]            |           {r10, r11}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r9}
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r48;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r9}
		"xor.b32 r28, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r4, r28, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r9}
		"xor.b32 r15, r15, r4;\n\t"
		"xor.b32 r9, r9, r5;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r9}
		"shf.r.wrap.b32 r28, r15, r9, 60;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 60;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r28}
		"add.cc.u32 r16, r16, r15;\n\t"
		"addc.u32 r17, r17, r28;\n\t"
		// A = {r4, r5}    B = {r30, r48}    C = {r16, r17}    D = {r15, r28}
		"xor.b32 r30, r30, r16;\n\t"
		"xor.b32 r48, r48, r17;\n\t"
		"shf.r.wrap.b32 r9, r30, r48, 43;\n\t"
		"shf.r.wrap.b32 r30, r48, r30, 43;\n\t"
		// A = {r4, r5}    B = {r30, r9}    C = {r16, r17}    D = {r15, r28}
		"add.cc.u32 r4, r4, r30;\n\t"
		"addc.u32 r5, r5, r9;\n\t"
		// A = {r4, r5}    B = {r30, r9}    C = {r16, r17}    D = {r15, r28}
		"xor.b32 r48, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r4, r48;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r30, r9}    C = {r16, r17}    D = {r15, r28}
		"xor.b32 r15, r15, r4;\n\t"
		"xor.b32 r28, r28, r5;\n\t"
		"shf.r.wrap.b32 r48, r15, r28, 5;\n\t"
		"shf.r.wrap.b32 r15, r28, r15, 5;\n\t"
		// A = {r4, r5}    B = {r30, r9}    C = {r16, r17}    D = {r48, r15}
		"add.cc.u32 r16, r16, r48;\n\t"
		"addc.u32 r17, r17, r15;\n\t"
		// A = {r4, r5}    B = {r30, r9}    C = {r16, r17}    D = {r48, r15}
		"xor.b32 r30, r30, r16;\n\t"
		"xor.b32 r9, r9, r17;\n\t"
		"shf.r.wrap.b32 r28, r30, r9, 18;\n\t"
		"shf.r.wrap.b32 r30, r9, r30, 18;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r16, r17}    D = {r48, r15}
		"lop3.b32 r9, r4, r28, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r30, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r28, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r30, r17, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r30, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r28, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r30, r17, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r30, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r28, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r30, r17, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r4, r28, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r30, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r28, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r30, r17, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r28, r30}    C = {r16, r17}    D = {r48, r15}
		/*
		* |------------------------[ROUND 13.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r27, r25}           |
		* |            v[ 5]            |           {r24, r29}           |
		* |            v[ 6]            |           {r26, r31}           |
		* |            v[ 7]            |           {r28, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r13}           |
		* |            v[13]            |           {r48, r15}           |
		* |            v[14]            |           {r10, r11}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           { r9, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r11}
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r25;\n\t"
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r11}
		"xor.b32 r9, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r9, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r11}
		"xor.b32 r10, r10, r6;\n\t"
		"xor.b32 r11, r11, r7;\n\t"
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r11}
		"shf.r.wrap.b32 r9, r10, r11, 60;\n\t"
		"shf.r.wrap.b32 r10, r11, r10, 60;\n\t"
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r9}
		"add.cc.u32 r18, r18, r10;\n\t"
		"addc.u32 r19, r19, r9;\n\t"
		// A = {r6, r7}    B = {r27, r25}    C = {r18, r19}    D = {r10, r9}
		"xor.b32 r27, r27, r18;\n\t"
		"xor.b32 r25, r25, r19;\n\t"
		"shf.r.wrap.b32 r11, r27, r25, 43;\n\t"
		"shf.r.wrap.b32 r27, r25, r27, 43;\n\t"
		// A = {r6, r7}    B = {r27, r11}    C = {r18, r19}    D = {r10, r9}
		"add.cc.u32 r6, r6, r27;\n\t"
		"addc.u32 r7, r7, r11;\n\t"
		// A = {r6, r7}    B = {r27, r11}    C = {r18, r19}    D = {r10, r9}
		"xor.b32 r25, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r27, r11}    C = {r18, r19}    D = {r10, r9}
		"xor.b32 r10, r10, r6;\n\t"
		"xor.b32 r9, r9, r7;\n\t"
		"shf.r.wrap.b32 r25, r10, r9, 5;\n\t"
		"shf.r.wrap.b32 r10, r9, r10, 5;\n\t"
		// A = {r6, r7}    B = {r27, r11}    C = {r18, r19}    D = {r25, r10}
		"add.cc.u32 r18, r18, r25;\n\t"
		"addc.u32 r19, r19, r10;\n\t"
		// A = {r6, r7}    B = {r27, r11}    C = {r18, r19}    D = {r25, r10}
		"xor.b32 r27, r27, r18;\n\t"
		"xor.b32 r11, r11, r19;\n\t"
		"shf.r.wrap.b32 r9, r27, r11, 18;\n\t"
		"shf.r.wrap.b32 r27, r11, r27, 18;\n\t"
		// A = {r6, r7}    B = {r9, r27}    C = {r18, r19}    D = {r25, r10}
		"lop3.b32 r11, r6, r9, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r27, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r9, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r27, r19, 0x08;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r27, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r9, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r27, r19, 0x40;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r27, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r9, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r27, r19, 0x04;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r11, r6, r9, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r27, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r9, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r27, r19, 0x80;\n\t"
		"lop3.b32 r25, r25, r11, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r9, r27}    C = {r18, r19}    D = {r25, r10}
		/*
		* |------------------------[ROUND 14.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           { r9, r27}           |
		* |            v[ 5]            |           {r24, r29}           |
		* |            v[ 6]            |           {r26, r31}           |
		* |            v[ 7]            |           {r28, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           { r8, r13}           |
		* |            v[13]            |           {r48, r15}           |
		* |            v[14]            |           {r25, r10}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r11, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r13}
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r27;\n\t"
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r13}
		"xor.b32 r11, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r0, r11, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r13}
		"xor.b32 r8, r8, r0;\n\t"
		"xor.b32 r13, r13, r1;\n\t"
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r13}
		"shf.r.wrap.b32 r11, r8, r13, 60;\n\t"
		"shf.r.wrap.b32 r8, r13, r8, 60;\n\t"
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r11}
		"add.cc.u32 r16, r16, r8;\n\t"
		"addc.u32 r17, r17, r11;\n\t"
		// A = {r0, r1}    B = {r9, r27}    C = {r16, r17}    D = {r8, r11}
		"xor.b32 r9, r9, r16;\n\t"
		"xor.b32 r27, r27, r17;\n\t"
		"shf.r.wrap.b32 r13, r9, r27, 43;\n\t"
		"shf.r.wrap.b32 r9, r27, r9, 43;\n\t"
		// A = {r0, r1}    B = {r9, r13}    C = {r16, r17}    D = {r8, r11}
		"add.cc.u32 r0, r0, r9;\n\t"
		"addc.u32 r1, r1, r13;\n\t"
		// A = {r0, r1}    B = {r9, r13}    C = {r16, r17}    D = {r8, r11}
		"xor.b32 r27, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r9, r13}    C = {r16, r17}    D = {r8, r11}
		"xor.b32 r8, r8, r0;\n\t"
		"xor.b32 r11, r11, r1;\n\t"
		"shf.r.wrap.b32 r27, r8, r11, 5;\n\t"
		"shf.r.wrap.b32 r8, r11, r8, 5;\n\t"
		// A = {r0, r1}    B = {r9, r13}    C = {r16, r17}    D = {r27, r8}
		"add.cc.u32 r16, r16, r27;\n\t"
		"addc.u32 r17, r17, r8;\n\t"
		// A = {r0, r1}    B = {r9, r13}    C = {r16, r17}    D = {r27, r8}
		"xor.b32 r9, r9, r16;\n\t"
		"xor.b32 r13, r13, r17;\n\t"
		"shf.r.wrap.b32 r11, r9, r13, 18;\n\t"
		"shf.r.wrap.b32 r9, r13, r9, 18;\n\t"
		// A = {r0, r1}    B = {r11, r9}    C = {r16, r17}    D = {r27, r8}
		"lop3.b32 r13, r0, r11, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r9, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r11, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r9, r17, 0x08;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r9, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r11, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r9, r17, 0x40;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r9, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r11, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r9, r17, 0x04;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		"lop3.b32 r13, r0, r11, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r9, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r11, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r9, r17, 0x80;\n\t"
		"lop3.b32 r27, r27, r13, r50, 0x1E;\n\t"
		"lop3.b32 r8, r8, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r11, r9}    C = {r16, r17}    D = {r27, r8}
		/*
		* |------------------------[ROUND 14.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r24, r29}           |
		* |            v[ 6]            |           {r26, r31}           |
		* |            v[ 7]            |           {r28, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27,  r8}           |
		* |            v[13]            |           {r48, r15}           |
		* |            v[14]            |           {r25, r10}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r13, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r15}
		"add.cc.u32 r2, r2, r24;\n\t"
		"addc.u32 r3, r3, r29;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r15}
		"xor.b32 r13, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r13, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r15}
		"xor.b32 r48, r48, r2;\n\t"
		"xor.b32 r15, r15, r3;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r15}
		"shf.r.wrap.b32 r13, r48, r15, 60;\n\t"
		"shf.r.wrap.b32 r48, r15, r48, 60;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r13}
		"add.cc.u32 r18, r18, r48;\n\t"
		"addc.u32 r19, r19, r13;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r18, r19}    D = {r48, r13}
		"xor.b32 r24, r24, r18;\n\t"
		"xor.b32 r29, r29, r19;\n\t"
		"shf.r.wrap.b32 r15, r24, r29, 43;\n\t"
		"shf.r.wrap.b32 r24, r29, r24, 43;\n\t"
		// A = {r2, r3}    B = {r24, r15}    C = {r18, r19}    D = {r48, r13}
		"add.cc.u32 r2, r2, r24;\n\t"
		"addc.u32 r3, r3, r15;\n\t"
		// A = {r2, r3}    B = {r24, r15}    C = {r18, r19}    D = {r48, r13}
		"xor.b32 r29, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r24, r15}    C = {r18, r19}    D = {r48, r13}
		"xor.b32 r48, r48, r2;\n\t"
		"xor.b32 r13, r13, r3;\n\t"
		"shf.r.wrap.b32 r29, r48, r13, 5;\n\t"
		"shf.r.wrap.b32 r48, r13, r48, 5;\n\t"
		// A = {r2, r3}    B = {r24, r15}    C = {r18, r19}    D = {r29, r48}
		"add.cc.u32 r18, r18, r29;\n\t"
		"addc.u32 r19, r19, r48;\n\t"
		// A = {r2, r3}    B = {r24, r15}    C = {r18, r19}    D = {r29, r48}
		"xor.b32 r24, r24, r18;\n\t"
		"xor.b32 r15, r15, r19;\n\t"
		"shf.r.wrap.b32 r13, r24, r15, 18;\n\t"
		"shf.r.wrap.b32 r24, r15, r24, 18;\n\t"
		// A = {r2, r3}    B = {r13, r24}    C = {r18, r19}    D = {r29, r48}
		"lop3.b32 r15, r2, r13, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r24, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r13, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r24, r19, 0x08;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r24, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r13, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r24, r19, 0x40;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r24, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r13, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r24, r19, 0x04;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		"lop3.b32 r15, r2, r13, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r24, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r13, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r24, r19, 0x80;\n\t"
		"lop3.b32 r29, r29, r15, r50, 0x1E;\n\t"
		"lop3.b32 r48, r48, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r13, r24}    C = {r18, r19}    D = {r29, r48}
		/*
		* |------------------------[ROUND 14.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r13, r24}           |
		* |            v[ 6]            |           {r26, r31}           |
		* |            v[ 7]            |           {r28, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27,  r8}           |
		* |            v[13]            |           {r29, r48}           |
		* |            v[14]            |           {r25, r10}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r15, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r10}
		"add.cc.u32 r4, r4, r26;\n\t"
		"addc.u32 r5, r5, r31;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r10}
		"xor.b32 r15, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r4, r15, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r10}
		"xor.b32 r25, r25, r4;\n\t"
		"xor.b32 r10, r10, r5;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r10}
		"shf.r.wrap.b32 r15, r25, r10, 60;\n\t"
		"shf.r.wrap.b32 r25, r10, r25, 60;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r15}
		"add.cc.u32 r20, r20, r25;\n\t"
		"addc.u32 r21, r21, r15;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r20, r21}    D = {r25, r15}
		"xor.b32 r26, r26, r20;\n\t"
		"xor.b32 r31, r31, r21;\n\t"
		"shf.r.wrap.b32 r10, r26, r31, 43;\n\t"
		"shf.r.wrap.b32 r26, r31, r26, 43;\n\t"
		// A = {r4, r5}    B = {r26, r10}    C = {r20, r21}    D = {r25, r15}
		"add.cc.u32 r4, r4, r26;\n\t"
		"addc.u32 r5, r5, r10;\n\t"
		// A = {r4, r5}    B = {r26, r10}    C = {r20, r21}    D = {r25, r15}
		"xor.b32 r31, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r26, r10}    C = {r20, r21}    D = {r25, r15}
		"xor.b32 r25, r25, r4;\n\t"
		"xor.b32 r15, r15, r5;\n\t"
		"shf.r.wrap.b32 r31, r25, r15, 5;\n\t"
		"shf.r.wrap.b32 r25, r15, r25, 5;\n\t"
		// A = {r4, r5}    B = {r26, r10}    C = {r20, r21}    D = {r31, r25}
		"add.cc.u32 r20, r20, r31;\n\t"
		"addc.u32 r21, r21, r25;\n\t"
		// A = {r4, r5}    B = {r26, r10}    C = {r20, r21}    D = {r31, r25}
		"xor.b32 r26, r26, r20;\n\t"
		"xor.b32 r10, r10, r21;\n\t"
		"shf.r.wrap.b32 r15, r26, r10, 18;\n\t"
		"shf.r.wrap.b32 r26, r10, r26, 18;\n\t"
		// A = {r4, r5}    B = {r15, r26}    C = {r20, r21}    D = {r31, r25}
		"lop3.b32 r10, r4, r15, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r26, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r15, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r26, r21, 0x08;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r26, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r15, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r26, r21, 0x40;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r26, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r15, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r26, r21, 0x04;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		"lop3.b32 r10, r4, r15, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r26, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r15, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r26, r21, 0x80;\n\t"
		"lop3.b32 r31, r31, r10, r50, 0x1E;\n\t"
		"lop3.b32 r25, r25, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r15, r26}    C = {r20, r21}    D = {r31, r25}
		/*
		* |------------------------[ROUND 14.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r13, r24}           |
		* |            v[ 6]            |           {r15, r26}           |
		* |            v[ 7]            |           {r28, r30}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27,  r8}           |
		* |            v[13]            |           {r29, r48}           |
		* |            v[14]            |           {r31, r25}           |
		* |            v[15]            |           {r14, r12}           |
		* |            temp0            |           {r10, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r12}
		"add.cc.u32 r6, r6, r28;\n\t"
		"addc.u32 r7, r7, r30;\n\t"
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r12}
		"xor.b32 r10, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r6, r10, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r12}
		"xor.b32 r14, r14, r6;\n\t"
		"xor.b32 r12, r12, r7;\n\t"
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r12}
		"shf.r.wrap.b32 r10, r14, r12, 60;\n\t"
		"shf.r.wrap.b32 r14, r12, r14, 60;\n\t"
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r10}
		"add.cc.u32 r22, r22, r14;\n\t"
		"addc.u32 r23, r23, r10;\n\t"
		// A = {r6, r7}    B = {r28, r30}    C = {r22, r23}    D = {r14, r10}
		"xor.b32 r28, r28, r22;\n\t"
		"xor.b32 r30, r30, r23;\n\t"
		"shf.r.wrap.b32 r12, r28, r30, 43;\n\t"
		"shf.r.wrap.b32 r28, r30, r28, 43;\n\t"
		// A = {r6, r7}    B = {r28, r12}    C = {r22, r23}    D = {r14, r10}
		"add.cc.u32 r6, r6, r28;\n\t"
		"addc.u32 r7, r7, r12;\n\t"
		// A = {r6, r7}    B = {r28, r12}    C = {r22, r23}    D = {r14, r10}
		"xor.b32 r30, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r6, r6, r30;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r28, r12}    C = {r22, r23}    D = {r14, r10}
		"xor.b32 r14, r14, r6;\n\t"
		"xor.b32 r10, r10, r7;\n\t"
		"shf.r.wrap.b32 r30, r14, r10, 5;\n\t"
		"shf.r.wrap.b32 r14, r10, r14, 5;\n\t"
		// A = {r6, r7}    B = {r28, r12}    C = {r22, r23}    D = {r30, r14}
		"add.cc.u32 r22, r22, r30;\n\t"
		"addc.u32 r23, r23, r14;\n\t"
		// A = {r6, r7}    B = {r28, r12}    C = {r22, r23}    D = {r30, r14}
		"xor.b32 r28, r28, r22;\n\t"
		"xor.b32 r12, r12, r23;\n\t"
		"shf.r.wrap.b32 r10, r28, r12, 18;\n\t"
		"shf.r.wrap.b32 r28, r12, r28, 18;\n\t"
		// A = {r6, r7}    B = {r10, r28}    C = {r22, r23}    D = {r30, r14}
		"lop3.b32 r12, r6, r10, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r28, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r10, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r28, r23, 0x08;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r28, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r10, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r28, r23, 0x40;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r28, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r10, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r28, r23, 0x04;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		"lop3.b32 r12, r6, r10, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r28, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r10, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r28, r23, 0x80;\n\t"
		"lop3.b32 r30, r30, r12, r50, 0x1E;\n\t"
		"lop3.b32 r14, r14, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r10, r28}    C = {r22, r23}    D = {r30, r14}
		/*
		* |------------------------[ROUND 14.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r13, r24}           |
		* |            v[ 6]            |           {r15, r26}           |
		* |            v[ 7]            |           {r10, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27,  r8}           |
		* |            v[13]            |           {r29, r48}           |
		* |            v[14]            |           {r31, r25}           |
		* |            v[15]            |           {r30, r14}           |
		* |            temp0            |           {r12, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r14}
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r24;\n\t"
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r14}
		"xor.b32 r12, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r0, r12, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r14}
		"xor.b32 r30, r30, r0;\n\t"
		"xor.b32 r14, r14, r1;\n\t"
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r14}
		"shf.r.wrap.b32 r12, r30, r14, 60;\n\t"
		"shf.r.wrap.b32 r30, r14, r30, 60;\n\t"
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r12}
		"add.cc.u32 r20, r20, r30;\n\t"
		"addc.u32 r21, r21, r12;\n\t"
		// A = {r0, r1}    B = {r13, r24}    C = {r20, r21}    D = {r30, r12}
		"xor.b32 r13, r13, r20;\n\t"
		"xor.b32 r24, r24, r21;\n\t"
		"shf.r.wrap.b32 r14, r13, r24, 43;\n\t"
		"shf.r.wrap.b32 r13, r24, r13, 43;\n\t"
		// A = {r0, r1}    B = {r13, r14}    C = {r20, r21}    D = {r30, r12}
		"add.cc.u32 r0, r0, r13;\n\t"
		"addc.u32 r1, r1, r14;\n\t"
		// A = {r0, r1}    B = {r13, r14}    C = {r20, r21}    D = {r30, r12}
		"xor.b32 r24, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r0, r0, r24;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r13, r14}    C = {r20, r21}    D = {r30, r12}
		"xor.b32 r30, r30, r0;\n\t"
		"xor.b32 r12, r12, r1;\n\t"
		"shf.r.wrap.b32 r24, r30, r12, 5;\n\t"
		"shf.r.wrap.b32 r30, r12, r30, 5;\n\t"
		// A = {r0, r1}    B = {r13, r14}    C = {r20, r21}    D = {r24, r30}
		"add.cc.u32 r20, r20, r24;\n\t"
		"addc.u32 r21, r21, r30;\n\t"
		// A = {r0, r1}    B = {r13, r14}    C = {r20, r21}    D = {r24, r30}
		"xor.b32 r13, r13, r20;\n\t"
		"xor.b32 r14, r14, r21;\n\t"
		"shf.r.wrap.b32 r12, r13, r14, 18;\n\t"
		"shf.r.wrap.b32 r13, r14, r13, 18;\n\t"
		// A = {r0, r1}    B = {r12, r13}    C = {r20, r21}    D = {r24, r30}
		"lop3.b32 r14, r0, r12, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r13, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r12, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r13, r21, 0x08;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r13, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r12, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r13, r21, 0x40;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r13, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r12, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r13, r21, 0x04;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		"lop3.b32 r14, r0, r12, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r13, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r12, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r13, r21, 0x80;\n\t"
		"lop3.b32 r24, r24, r14, r50, 0x1E;\n\t"
		"lop3.b32 r30, r30, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r12, r13}    C = {r20, r21}    D = {r24, r30}
		/*
		* |------------------------[ROUND 14.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r12, r13}           |
		* |            v[ 6]            |           {r15, r26}           |
		* |            v[ 7]            |           {r10, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r27,  r8}           |
		* |            v[13]            |           {r29, r48}           |
		* |            v[14]            |           {r31, r25}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r14, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r8}
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r26;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r8}
		"xor.b32 r14, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r2, r14, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r8}
		"xor.b32 r27, r27, r2;\n\t"
		"xor.b32 r8, r8, r3;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r8}
		"shf.r.wrap.b32 r14, r27, r8, 60;\n\t"
		"shf.r.wrap.b32 r27, r8, r27, 60;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r14}
		"add.cc.u32 r22, r22, r27;\n\t"
		"addc.u32 r23, r23, r14;\n\t"
		// A = {r2, r3}    B = {r15, r26}    C = {r22, r23}    D = {r27, r14}
		"xor.b32 r15, r15, r22;\n\t"
		"xor.b32 r26, r26, r23;\n\t"
		"shf.r.wrap.b32 r8, r15, r26, 43;\n\t"
		"shf.r.wrap.b32 r15, r26, r15, 43;\n\t"
		// A = {r2, r3}    B = {r15, r8}    C = {r22, r23}    D = {r27, r14}
		"add.cc.u32 r2, r2, r15;\n\t"
		"addc.u32 r3, r3, r8;\n\t"
		// A = {r2, r3}    B = {r15, r8}    C = {r22, r23}    D = {r27, r14}
		"xor.b32 r26, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r2, r2, r26;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r15, r8}    C = {r22, r23}    D = {r27, r14}
		"xor.b32 r27, r27, r2;\n\t"
		"xor.b32 r14, r14, r3;\n\t"
		"shf.r.wrap.b32 r26, r27, r14, 5;\n\t"
		"shf.r.wrap.b32 r27, r14, r27, 5;\n\t"
		// A = {r2, r3}    B = {r15, r8}    C = {r22, r23}    D = {r26, r27}
		"add.cc.u32 r22, r22, r26;\n\t"
		"addc.u32 r23, r23, r27;\n\t"
		// A = {r2, r3}    B = {r15, r8}    C = {r22, r23}    D = {r26, r27}
		"xor.b32 r15, r15, r22;\n\t"
		"xor.b32 r8, r8, r23;\n\t"
		"shf.r.wrap.b32 r14, r15, r8, 18;\n\t"
		"shf.r.wrap.b32 r15, r8, r15, 18;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r22, r23}    D = {r26, r27}
		"lop3.b32 r8, r2, r14, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r15, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r14, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r15, r23, 0x08;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r15, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r14, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r15, r23, 0x40;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r15, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r14, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r15, r23, 0x04;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		"lop3.b32 r8, r2, r14, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r15, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r14, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r15, r23, 0x80;\n\t"
		"lop3.b32 r26, r26, r8, r50, 0x1E;\n\t"
		"lop3.b32 r27, r27, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r14, r15}    C = {r22, r23}    D = {r26, r27}
		/*
		* |------------------------[ROUND 14.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r12, r13}           |
		* |            v[ 6]            |           {r14, r15}           |
		* |            v[ 7]            |           {r10, r28}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r27}           |
		* |            v[13]            |           {r29, r48}           |
		* |            v[14]            |           {r31, r25}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           { r8, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r48}
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r28;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r48}
		"xor.b32 r8, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r4, r8, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r48}
		"xor.b32 r29, r29, r4;\n\t"
		"xor.b32 r48, r48, r5;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r48}
		"shf.r.wrap.b32 r8, r29, r48, 60;\n\t"
		"shf.r.wrap.b32 r29, r48, r29, 60;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r8}
		"add.cc.u32 r16, r16, r29;\n\t"
		"addc.u32 r17, r17, r8;\n\t"
		// A = {r4, r5}    B = {r10, r28}    C = {r16, r17}    D = {r29, r8}
		"xor.b32 r10, r10, r16;\n\t"
		"xor.b32 r28, r28, r17;\n\t"
		"shf.r.wrap.b32 r48, r10, r28, 43;\n\t"
		"shf.r.wrap.b32 r10, r28, r10, 43;\n\t"
		// A = {r4, r5}    B = {r10, r48}    C = {r16, r17}    D = {r29, r8}
		"add.cc.u32 r4, r4, r10;\n\t"
		"addc.u32 r5, r5, r48;\n\t"
		// A = {r4, r5}    B = {r10, r48}    C = {r16, r17}    D = {r29, r8}
		"xor.b32 r28, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r4, r4, r28;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r10, r48}    C = {r16, r17}    D = {r29, r8}
		"xor.b32 r29, r29, r4;\n\t"
		"xor.b32 r8, r8, r5;\n\t"
		"shf.r.wrap.b32 r28, r29, r8, 5;\n\t"
		"shf.r.wrap.b32 r29, r8, r29, 5;\n\t"
		// A = {r4, r5}    B = {r10, r48}    C = {r16, r17}    D = {r28, r29}
		"add.cc.u32 r16, r16, r28;\n\t"
		"addc.u32 r17, r17, r29;\n\t"
		// A = {r4, r5}    B = {r10, r48}    C = {r16, r17}    D = {r28, r29}
		"xor.b32 r10, r10, r16;\n\t"
		"xor.b32 r48, r48, r17;\n\t"
		"shf.r.wrap.b32 r8, r10, r48, 18;\n\t"
		"shf.r.wrap.b32 r10, r48, r10, 18;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r16, r17}    D = {r28, r29}
		"lop3.b32 r48, r4, r8, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r10, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r8, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r10, r17, 0x08;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r10, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r8, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r10, r17, 0x40;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r10, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r8, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r10, r17, 0x04;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		"lop3.b32 r48, r4, r8, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r10, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r8, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r10, r17, 0x80;\n\t"
		"lop3.b32 r28, r28, r48, r50, 0x1E;\n\t"
		"lop3.b32 r29, r29, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r8, r10}    C = {r16, r17}    D = {r28, r29}
		/*
		* |------------------------[ROUND 14.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r11,  r9}           |
		* |            v[ 5]            |           {r12, r13}           |
		* |            v[ 6]            |           {r14, r15}           |
		* |            v[ 7]            |           { r8, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r27}           |
		* |            v[13]            |           {r28, r29}           |
		* |            v[14]            |           {r31, r25}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r48, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r25}
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r9;\n\t"
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r25}
		"xor.b32 r48, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r6, r48, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r25}
		"xor.b32 r31, r31, r6;\n\t"
		"xor.b32 r25, r25, r7;\n\t"
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r25}
		"shf.r.wrap.b32 r48, r31, r25, 60;\n\t"
		"shf.r.wrap.b32 r31, r25, r31, 60;\n\t"
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r48}
		"add.cc.u32 r18, r18, r31;\n\t"
		"addc.u32 r19, r19, r48;\n\t"
		// A = {r6, r7}    B = {r11, r9}    C = {r18, r19}    D = {r31, r48}
		"xor.b32 r11, r11, r18;\n\t"
		"xor.b32 r9, r9, r19;\n\t"
		"shf.r.wrap.b32 r25, r11, r9, 43;\n\t"
		"shf.r.wrap.b32 r11, r9, r11, 43;\n\t"
		// A = {r6, r7}    B = {r11, r25}    C = {r18, r19}    D = {r31, r48}
		"add.cc.u32 r6, r6, r11;\n\t"
		"addc.u32 r7, r7, r25;\n\t"
		// A = {r6, r7}    B = {r11, r25}    C = {r18, r19}    D = {r31, r48}
		"xor.b32 r9, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r6, r9;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r11, r25}    C = {r18, r19}    D = {r31, r48}
		"xor.b32 r31, r31, r6;\n\t"
		"xor.b32 r48, r48, r7;\n\t"
		"shf.r.wrap.b32 r9, r31, r48, 5;\n\t"
		"shf.r.wrap.b32 r31, r48, r31, 5;\n\t"
		// A = {r6, r7}    B = {r11, r25}    C = {r18, r19}    D = {r9, r31}
		"add.cc.u32 r18, r18, r9;\n\t"
		"addc.u32 r19, r19, r31;\n\t"
		// A = {r6, r7}    B = {r11, r25}    C = {r18, r19}    D = {r9, r31}
		"xor.b32 r11, r11, r18;\n\t"
		"xor.b32 r25, r25, r19;\n\t"
		"shf.r.wrap.b32 r48, r11, r25, 18;\n\t"
		"shf.r.wrap.b32 r11, r25, r11, 18;\n\t"
		// A = {r6, r7}    B = {r48, r11}    C = {r18, r19}    D = {r9, r31}
		"lop3.b32 r25, r6, r48, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r11, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r48, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r11, r19, 0x08;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r11, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r48, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r11, r19, 0x40;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r11, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r48, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r11, r19, 0x04;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		"lop3.b32 r25, r6, r48, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r11, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r48, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r11, r19, 0x80;\n\t"
		"lop3.b32 r9, r9, r25, r50, 0x1E;\n\t"
		"lop3.b32 r31, r31, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r48, r11}    C = {r18, r19}    D = {r9, r31}
		/*
		* |------------------------[ROUND 15.0]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r48, r11}           |
		* |            v[ 5]            |           {r12, r13}           |
		* |            v[ 6]            |           {r14, r15}           |
		* |            v[ 7]            |           { r8, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r26, r27}           |
		* |            v[13]            |           {r28, r29}           |
		* |            v[14]            |           { r9, r31}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r25, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r27}
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r11;\n\t"
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r27}
		"xor.b32 r25, 0x00, 0xF92CA000;\n\t"
		"xor.b32 r49, 0x00, 0xBAFCD004;\n\t"
		"add.cc.u32 r0, r25, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r27}
		"xor.b32 r26, r26, r0;\n\t"
		"xor.b32 r27, r27, r1;\n\t"
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r27}
		"shf.r.wrap.b32 r25, r26, r27, 60;\n\t"
		"shf.r.wrap.b32 r26, r27, r26, 60;\n\t"
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r25}
		"add.cc.u32 r16, r16, r26;\n\t"
		"addc.u32 r17, r17, r25;\n\t"
		// A = {r0, r1}    B = {r48, r11}    C = {r16, r17}    D = {r26, r25}
		"xor.b32 r48, r48, r16;\n\t"
		"xor.b32 r11, r11, r17;\n\t"
		"shf.r.wrap.b32 r27, r48, r11, 43;\n\t"
		"shf.r.wrap.b32 r48, r11, r48, 43;\n\t"
		// A = {r0, r1}    B = {r48, r27}    C = {r16, r17}    D = {r26, r25}
		"add.cc.u32 r0, r0, r48;\n\t"
		"addc.u32 r1, r1, r27;\n\t"
		// A = {r0, r1}    B = {r48, r27}    C = {r16, r17}    D = {r26, r25}
		"xor.b32 r11, r36, 0xAE9F9000;\n\t"
		"xor.b32 r49, r37, 0xA47B39A2;\n\t"
		"add.cc.u32 r0, r0, r11;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r48, r27}    C = {r16, r17}    D = {r26, r25}
		"xor.b32 r26, r26, r0;\n\t"
		"xor.b32 r25, r25, r1;\n\t"
		"shf.r.wrap.b32 r11, r26, r25, 5;\n\t"
		"shf.r.wrap.b32 r26, r25, r26, 5;\n\t"
		// A = {r0, r1}    B = {r48, r27}    C = {r16, r17}    D = {r11, r26}
		"add.cc.u32 r16, r16, r11;\n\t"
		"addc.u32 r17, r17, r26;\n\t"
		// A = {r0, r1}    B = {r48, r27}    C = {r16, r17}    D = {r11, r26}
		"xor.b32 r48, r48, r16;\n\t"
		"xor.b32 r27, r27, r17;\n\t"
		"shf.r.wrap.b32 r25, r48, r27, 18;\n\t"
		"shf.r.wrap.b32 r48, r27, r48, 18;\n\t"
		// A = {r0, r1}    B = {r25, r48}    C = {r16, r17}    D = {r11, r26}
		"lop3.b32 r27, r0, r25, r16, 0x01;\n\t"
		"lop3.b32 r49, r1, r48, r17, 0x01;\n\t"
		"lop3.b32 r50, r0, r25, r16, 0x08;\n\t"
		"lop3.b32 r51, r1, r48, r17, 0x08;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r16, 0x20;\n\t"
		"lop3.b32 r49, r1, r48, r17, 0x20;\n\t"
		"lop3.b32 r50, r0, r25, r16, 0x40;\n\t"
		"lop3.b32 r51, r1, r48, r17, 0x40;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r16, 0x02;\n\t"
		"lop3.b32 r49, r1, r48, r17, 0x02;\n\t"
		"lop3.b32 r50, r0, r25, r16, 0x04;\n\t"
		"lop3.b32 r51, r1, r48, r17, 0x04;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		"lop3.b32 r27, r0, r25, r16, 0x10;\n\t"
		"lop3.b32 r49, r1, r48, r17, 0x10;\n\t"
		"lop3.b32 r50, r0, r25, r16, 0x80;\n\t"
		"lop3.b32 r51, r1, r48, r17, 0x80;\n\t"
		"lop3.b32 r11, r11, r27, r50, 0x1E;\n\t"
		"lop3.b32 r26, r26, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r25, r48}    C = {r16, r17}    D = {r11, r26}
		/*
		* |------------------------[ROUND 15.1]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r12, r13}           |
		* |            v[ 6]            |           {r14, r15}           |
		* |            v[ 7]            |           { r8, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r26}           |
		* |            v[13]            |           {r28, r29}           |
		* |            v[14]            |           { r9, r31}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r27, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r29}
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r13;\n\t"
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r29}
		"xor.b32 r27, 0x00, 0x9632463E;\n\t"
		"xor.b32 r49, 0x00, 0x2FE452DA;\n\t"
		"add.cc.u32 r2, r27, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r29}
		"xor.b32 r28, r28, r2;\n\t"
		"xor.b32 r29, r29, r3;\n\t"
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r29}
		"shf.r.wrap.b32 r27, r28, r29, 60;\n\t"
		"shf.r.wrap.b32 r28, r29, r28, 60;\n\t"
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r27}
		"add.cc.u32 r18, r18, r28;\n\t"
		"addc.u32 r19, r19, r27;\n\t"
		// A = {r2, r3}    B = {r12, r13}    C = {r18, r19}    D = {r28, r27}
		"xor.b32 r12, r12, r18;\n\t"
		"xor.b32 r13, r13, r19;\n\t"
		"shf.r.wrap.b32 r29, r12, r13, 43;\n\t"
		"shf.r.wrap.b32 r12, r13, r12, 43;\n\t"
		// A = {r2, r3}    B = {r12, r29}    C = {r18, r19}    D = {r28, r27}
		"add.cc.u32 r2, r2, r12;\n\t"
		"addc.u32 r3, r3, r29;\n\t"
		// A = {r2, r3}    B = {r12, r29}    C = {r18, r19}    D = {r28, r27}
		"xor.b32 r13, r44, 0x4DC879DD;\n\t"
		"xor.b32 r49, r45, 0x4606AD36;\n\t"
		"add.cc.u32 r2, r2, r13;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r12, r29}    C = {r18, r19}    D = {r28, r27}
		"xor.b32 r28, r28, r2;\n\t"
		"xor.b32 r27, r27, r3;\n\t"
		"shf.r.wrap.b32 r13, r28, r27, 5;\n\t"
		"shf.r.wrap.b32 r28, r27, r28, 5;\n\t"
		// A = {r2, r3}    B = {r12, r29}    C = {r18, r19}    D = {r13, r28}
		"add.cc.u32 r18, r18, r13;\n\t"
		"addc.u32 r19, r19, r28;\n\t"
		// A = {r2, r3}    B = {r12, r29}    C = {r18, r19}    D = {r13, r28}
		"xor.b32 r12, r12, r18;\n\t"
		"xor.b32 r29, r29, r19;\n\t"
		"shf.r.wrap.b32 r27, r12, r29, 18;\n\t"
		"shf.r.wrap.b32 r12, r29, r12, 18;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r18, r19}    D = {r13, r28}
		"lop3.b32 r29, r2, r27, r18, 0x01;\n\t"
		"lop3.b32 r49, r3, r12, r19, 0x01;\n\t"
		"lop3.b32 r50, r2, r27, r18, 0x08;\n\t"
		"lop3.b32 r51, r3, r12, r19, 0x08;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r18, 0x20;\n\t"
		"lop3.b32 r49, r3, r12, r19, 0x20;\n\t"
		"lop3.b32 r50, r2, r27, r18, 0x40;\n\t"
		"lop3.b32 r51, r3, r12, r19, 0x40;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r18, 0x02;\n\t"
		"lop3.b32 r49, r3, r12, r19, 0x02;\n\t"
		"lop3.b32 r50, r2, r27, r18, 0x04;\n\t"
		"lop3.b32 r51, r3, r12, r19, 0x04;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		"lop3.b32 r29, r2, r27, r18, 0x10;\n\t"
		"lop3.b32 r49, r3, r12, r19, 0x10;\n\t"
		"lop3.b32 r50, r2, r27, r18, 0x80;\n\t"
		"lop3.b32 r51, r3, r12, r19, 0x80;\n\t"
		"lop3.b32 r13, r13, r29, r50, 0x1E;\n\t"
		"lop3.b32 r28, r28, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r27, r12}    C = {r18, r19}    D = {r13, r28}
		/*
		* |------------------------[ROUND 15.2]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r27, r12}           |
		* |            v[ 6]            |           {r14, r15}           |
		* |            v[ 7]            |           { r8, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r26}           |
		* |            v[13]            |           {r13, r28}           |
		* |            v[14]            |           { r9, r31}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r29, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r31}
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r15;\n\t"
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r31}
		"xor.b32 r29, 0x00, 0x6226F800;\n\t"
		"xor.b32 r49, 0x00, 0x98A7B549;\n\t"
		"add.cc.u32 r4, r29, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r31}
		"xor.b32 r9, r9, r4;\n\t"
		"xor.b32 r31, r31, r5;\n\t"
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r31}
		"shf.r.wrap.b32 r29, r9, r31, 60;\n\t"
		"shf.r.wrap.b32 r9, r31, r9, 60;\n\t"
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r29}
		"add.cc.u32 r20, r20, r9;\n\t"
		"addc.u32 r21, r21, r29;\n\t"
		// A = {r4, r5}    B = {r14, r15}    C = {r20, r21}    D = {r9, r29}
		"xor.b32 r14, r14, r20;\n\t"
		"xor.b32 r15, r15, r21;\n\t"
		"shf.r.wrap.b32 r31, r14, r15, 43;\n\t"
		"shf.r.wrap.b32 r14, r15, r14, 43;\n\t"
		// A = {r4, r5}    B = {r14, r31}    C = {r20, r21}    D = {r9, r29}
		"add.cc.u32 r4, r4, r14;\n\t"
		"addc.u32 r5, r5, r31;\n\t"
		// A = {r4, r5}    B = {r14, r31}    C = {r20, r21}    D = {r9, r29}
		"xor.b32 r15, r32, 0xD489E800;\n\t"
		"xor.b32 r49, r33, 0xA51B6A89;\n\t"
		"add.cc.u32 r4, r4, r15;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r14, r31}    C = {r20, r21}    D = {r9, r29}
		"xor.b32 r9, r9, r4;\n\t"
		"xor.b32 r29, r29, r5;\n\t"
		"shf.r.wrap.b32 r15, r9, r29, 5;\n\t"
		"shf.r.wrap.b32 r9, r29, r9, 5;\n\t"
		// A = {r4, r5}    B = {r14, r31}    C = {r20, r21}    D = {r15, r9}
		"add.cc.u32 r20, r20, r15;\n\t"
		"addc.u32 r21, r21, r9;\n\t"
		// A = {r4, r5}    B = {r14, r31}    C = {r20, r21}    D = {r15, r9}
		"xor.b32 r14, r14, r20;\n\t"
		"xor.b32 r31, r31, r21;\n\t"
		"shf.r.wrap.b32 r29, r14, r31, 18;\n\t"
		"shf.r.wrap.b32 r14, r31, r14, 18;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r20, r21}    D = {r15, r9}
		"lop3.b32 r31, r4, r29, r20, 0x01;\n\t"
		"lop3.b32 r49, r5, r14, r21, 0x01;\n\t"
		"lop3.b32 r50, r4, r29, r20, 0x08;\n\t"
		"lop3.b32 r51, r5, r14, r21, 0x08;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r20, 0x20;\n\t"
		"lop3.b32 r49, r5, r14, r21, 0x20;\n\t"
		"lop3.b32 r50, r4, r29, r20, 0x40;\n\t"
		"lop3.b32 r51, r5, r14, r21, 0x40;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r20, 0x02;\n\t"
		"lop3.b32 r49, r5, r14, r21, 0x02;\n\t"
		"lop3.b32 r50, r4, r29, r20, 0x04;\n\t"
		"lop3.b32 r51, r5, r14, r21, 0x04;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		"lop3.b32 r31, r4, r29, r20, 0x10;\n\t"
		"lop3.b32 r49, r5, r14, r21, 0x10;\n\t"
		"lop3.b32 r50, r4, r29, r20, 0x80;\n\t"
		"lop3.b32 r51, r5, r14, r21, 0x80;\n\t"
		"lop3.b32 r15, r15, r31, r50, 0x1E;\n\t"
		"lop3.b32 r9, r9, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r29, r14}    C = {r20, r21}    D = {r15, r9}
		/*
		* |------------------------[ROUND 15.3]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r27, r12}           |
		* |            v[ 6]            |           {r29, r14}           |
		* |            v[ 7]            |           { r8, r10}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r26}           |
		* |            v[13]            |           {r13, r28}           |
		* |            v[14]            |           {r15,  r9}           |
		* |            v[15]            |           {r24, r30}           |
		* |            temp0            |           {r31, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r30}
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r10;\n\t"
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r30}
		"xor.b32 r31, r38, 0xE77E6488;\n\t"
		"xor.b32 r49, r39, 0x0C0EFA33;\n\t"
		"add.cc.u32 r6, r31, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r30}
		"xor.b32 r24, r24, r6;\n\t"
		"xor.b32 r30, r30, r7;\n\t"
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r30}
		"shf.r.wrap.b32 r31, r24, r30, 60;\n\t"
		"shf.r.wrap.b32 r24, r30, r24, 60;\n\t"
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r31}
		"add.cc.u32 r22, r22, r24;\n\t"
		"addc.u32 r23, r23, r31;\n\t"
		// A = {r6, r7}    B = {r8, r10}    C = {r22, r23}    D = {r24, r31}
		"xor.b32 r8, r8, r22;\n\t"
		"xor.b32 r10, r10, r23;\n\t"
		"shf.r.wrap.b32 r30, r8, r10, 43;\n\t"
		"shf.r.wrap.b32 r8, r10, r8, 43;\n\t"
		// A = {r6, r7}    B = {r8, r30}    C = {r22, r23}    D = {r24, r31}
		"add.cc.u32 r6, r6, r8;\n\t"
		"addc.u32 r7, r7, r30;\n\t"
		// A = {r6, r7}    B = {r8, r30}    C = {r22, r23}    D = {r24, r31}
		"xor.b32 r10, 0x00, 0x0C59EB1B;\n\t"
		"xor.b32 r49, 0x00, 0x531655D9;\n\t"
		"add.cc.u32 r6, r6, r10;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r8, r30}    C = {r22, r23}    D = {r24, r31}
		"xor.b32 r24, r24, r6;\n\t"
		"xor.b32 r31, r31, r7;\n\t"
		"shf.r.wrap.b32 r10, r24, r31, 5;\n\t"
		"shf.r.wrap.b32 r24, r31, r24, 5;\n\t"
		// A = {r6, r7}    B = {r8, r30}    C = {r22, r23}    D = {r10, r24}
		"add.cc.u32 r22, r22, r10;\n\t"
		"addc.u32 r23, r23, r24;\n\t"
		// A = {r6, r7}    B = {r8, r30}    C = {r22, r23}    D = {r10, r24}
		"xor.b32 r8, r8, r22;\n\t"
		"xor.b32 r30, r30, r23;\n\t"
		"shf.r.wrap.b32 r31, r8, r30, 18;\n\t"
		"shf.r.wrap.b32 r8, r30, r8, 18;\n\t"
		// A = {r6, r7}    B = {r31, r8}    C = {r22, r23}    D = {r10, r24}
		"lop3.b32 r30, r6, r31, r22, 0x01;\n\t"
		"lop3.b32 r49, r7, r8, r23, 0x01;\n\t"
		"lop3.b32 r50, r6, r31, r22, 0x08;\n\t"
		"lop3.b32 r51, r7, r8, r23, 0x08;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r22, 0x20;\n\t"
		"lop3.b32 r49, r7, r8, r23, 0x20;\n\t"
		"lop3.b32 r50, r6, r31, r22, 0x40;\n\t"
		"lop3.b32 r51, r7, r8, r23, 0x40;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r22, 0x02;\n\t"
		"lop3.b32 r49, r7, r8, r23, 0x02;\n\t"
		"lop3.b32 r50, r6, r31, r22, 0x04;\n\t"
		"lop3.b32 r51, r7, r8, r23, 0x04;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		"lop3.b32 r30, r6, r31, r22, 0x10;\n\t"
		"lop3.b32 r49, r7, r8, r23, 0x10;\n\t"
		"lop3.b32 r50, r6, r31, r22, 0x80;\n\t"
		"lop3.b32 r51, r7, r8, r23, 0x80;\n\t"
		"lop3.b32 r10, r10, r30, r50, 0x1E;\n\t"
		"lop3.b32 r24, r24, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r31, r8}    C = {r22, r23}    D = {r10, r24}
		/*
		* |------------------------[ROUND 15.4]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r27, r12}           |
		* |            v[ 6]            |           {r29, r14}           |
		* |            v[ 7]            |           {r31,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r26}           |
		* |            v[13]            |           {r13, r28}           |
		* |            v[14]            |           {r15,  r9}           |
		* |            v[15]            |           {r10, r24}           |
		* |            temp0            |           {r30, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r24}
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r12;\n\t"
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r24}
		"xor.b32 r30, 0x00, 0x839525E7;\n\t"
		"xor.b32 r49, 0x00, 0x64A39957;\n\t"
		"add.cc.u32 r0, r30, r0;\n\t"
		"addc.u32 r1, r49, r1;\n\t"
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r24}
		"xor.b32 r10, r10, r0;\n\t"
		"xor.b32 r24, r24, r1;\n\t"
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r24}
		"shf.r.wrap.b32 r30, r10, r24, 60;\n\t"
		"shf.r.wrap.b32 r10, r24, r10, 60;\n\t"
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r30}
		"add.cc.u32 r20, r20, r10;\n\t"
		"addc.u32 r21, r21, r30;\n\t"
		// A = {r0, r1}    B = {r27, r12}    C = {r20, r21}    D = {r10, r30}
		"xor.b32 r27, r27, r20;\n\t"
		"xor.b32 r12, r12, r21;\n\t"
		"shf.r.wrap.b32 r24, r27, r12, 43;\n\t"
		"shf.r.wrap.b32 r27, r12, r27, 43;\n\t"
		// A = {r0, r1}    B = {r27, r24}    C = {r20, r21}    D = {r10, r30}
		"add.cc.u32 r0, r0, r27;\n\t"
		"addc.u32 r1, r1, r24;\n\t"
		// A = {r0, r1}    B = {r27, r24}    C = {r20, r21}    D = {r10, r30}
		"xor.b32 r12, r40, 0x309911EB;\n\t"
		"xor.b32 r49, r41, 0x4F452FEC;\n\t"
		"add.cc.u32 r0, r0, r12;\n\t"
		"addc.u32 r1, r1, r49;\n\t"
		// A = {r0, r1}    B = {r27, r24}    C = {r20, r21}    D = {r10, r30}
		"xor.b32 r10, r10, r0;\n\t"
		"xor.b32 r30, r30, r1;\n\t"
		"shf.r.wrap.b32 r12, r10, r30, 5;\n\t"
		"shf.r.wrap.b32 r10, r30, r10, 5;\n\t"
		// A = {r0, r1}    B = {r27, r24}    C = {r20, r21}    D = {r12, r10}
		"add.cc.u32 r20, r20, r12;\n\t"
		"addc.u32 r21, r21, r10;\n\t"
		// A = {r0, r1}    B = {r27, r24}    C = {r20, r21}    D = {r12, r10}
		"xor.b32 r27, r27, r20;\n\t"
		"xor.b32 r24, r24, r21;\n\t"
		"shf.r.wrap.b32 r30, r27, r24, 18;\n\t"
		"shf.r.wrap.b32 r27, r24, r27, 18;\n\t"
		// A = {r0, r1}    B = {r30, r27}    C = {r20, r21}    D = {r12, r10}
		"lop3.b32 r24, r0, r30, r20, 0x01;\n\t"
		"lop3.b32 r49, r1, r27, r21, 0x01;\n\t"
		"lop3.b32 r50, r0, r30, r20, 0x08;\n\t"
		"lop3.b32 r51, r1, r27, r21, 0x08;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r20, 0x20;\n\t"
		"lop3.b32 r49, r1, r27, r21, 0x20;\n\t"
		"lop3.b32 r50, r0, r30, r20, 0x40;\n\t"
		"lop3.b32 r51, r1, r27, r21, 0x40;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r20, 0x02;\n\t"
		"lop3.b32 r49, r1, r27, r21, 0x02;\n\t"
		"lop3.b32 r50, r0, r30, r20, 0x04;\n\t"
		"lop3.b32 r51, r1, r27, r21, 0x04;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		"lop3.b32 r24, r0, r30, r20, 0x10;\n\t"
		"lop3.b32 r49, r1, r27, r21, 0x10;\n\t"
		"lop3.b32 r50, r0, r30, r20, 0x80;\n\t"
		"lop3.b32 r51, r1, r27, r21, 0x80;\n\t"
		"lop3.b32 r12, r12, r24, r50, 0x1E;\n\t"
		"lop3.b32 r10, r10, r49, r51, 0x1E;\n\t"
		// A = {r0, r1}    B = {r30, r27}    C = {r20, r21}    D = {r12, r10}
		/*
		* |------------------------[ROUND 15.5]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r30, r27}           |
		* |            v[ 6]            |           {r29, r14}           |
		* |            v[ 7]            |           {r31,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r11, r26}           |
		* |            v[13]            |           {r13, r28}           |
		* |            v[14]            |           {r15,  r9}           |
		* |            v[15]            |           {r12, r10}           |
		* |            temp0            |           {r24, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r26}
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r14;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r26}
		"xor.b32 r24, r42, 0x74E1022C;\n\t"
		"xor.b32 r49, r43, 0x3CFCC66F;\n\t"
		"add.cc.u32 r2, r24, r2;\n\t"
		"addc.u32 r3, r49, r3;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r26}
		"xor.b32 r11, r11, r2;\n\t"
		"xor.b32 r26, r26, r3;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r26}
		"shf.r.wrap.b32 r24, r11, r26, 60;\n\t"
		"shf.r.wrap.b32 r11, r26, r11, 60;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r24}
		"add.cc.u32 r22, r22, r11;\n\t"
		"addc.u32 r23, r23, r24;\n\t"
		// A = {r2, r3}    B = {r29, r14}    C = {r22, r23}    D = {r11, r24}
		"xor.b32 r29, r29, r22;\n\t"
		"xor.b32 r14, r14, r23;\n\t"
		"shf.r.wrap.b32 r26, r29, r14, 43;\n\t"
		"shf.r.wrap.b32 r29, r14, r29, 43;\n\t"
		// A = {r2, r3}    B = {r29, r26}    C = {r22, r23}    D = {r11, r24}
		"add.cc.u32 r2, r2, r29;\n\t"
		"addc.u32 r3, r3, r26;\n\t"
		// A = {r2, r3}    B = {r29, r26}    C = {r22, r23}    D = {r11, r24}
		"xor.b32 r14, r46, 0x3D47C800;\n\t"
		"xor.b32 r49, r47, 0xBBA055B5;\n\t"
		"add.cc.u32 r2, r2, r14;\n\t"
		"addc.u32 r3, r3, r49;\n\t"
		// A = {r2, r3}    B = {r29, r26}    C = {r22, r23}    D = {r11, r24}
		"xor.b32 r11, r11, r2;\n\t"
		"xor.b32 r24, r24, r3;\n\t"
		"shf.r.wrap.b32 r14, r11, r24, 5;\n\t"
		"shf.r.wrap.b32 r11, r24, r11, 5;\n\t"
		// A = {r2, r3}    B = {r29, r26}    C = {r22, r23}    D = {r14, r11}
		"add.cc.u32 r22, r22, r14;\n\t"
		"addc.u32 r23, r23, r11;\n\t"
		// A = {r2, r3}    B = {r29, r26}    C = {r22, r23}    D = {r14, r11}
		"xor.b32 r29, r29, r22;\n\t"
		"xor.b32 r26, r26, r23;\n\t"
		"shf.r.wrap.b32 r24, r29, r26, 18;\n\t"
		"shf.r.wrap.b32 r29, r26, r29, 18;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r22, r23}    D = {r14, r11}
		"lop3.b32 r26, r2, r24, r22, 0x01;\n\t"
		"lop3.b32 r49, r3, r29, r23, 0x01;\n\t"
		"lop3.b32 r50, r2, r24, r22, 0x08;\n\t"
		"lop3.b32 r51, r3, r29, r23, 0x08;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r22, 0x20;\n\t"
		"lop3.b32 r49, r3, r29, r23, 0x20;\n\t"
		"lop3.b32 r50, r2, r24, r22, 0x40;\n\t"
		"lop3.b32 r51, r3, r29, r23, 0x40;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r22, 0x02;\n\t"
		"lop3.b32 r49, r3, r29, r23, 0x02;\n\t"
		"lop3.b32 r50, r2, r24, r22, 0x04;\n\t"
		"lop3.b32 r51, r3, r29, r23, 0x04;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		"lop3.b32 r26, r2, r24, r22, 0x10;\n\t"
		"lop3.b32 r49, r3, r29, r23, 0x10;\n\t"
		"lop3.b32 r50, r2, r24, r22, 0x80;\n\t"
		"lop3.b32 r51, r3, r29, r23, 0x80;\n\t"
		"lop3.b32 r14, r14, r26, r50, 0x1E;\n\t"
		"lop3.b32 r11, r11, r49, r51, 0x1E;\n\t"
		// A = {r2, r3}    B = {r24, r29}    C = {r22, r23}    D = {r14, r11}
		/*
		* |------------------------[ROUND 15.6]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r30, r27}           |
		* |            v[ 6]            |           {r24, r29}           |
		* |            v[ 7]            |           {r31,  r8}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r11}           |
		* |            v[13]            |           {r13, r28}           |
		* |            v[14]            |           {r15,  r9}           |
		* |            v[15]            |           {r12, r10}           |
		* |            temp0            |           {r26, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r28}
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r8;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r28}
		"xor.b32 r26, 0x00, 0x81AAE000;\n\t"
		"xor.b32 r49, 0x00, 0xD859E6F0;\n\t"
		"add.cc.u32 r4, r26, r4;\n\t"
		"addc.u32 r5, r49, r5;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r28}
		"xor.b32 r13, r13, r4;\n\t"
		"xor.b32 r28, r28, r5;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r28}
		"shf.r.wrap.b32 r26, r13, r28, 60;\n\t"
		"shf.r.wrap.b32 r13, r28, r13, 60;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r26}
		"add.cc.u32 r16, r16, r13;\n\t"
		"addc.u32 r17, r17, r26;\n\t"
		// A = {r4, r5}    B = {r31, r8}    C = {r16, r17}    D = {r13, r26}
		"xor.b32 r31, r31, r16;\n\t"
		"xor.b32 r8, r8, r17;\n\t"
		"shf.r.wrap.b32 r28, r31, r8, 43;\n\t"
		"shf.r.wrap.b32 r31, r8, r31, 43;\n\t"
		// A = {r4, r5}    B = {r31, r28}    C = {r16, r17}    D = {r13, r26}
		"add.cc.u32 r4, r4, r31;\n\t"
		"addc.u32 r5, r5, r28;\n\t"
		// A = {r4, r5}    B = {r31, r28}    C = {r16, r17}    D = {r13, r26}
		"xor.b32 r8, 0x00, 0x7B560E6B;\n\t"
		"xor.b32 r49, 0x00, 0x63D98059;\n\t"
		"add.cc.u32 r4, r4, r8;\n\t"
		"addc.u32 r5, r5, r49;\n\t"
		// A = {r4, r5}    B = {r31, r28}    C = {r16, r17}    D = {r13, r26}
		"xor.b32 r13, r13, r4;\n\t"
		"xor.b32 r26, r26, r5;\n\t"
		"shf.r.wrap.b32 r8, r13, r26, 5;\n\t"
		"shf.r.wrap.b32 r13, r26, r13, 5;\n\t"
		// A = {r4, r5}    B = {r31, r28}    C = {r16, r17}    D = {r8, r13}
		"add.cc.u32 r16, r16, r8;\n\t"
		"addc.u32 r17, r17, r13;\n\t"
		// A = {r4, r5}    B = {r31, r28}    C = {r16, r17}    D = {r8, r13}
		"xor.b32 r31, r31, r16;\n\t"
		"xor.b32 r28, r28, r17;\n\t"
		"shf.r.wrap.b32 r26, r31, r28, 18;\n\t"
		"shf.r.wrap.b32 r31, r28, r31, 18;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r16, r17}    D = {r8, r13}
		"lop3.b32 r28, r4, r26, r16, 0x01;\n\t"
		"lop3.b32 r49, r5, r31, r17, 0x01;\n\t"
		"lop3.b32 r50, r4, r26, r16, 0x08;\n\t"
		"lop3.b32 r51, r5, r31, r17, 0x08;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r16, 0x20;\n\t"
		"lop3.b32 r49, r5, r31, r17, 0x20;\n\t"
		"lop3.b32 r50, r4, r26, r16, 0x40;\n\t"
		"lop3.b32 r51, r5, r31, r17, 0x40;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r16, 0x02;\n\t"
		"lop3.b32 r49, r5, r31, r17, 0x02;\n\t"
		"lop3.b32 r50, r4, r26, r16, 0x04;\n\t"
		"lop3.b32 r51, r5, r31, r17, 0x04;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		"lop3.b32 r28, r4, r26, r16, 0x10;\n\t"
		"lop3.b32 r49, r5, r31, r17, 0x10;\n\t"
		"lop3.b32 r50, r4, r26, r16, 0x80;\n\t"
		"lop3.b32 r51, r5, r31, r17, 0x80;\n\t"
		"lop3.b32 r8, r8, r28, r50, 0x1E;\n\t"
		"lop3.b32 r13, r13, r49, r51, 0x1E;\n\t"
		// A = {r4, r5}    B = {r26, r31}    C = {r16, r17}    D = {r8, r13}
		/*
		* |------------------------[ROUND 15.7]--------------------------|
		* |        **VARIABLE**         |       **REGISTER PAIR**        |
		* |- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - |
		* |            v[ 0]            |           { r0,  r1}           |
		* |            v[ 1]            |           { r2,  r3}           |
		* |            v[ 2]            |           { r4,  r5}           |
		* |            v[ 3]            |           { r6,  r7}           |
		* |            v[ 4]            |           {r25, r48}           |
		* |            v[ 5]            |           {r30, r27}           |
		* |            v[ 6]            |           {r24, r29}           |
		* |            v[ 7]            |           {r26, r31}           |
		* |            v[ 8]            |           {r16, r17}           |
		* |            v[ 9]            |           {r18, r19}           |
		* |            v[10]            |           {r20, r21}           |
		* |            v[11]            |           {r22, r23}           |
		* |            v[12]            |           {r14, r11}           |
		* |            v[13]            |           { r8, r13}           |
		* |            v[14]            |           {r15,  r9}           |
		* |            v[15]            |           {r12, r10}           |
		* |            temp0            |           {r28, r49}           |
		* |            temp1            |           {r50, r51}           |
		* |--------------------------------------------------------------|
		*/
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r9}
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r48;\n\t"
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r9}
		"xor.b32 r28, 0x00, 0xDAE5B800;\n\t"
		"xor.b32 r49, 0x00, 0xD1A00BA6;\n\t"
		"add.cc.u32 r6, r28, r6;\n\t"
		"addc.u32 r7, r49, r7;\n\t"
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r9}
		"xor.b32 r15, r15, r6;\n\t"
		"xor.b32 r9, r9, r7;\n\t"
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r9}
		"shf.r.wrap.b32 r28, r15, r9, 60;\n\t"
		"shf.r.wrap.b32 r15, r9, r15, 60;\n\t"
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r28}
		"add.cc.u32 r18, r18, r15;\n\t"
		"addc.u32 r19, r19, r28;\n\t"
		// A = {r6, r7}    B = {r25, r48}    C = {r18, r19}    D = {r15, r28}
		"xor.b32 r25, r25, r18;\n\t"
		"xor.b32 r48, r48, r19;\n\t"
		"shf.r.wrap.b32 r9, r25, r48, 43;\n\t"
		"shf.r.wrap.b32 r25, r48, r25, 43;\n\t"
		// A = {r6, r7}    B = {r25, r9}    C = {r18, r19}    D = {r15, r28}
		"add.cc.u32 r6, r6, r25;\n\t"
		"addc.u32 r7, r7, r9;\n\t"
		// A = {r6, r7}    B = {r25, r9}    C = {r18, r19}    D = {r15, r28}
		"xor.b32 r48, r34, 0x0B723800;\n\t"
		"xor.b32 r49, r35, 0xD35B2E0E;\n\t"
		"add.cc.u32 r6, r6, r48;\n\t"
		"addc.u32 r7, r7, r49;\n\t"
		// A = {r6, r7}    B = {r25, r9}    C = {r18, r19}    D = {r15, r28}
		"xor.b32 r15, r15, r6;\n\t"
		"xor.b32 r28, r28, r7;\n\t"
		"shf.r.wrap.b32 r48, r15, r28, 5;\n\t"
		"shf.r.wrap.b32 r15, r28, r15, 5;\n\t"
		// A = {r6, r7}    B = {r25, r9}    C = {r18, r19}    D = {r48, r15}
		"add.cc.u32 r18, r18, r48;\n\t"
		"addc.u32 r19, r19, r15;\n\t"
		// A = {r6, r7}    B = {r25, r9}    C = {r18, r19}    D = {r48, r15}
		"xor.b32 r25, r25, r18;\n\t"
		"xor.b32 r9, r9, r19;\n\t"
		"shf.r.wrap.b32 r28, r25, r9, 18;\n\t"
		"shf.r.wrap.b32 r25, r9, r25, 18;\n\t"
		// A = {r6, r7}    B = {r28, r25}    C = {r18, r19}    D = {r48, r15}
		"lop3.b32 r9, r6, r28, r18, 0x01;\n\t"
		"lop3.b32 r49, r7, r25, r19, 0x01;\n\t"
		"lop3.b32 r50, r6, r28, r18, 0x08;\n\t"
		"lop3.b32 r51, r7, r25, r19, 0x08;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r18, 0x20;\n\t"
		"lop3.b32 r49, r7, r25, r19, 0x20;\n\t"
		"lop3.b32 r50, r6, r28, r18, 0x40;\n\t"
		"lop3.b32 r51, r7, r25, r19, 0x40;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r18, 0x02;\n\t"
		"lop3.b32 r49, r7, r25, r19, 0x02;\n\t"
		"lop3.b32 r50, r6, r28, r18, 0x04;\n\t"
		"lop3.b32 r51, r7, r25, r19, 0x04;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		"lop3.b32 r9, r6, r28, r18, 0x10;\n\t"
		"lop3.b32 r49, r7, r25, r19, 0x10;\n\t"
		"lop3.b32 r50, r6, r28, r18, 0x80;\n\t"
		"lop3.b32 r51, r7, r25, r19, 0x80;\n\t"
		"lop3.b32 r48, r48, r9, r50, 0x1E;\n\t"
		"lop3.b32 r15, r15, r49, r51, 0x1E;\n\t"
		// A = {r6, r7}    B = {r28, r25}    C = {r18, r19}    D = {r48, r15}
		// v[0] = {r0, r1}
		// v[1] = {r2, r3}
		// v[2] = {r4, r5}
		// v[3] = {r6, r7}
		// v[4] = {r28, r25}
		// v[5] = {r30, r27}
		// v[6] = {r24, r29}
		// v[7] = {r26, r31}
		// v[8] = {r16, r17}
		// v[9] = {r18, r19}
		// v[10] = {r20, r21}
		// v[11] = {r22, r23}
		// v[12] = {r14, r11}
		// v[13] = {r8, r13}
		// v[14] = {r48, r15}
		// v[15] = {r12, r10}
		// header[0] = {r32, r33}
		// header[1] = {r34, r35}
		// header[2] = {r36, r37}
		// header[3] = {r38, r39}
		// header[4] = {r40, r41}
		// header[5] = {r42, r43}
		// header[6] = {r44, r45}
		// header[7] = {r46, r47}
		"lop3.b32 r32, 0xF107AD85, r0, r16, 0x96;\n\t"
		"lop3.b32 r33, 0x4BBF42C1, r1, r17, 0x96;\n\t"
		"lop3.b32 r38, 0x4658F253, r6, r22, 0x96;\n\t"
		"lop3.b32 r39, 0xC6759572, r7, r23, 0x96;\n\t"
		"lop3.b32 r44, 0x3C60BAA8, r24, r48, 0x96;\n\t"
		"lop3.b32 r45, 0xB1DA3AB6, r29, r15, 0x96;\n\t"
		"lop3.b32 r32, r32, r38, r44, 0x96;\n\t"
		"lop3.b32 r33, r33, r39, r45, 0x96;\n\t"
		"mov.b64 %0, {r32, r33};\n\t}"

		: "=l"(result) : "l"(h0), "l"(h1), "l"(h2), "l"(h3), "l"(h4), "l"(h5), "l"(h6), "l"(h7));

	return result;
}

#if CPU_SHARES
#define WORK_PER_THREAD 256
#else
#define WORK_PER_THREAD 256
#endif

#if HIGH_RESOURCE
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 1024
#else
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 512
#endif

int blocksize = DEFAULT_BLOCKSIZE;
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;

bool verboseOutput = false;

/*
* Kernel function to search a range of nonces for a solution falling under the macro-configured difficulty (CPU=2^24, GPU=2^32).
*/
__global__ void vblakeHasher(uint32_t *nonceStart, uint32_t *nonceOut, uint64_t *hashStartOut, uint64_t const *headerIn)
{
	// Generate a unique starting nonce for each thread that doesn't overlap with the work of any other thread
	const uint32_t workStart = ((blockDim.x * blockIdx.x + threadIdx.x)  * WORK_PER_THREAD) + nonceStart[0];

	uint64_t nonceHeaderSection = headerIn[7];

	// Run the hash WORK_PER_THREAD times
	for (unsigned int nonce = workStart; nonce < workStart + WORK_PER_THREAD; nonce++) {
		// Zero out nonce position and write new nonce to last 32 bits of prototype header
		nonceHeaderSection &= 0x00000000FFFFFFFFu;
		nonceHeaderSection |= (((uint64_t)nonce) << 32);

		uint64_t hashStart = vBlake(headerIn[0], headerIn[1], headerIn[2], headerIn[3], headerIn[4], headerIn[5], headerIn[6], nonceHeaderSection);

		if ((hashStart &

#if CPU_SHARES
			0x0000000000FFFFFFu // 2^24 difficulty
#else
			0x00000000FFFFFFFFu // 2^32 difficulty
#endif
			) == 0) {
			// Check that found solution is better than existing solution if one has already been found on this run of the kernel (always send back highest-quality work)
			if (hashStartOut[0] > hashStart || hashStartOut[0] == 0) {
				nonceOut[0] = nonce;
				hashStartOut[0] = hashStart;
			}

			// exit loop early
			nonce = workStart + WORK_PER_THREAD;
		}
	}
}

void promptExit(int exitCode)
{
	cout << "Exiting in 10 seconds..." << endl;
	std::this_thread::sleep_for(std::chrono::milliseconds(10000));
	exit(exitCode);
}

/**
* Takes the provided timestamp and places it in the header
*/
void embedTimestampInHeader(uint8_t *header, uint32_t timestamp)
{
	header[55] = (timestamp & 0x000000FF);
	header[54] = (timestamp & 0x0000FF00) >> 8;
	header[53] = (timestamp & 0x00FF0000) >> 16;
	header[52] = (timestamp & 0xFF000000) >> 24;
}

/**
* Returns a 64-byte header to attempt to mine with.
*/
uint64_t* getWork(UCPClient& ucpClient, uint32_t timestamp)
{
	uint64_t *header = new uint64_t[8];
	ucpClient.copyHeaderToHash((byte *)header);
	embedTimestampInHeader((uint8_t*)header, timestamp);
	return header;
}

int deviceToUse = 0;

#if NVML
nvmlDevice_t device;
void readyNVML(int deviceIndex) {
	nvmlInit();
	nvmlDeviceGetHandleByIndex(deviceIndex, &device);
}
int getTemperature() {
	unsigned int temperature;
	nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
	return temperature;
}

int getCoreClock() {
	unsigned int clock;
	nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &clock);
	return clock;
}

int getMemoryClock() {
	unsigned int memClock;
	nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &memClock);
	return memClock;
}
#else
void readyNVML(int deviceIndex) {
	// Do Nothing
}

int getTemperature() {
	return -1;
}

int getCoreClock() {
	return -1;
}

int getMemoryClock() {
	return -1;
}
#endif

#define SHARE_SUBMISSION_NO_RESPONSE_WARN_THRESHOLD 50

void vprintf(char* toprint) {
	if (verboseOutput) {
		printf(toprint);
	}
}

void printHelpAndExit() {
	printf("VeriBlock vBlake GPU CUDA Miner v1.0\n");
	printf("Required Arguments:\n");
	printf("-o <poolAddress>           The pool address to mine to in the format host:port\n");
	printf("-u <username>              The username (often an address) used at the pool\n");
	printf("Optional Arguments:\n");
	printf("-p <password>              The miner/worker password to use on the pool\n");
	printf("-d <deviceNum>             The ordinal of the device to use (default 0)\n");
	printf("-tpb <threadPerBlock>      The threads per block to use with the Blake kernel (default %d)\n", DEFAULT_THREADS_PER_BLOCK);
	printf("-bs <blockSize>            The blocksize to use with the vBlake kernel (default %d)\n", DEFAULT_BLOCKSIZE);
	printf("-l <enableLogging>         Whether to log to a file (default true)\n");
	printf("-v <enableVerboseOutput>   Whether to enable verbose output for debugging (default false)\n");
	printf("\n");
	printf("Example command line:\n");
	printf("VeriBlock-NodeCore-PoW-CUDA -u VHT36jJyoVFN7ap5Gu77Crua2BMv5j -o testnet-pool-gpu.veriblock.org:8501 -l false\n");
	promptExit(0);
}

#ifdef _WIN32
static WSADATA g_wsa_data;
#endif

char net_init(void)
{
#ifdef _WIN32
	return (WSAStartup(MAKEWORD(2, 2), &g_wsa_data) == NO_ERROR);
#elif __linux__
	return 1;
#endif
}

void net_deinit(void)
{
#ifdef _WIN32
	WSACleanup();
#endif
}

string net_dns_resolve(const char* hostname)
{
	struct addrinfo hints, *results, *item;
	int status;
	char ipstr[INET6_ADDRSTRLEN];

	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;  /* AF_INET6 to force version */
	hints.ai_socktype = SOCK_STREAM;

	if ((status = getaddrinfo(hostname, NULL, &hints, &results)) != 0)
	{
		fprintf(stderr, "failed to resolve hostname \"%s\": %s", hostname, gai_strerror(status));
		return "invalid hostname";
	}

	printf("IP addresses for %s:\n\n", hostname);

	string ret;

	for (item = results; item != NULL; item = item->ai_next)
	{
		void* addr;
		char* ipver;

		/* get pointer to the address itself */
		/* different fields in IPv4 and IPv6 */
		if (item->ai_family == AF_INET)  /* address is IPv4 */
		{
			struct sockaddr_in* ipv4 = (struct sockaddr_in*)item->ai_addr;
			addr = &(ipv4->sin_addr);
			ipver = "IPv4";
		}
		else  /* address is IPv6 */
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

char outputBuffer[8192];
int main(int argc, char *argv[])
{
	// Check for help argument (only -h)
	for (int i = 1; i < argc; i++) {
		char* argument = argv[i];

		if (!strcmp(argument, "-h"))
		{
			printHelpAndExit();
		}
	}

	if (argc % 2 != 1) {
		sprintf(outputBuffer, "GPU miner must be provided valid argument pairs!");
		cerr << outputBuffer << endl;
		printHelpAndExit();
	}

	string hostAndPort = ""; //  "94.130.64.18:8501";
	string username = ""; // "VGX71bcRsEh4HZzhbA9Nj7GQNH5jGw";
	string password = "";

	if (argc > 1)
	{
		for (int i = 1; i < argc; i += 2)
		{
			char* argument = argv[i];
			printf("%s\n", argument);
			if (argument[0] == '-' && argument[1] == 'd')
			{
				if (strlen(argv[i + 1]) == 2) {
					// device num >= 10
					deviceToUse = (argv[i + 1][0] - 48) * 10 + (argv[i + 1][1] - 48);
				}
				else {
					deviceToUse = argv[i + 1][0] - 48;
				}
			}
			else if (!strcmp(argument, "-o"))
			{
				hostAndPort = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-u"))
			{
				username = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-p"))
			{
				password = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-tpb"))
			{
				threadsPerBlock = stoi(argv[i + 1]);
			}
			else if (!strcmp(argument, "-bs"))
			{
				blocksize = stoi(argv[i + 1]);
			}
			else if (!strcmp(argument, "-l"))
			{
				// to lower case conversion
				for (int j = 0; j < strlen(argv[i + 1]); j++)
				{
					argv[i + 1][j] = tolower(argv[i + 1][j]);
				}
				if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t"))
				{
					Log::setEnabled(true);
				}
				else
				{
					Log::setEnabled(false);
				}
			}
			else if (!strcmp(argument, "-v"))
			{
				// to lower case conversion
				for (int j = 0; j < strlen(argv[i + 1]); j++)
				{
					argv[i + 1][j] = tolower(argv[i + 1][j]);
				}
				if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t"))
				{
					verboseOutput = true;
				}
				else
				{
					verboseOutput = false;
				}
			}
		}
	}
	else {
		printHelpAndExit();
	}

	if (HIGH_RESOURCE) {
		sprintf(outputBuffer, "Resource Utilization: HIGH");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Resource Utilization: LOW");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	if (NVML) {
		sprintf(outputBuffer, "NVML Status: ENABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "NVML Status: DISABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	if (CPU_SHARES) {
		sprintf(outputBuffer, "Share Type: CPU");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Share Type: GPU");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	if (BENCHMARK) {
		sprintf(outputBuffer, "Benchmark Mode: ENABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Benchmark Mode: DISABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	// No effect if NVML is not enabled
	readyNVML(deviceToUse);

#ifdef _WIN32
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
#else
#endif

	if (hostAndPort.compare("") == 0) {
		string error = "You must specify a host in the command line arguments! Example: \n-o 127.0.0.1:8501 or localhost:8501";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	if (username.compare("") == 0) {
		string error = "You must specify a username in the command line arguments! Example: \n-u V5bLSbCqj9VzQR3MNANqL13YC2tUep";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	string host = hostAndPort.substr(0, hostAndPort.find(":"));
	//GetHostByName
	net_init();
	host = net_dns_resolve(host.c_str());
	net_deinit();

	string portString = hostAndPort.substr(hostAndPort.find(":") + 1);

	// Ensure that port is numeric
	if (portString.find_first_not_of("1234567890") != string::npos) {
		string error = "You must specify a host in the command line arguments! Example: \n-o 127.0.0.1:8501 or localhost:8501";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	int port = stoi(portString);

	sprintf(outputBuffer, "Attempting to mine to pool %s:%d with username %s and password %s...", host.c_str(), port, username.c_str(), password.c_str());
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	UCPClient ucpClient(host, port, username, password);

	byte target[24];
	ucpClient.copyMiningTarget(target);

	sprintf(outputBuffer, "Using Device: %d\n\n", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	int version, ret;
	ret = cudaDriverGetVersion(&version);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA driver version: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	int runtimeVersion;
	ret = cudaRuntimeGetVersion(&runtimeVersion);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA runtime version: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}


	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA device count: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	cudaDeviceProp deviceProp;

#if NVML
	char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
	nvmlSystemGetDriverVersion(driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
#else
	char driver[] = "???.?? (NVML NOT ENABLED)";
#endif

	sprintf(outputBuffer, "CUDA Version: %.1f", ((float)version / 1000));
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "CUDA Runtime Version: %d", runtimeVersion);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "NVidia Driver Version: %s", driver);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "CUDA Devices: %d", deviceCount);
	cout << outputBuffer << endl << endl;
	Log::info(outputBuffer);

	string selectedDeviceName;
	// Print out information about all available CUDA devices on system
	for (int count = 0; count < deviceCount; count++)
	{
		ret = cudaGetDeviceProperties(&deviceProp, count);
		if (ret != cudaSuccess)
		{
			sprintf(outputBuffer, "An error occurred while getting the CUDA device properties: %d", ret);
			cerr << outputBuffer << endl;
			Log::error(outputBuffer);
		}

		if (count == deviceToUse) {
			selectedDeviceName = deviceProp.name;
		}

		sprintf(outputBuffer, "Device #%d (%s):", count, deviceProp.name);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Clock Rate:              %d MHz", (deviceProp.clockRate / 1024));
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Is Integrated:           %s", (deviceProp.integrated == 0 ? "false" : "true"));
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Compute Capability:      %d.%d", deviceProp.major, deviceProp.minor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Kernel Concurrency:      %d", deviceProp.concurrentKernels);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Max Grid Size:           %d x %d x %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Max Threads per Block:   %d", deviceProp.maxThreadsPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Registers per Block:     %d", deviceProp.regsPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Registers per SM:        %d", deviceProp.regsPerMultiprocessor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Processor Count:         %d", deviceProp.multiProcessorCount);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Shared Memory/Block:     %zd", deviceProp.sharedMemPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Shared Memory/Proc:      %zd", deviceProp.sharedMemPerMultiprocessor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Warp Size:               %d", deviceProp.warpSize);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	sprintf(outputBuffer, "Mining on device #%d...", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	ret = cudaSetDevice(deviceToUse);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "CUDA encountered an error while setting the device to %d:%d", deviceToUse, ret);
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
	}

	cudaDeviceReset();

	// Don't have GPU busy-wait on GPU
	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

	cudaError_t e = cudaGetLastError();
	sprintf(outputBuffer, "Last error: %s\n", cudaGetErrorString(e));
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	// Run initialization of device before beginning timer
	uint64_t* header = getWork(ucpClient, (uint32_t)std::time(0));

	unsigned long long startTime = std::time(0);
	uint32_t nonceResult[1] = { 0 };
	uint64_t hashStart[1] = { 0 };

	unsigned long long hashes = 0;
	cudaError_t cudaStatus;

	uint32_t count = 0;

	int numLines = 0;

	// Mining loop
	while (true) {
		vprintf("top of mining loop\n");
		count++;
		long timestamp = (long)std::time(0);
		delete[] header;
		vprintf("Getting work...\n");
		header = getWork(ucpClient, timestamp);
		vprintf("Getting job id...\n");
		int jobId = ucpClient.getJobId();
		count++;
		vprintf("Running kernel...\n");
		cudaStatus = grindNonces(nonceResult, hashStart, header);
		vprintf("Kernel finished...\n");
		if (cudaStatus != cudaSuccess) {
			cudaError_t e = cudaGetLastError();
			sprintf(outputBuffer, "Error from running grindNonces: %s\nThis often occurs when a GPU overheats, has an unstable overclock, or has too aggressive launch parameters\nfor the vBlake kernel.\nYou can try using less aggressive settings, like:\n-tpb 256 -bs 256\nAnd try increasing these numbers until you hit instability issues again.", cudaGetErrorString(e));
			cerr << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}

		unsigned long long totalTime = std::time(0) - startTime;
		hashes += (blocksize * threadsPerBlock * WORK_PER_THREAD);

		double hashSpeed = (double)hashes;
		hashSpeed /= (totalTime * 1024 * 1024);

		if (count % 10 == 0) {
			int validShares = ucpClient.getValidShares();
			int invalidShares = ucpClient.getInvalidShares();
			int totalAccountedForShares = invalidShares + validShares;
			int totalSubmittedShares = ucpClient.getSentShares();
			int unaccountedForShares = totalSubmittedShares - totalAccountedForShares;

			double percentage = ((double)validShares) / totalAccountedForShares;
			percentage *= 100;
			// printf("[GPU #%d (%s)] : %f MH/second    valid shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);

			printf("[GPU #%d (%s)] : %0.2f MH/s shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);
		}

		if (nonceResult[0] != 0x01000000 && nonceResult[0] != 0) {
			uint32_t nonce = *nonceResult;
			nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) | ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));

			ucpClient.submitWork(jobId, timestamp, nonce);

			nonceResult[0] = 0;

			char line[100];

			// Hash coming from GPU is reversed
			uint64_t hashFlipped = 0;
			hashFlipped |= (hashStart[0] & 0x00000000000000FF) << 56;
			hashFlipped |= (hashStart[0] & 0x000000000000FF00) << 40;
			hashFlipped |= (hashStart[0] & 0x0000000000FF0000) << 24;
			hashFlipped |= (hashStart[0] & 0x00000000FF000000) << 8;
			hashFlipped |= (hashStart[0] & 0x000000FF00000000) >> 8;
			hashFlipped |= (hashStart[0] & 0x0000FF0000000000) >> 24;
			hashFlipped |= (hashStart[0] & 0x00FF000000000000) >> 40;
			hashFlipped |= (hashStart[0] & 0xFF00000000000000) >> 56;

#if CPU_SHARES 
			sprintf(line, "\t Share Found @ 2^24! {%#018llx} [nonce: %#08lx]", hashFlipped, nonce);
#else
			sprintf(line, "\t Share Found @ 2^32! {%#018llx} [nonce: %#08lx]", hashFlipped, nonce);
#endif

			cout << line << endl;
			vprintf("Logging\n");
			Log::info(line);
			vprintf("Done logging\n");
			vprintf("Made line\n");

			numLines++;

			// Uncomment these lines to get access to this data for display purposes
			/*
			long long extraNonce = ucpClient.getStartExtraNonce();
			int jobId = ucpClient.getJobId();
			int encodedDifficulty = ucpClient.getEncodedDifficulty();
			string previousBlockHashHex = ucpClient.getPreviousBlockHash();
			string merkleRoot = ucpClient.getMerkleRoot();
			*/

		}
		vprintf("About to restart loop...\n");
	}

	printf("Resetting device...\n");
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	printf("Done resetting device...\n");

	getchar();
	return 0;
}

uint32_t lastNonceStart = 0;

// Grind Through vBlake nonces with the provided header, setting the resultant nonce and associated hash start if a high-difficulty solution is found
cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header)
{
	// Device memory
	uint32_t *dev_nonceStart = 0;
	uint64_t *dev_header = 0;
	uint32_t *dev_nonceResult = 0;
	uint64_t *dev_hashStart = 0;

	// Ensure that nonces don't overlap previous work
	uint32_t nonceStart = (uint64_t)lastNonceStart + (WORK_PER_THREAD * blocksize * threadsPerBlock);
	lastNonceStart = nonceStart;

	cudaError_t cudaStatus;

	// Select GPU to run on
	cudaStatus = cudaSetDevice(deviceToUse);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header
	cudaStatus = cudaMalloc((void**)&dev_nonceStart, 1 * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy starting nonce to GPU
	cudaStatus = cudaMemcpy(dev_nonceStart, &nonceStart, sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header.
	cudaStatus = cudaMalloc((void**)&dev_nonceResult, 1 * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header.
	cudaStatus = cudaMalloc((void**)&dev_hashStart, 1 * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_header, 8 * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_header, header, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	vblakeHasher << < blocksize, threadsPerBlock >> >(dev_nonceStart, dev_nonceResult, dev_hashStart, dev_header);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "grindNonces launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaDeviceSynchronize returned error code %d after launching grindNonces!\n", cudaStatus);
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(nonceResult, dev_nonceResult, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hashStart, dev_hashStart, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

Error:
	cudaFree(dev_nonceStart);
	cudaFree(dev_header);
	cudaFree(dev_nonceResult);
	cudaFree(dev_hashStart);
	return cudaStatus;
}
